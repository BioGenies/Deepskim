import logging
from abc import ABC, abstractmethod
from typing import Tuple, List, Union
import pandas as pd
from datasets import Dataset, DatasetDict
import re

from transformers import PreTrainedTokenizer


def mark_keywords(text: str, keywords: list[str], marker: str = "[KEY]") -> str:
    """For each word in keywords, surrounds it with the marker."""
    for kw in keywords:
        # \b – word boundaries, re.IGNORECASE – case-insensitive
        pattern = rf"\b{re.escape(kw)}\b"
        text = re.sub(pattern, f"{marker} {kw} {marker}", text, flags=re.IGNORECASE)
    return text


# def preprocess_batch_titles(examples, tokenizer, keywords: list[str], max_length):
#     marked = [mark_keywords(t, keywords) for t in examples["Title"]]
#     tok = tokenizer(
#         marked,
#         truncation=True,
#         padding="max_length",
#         max_length=max_length,
#     )
#     tok["Title2"] = marked
#     return tok

# def preprocess_batch_abstract(examples, tokenizer, keywords, max_length):
#     marked = [mark_keywords(a, keywords) for a in examples["Abstract"]]
#     tok = tokenizer(
#         marked,
#         truncation=True,
#         padding="max_length",
#         max_length=max_length,
#     )
#     tok["Abstract2"] = marked
#     return tok


def preprocess_batch_abstract(examples, tokenizer, keywords, max_length):
    raw_input = [
        f"{t} [T_END] {j} [J_END] {a}"
        for t, j, a in zip(examples["Title"], examples["Journal"], examples["Abstract"])
    ]

    marked_input = [mark_keywords(text, keywords) for text in raw_input]

    tok = tokenizer(
        marked_input,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    tok["input_text"] = marked_input

    return tok


class DatasetConverter(ABC):
    @abstractmethod
    def convert(
        self,
        splits,
        tokenizer: PreTrainedTokenizer,
        keywords: List[str],
        max_length: int = 350,
    ):
        """
        Convert raw splits into HuggingFace Datasets for Trainer.
        """
        pass


class TrainTestConverter(DatasetConverter):
    def convert(
        self,
        splits: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        tokenizer: PreTrainedTokenizer,
        keywords: List[str],
        max_length: int = 350,
    ) -> DatasetDict:
        """
        Convert a train-test split into an HF DatasetDict with 'train' and 'test' splits.
        The target values are stored in the 'labels' column as expected by transformers.Trainer.
        """
        X_train, X_test, y_train, y_test = splits

        # Prepare DataFrames, assigning labels directly
        train_df = X_train.copy()
        train_df["labels"] = y_train.values
        test_df = X_test.copy()
        test_df["labels"] = y_test.values

        # Create a DatasetDict from pandas DataFrames
        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df),
                "test": Dataset.from_pandas(test_df),
            }
        )

        dataset = dataset.map(
            lambda batch: preprocess_batch_abstract(
                batch, tokenizer, keywords, max_length
            ),
            batched=True,
        )

        logging.info(
            "Converted train-test split into a single DatasetDict with 'train' and 'test' Datasets."
        )
        return dataset


class FoldsConverter(DatasetConverter):
    def __init__(self, base_converter: TrainTestConverter):
        """
        base_converter: instance of TrainTestConverter,
                        which can convert a single split.
        """
        self.base = base_converter

    def convert(
        self,
        folds: List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
        tokenizer: PreTrainedTokenizer,
        keywords: List[str],
        max_length: int = 350,
    ) -> List[DatasetDict]:
        """
        For each fold (X_train, X_test, y_train, y_test)
        calls base_converter.convert(...) and returns a list
        of ready-to-use DatasetDicts.
        """
        all_datasets = []
        for idx, split in enumerate(folds):
            ds = self.base.convert(split, tokenizer, keywords, max_length)
            logging.info(f"Fold {idx}: conversion finished.")
            all_datasets.append(ds)
        return all_datasets


class DataFrameConverter(DatasetConverter):
    def convert(
        self,
        splits: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        tokenizer: PreTrainedTokenizer,
        keywords: List[str],
        max_length: int = 350,
    ) -> DatasetDict:
        """
        Convert a single DataFrame (or (X, y)) into a HuggingFace DatasetDict.
        - If `splits` is a tuple (X, y) we attach y as 'labels'.
        - If `splits` is a DataFrame and it lacks 'labels', we add 'labels' filled with -1
          (marker for "no label"; Trainer.predict will still work).
        Returns: DatasetDict({"test": Dataset})
        """
        # Normalize input -> get a DataFrame that always has 'labels' column
        if isinstance(splits, tuple):
            X, y = splits
            df = X.copy()
            df["labels"] = y.values
            has_labels = True
        elif isinstance(splits, pd.DataFrame):
            df = splits.copy()
            if "labels" in df.columns:
                has_labels = True
            else:
                # No labels present: create a placeholder column with -1
                df["labels"] = -1
                has_labels = False
        else:
            raise TypeError(
                "splits must be a pandas.DataFrame or a (DataFrame, Series) tuple"
            )

        # Create DatasetDict with single split 'all'
        dataset = DatasetDict({"test": Dataset.from_pandas(df)})

        dataset = dataset.map(
            lambda batch: preprocess_batch_abstract(
                batch, tokenizer, keywords, max_length
            ),
            batched=True,
        )

        logging.info(
            "Converted DataFrame into DatasetDict with single split 'test'. has_labels=%s",
            has_labels,
        )
        return dataset
