import logging
from abc import ABC, abstractmethod
from typing import Tuple, List, Iterable
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


def mark_keywords2(text: str, keywords: Iterable[str], marker: str = "[KEY]") -> str:
    """
    Marks every occurrence of a word from keywords with a marker (e.g., [KEY] word [KEY]).
    Works with punctuation and avoids substituting fragments of other words.

    Args:
        text: The input string to search within.
        keywords: An iterable of strings representing the keywords to mark.
        marker: The string to use as a marker around keywords (default is "[KEY]").

    Returns:
        A string with the keywords marked by the specified marker.
    """

    def repl(match):
        w = match.group(0)
        return f"{marker} {w} {marker}"

    keywords_sorted = sorted(set(keywords), key=len, reverse=True)
    for kw in keywords_sorted:
        if not kw:
            continue
        pattern = r"\b" + re.escape(kw) + r"\b"
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


# def preprocess_batch(
#     examples: Dict, tokenizer: PreTrainedTokenizer, keywords: List[str]
# ) -> Dict:
#     """
#     Example preprocessing: concatenate Journal and Abstract, mark keywords, tokenize.
#     """
#     combined_texts = []
#     for journal, abstract in zip(
#         examples.get("Journal", []), examples.get("Abstract", [])
#     ):
#         text = f"{journal} [JOURNAL_END] {abstract}"
#         # Assuming a function mark_keywords
#         text = mark_keywords(text, keywords)
#         combined_texts.append(text)
#     return tokenizer(
#         combined_texts,
#         truncation=True,
#         padding="max_length",
#         max_length=350,
#     )

# def preprocess_batch_titles(examples, tokenizer, keywords: list[str]):
#     titles = [mark_keywords(title, keywords) for title in examples["Title"]]
#     return tokenizer(
#         titles,
#         truncation=True,
#         padding="max_length",
#         max_length=32,
#     )


def preprocess_batch_abstract(examples, tokenizer, keywords, max_length):
    marked = [mark_keywords(a, keywords) for a in examples["Abstract"]]
    tok = tokenizer(
        marked,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tok["Abstract2"] = marked
    return tok


def preprocess_batch_titles(examples, tokenizer, keywords: list[str], max_length):
    marked = [mark_keywords(t, keywords) for t in examples["Title"]]
    tok = tokenizer(
        marked,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tok["Title2"] = marked
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
