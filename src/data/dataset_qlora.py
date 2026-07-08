import pandas as pd
from transformers import AutoTokenizer
import logging

from data.data_splitter import DataSplitter, TrainTestSplit
from data.dataset_builder import TrainTestConverter, DataFrameConverter

from .prompts.Jun9_prompt import get_prompt
from .exclusion_map import exclusion_reason_map

from datasets import Dataset, DatasetDict


def prepare_dataset(
    train_file_path: str,
    val_file_path: str,
    test_file_path: str,
    target: str,
    seed: int,
    eos_token: str = None,
    max_abstract_len: int = 450,
    **kwargs,
):
    """
    Transforms a pd.DataFrame into tokenized datasets for training and validation compatible with AutoCompletionOnlyLLMs.
    The datasets are huggingface Dataset objects.

    Args:
        train_file_path: str - path to training data CSV file
        val_file_path: str - path to validation data CSV file
        test_file_path: str - path to test data CSV file
        target: str - target column name
        seed: int,
        eos_token=None


    Returns:
        Tuple of (train_dataset, test_dataset)
    """

    def _reason_code(reason_to_exclude):
        """Map a free-text rejection reason to a single reason code (R0-R4), or None
        (no reason, or a metadata/'Other' reason not in the content codebook)."""
        if type(reason_to_exclude) == str:
            reasons = reason_to_exclude.split(",")
            reason = sorted(reasons)[0].strip()
            return exclusion_reason_map.get(reason)
        return None

    def _build_target(label, reason_to_exclude):
        """Return (completion, sufficiency) for one row, or None to drop it.

        Handles both the legacy int labels (1/0) and the refined string labels
        ('include'/'exclude'/'unclear'). sufficiency=1 marks an 'unclear' paper
        (abstract insufficient to decide); it gets a placeholder leaning
        completion whose decision/reason loss is masked downstream
        (ReasonCodeSFTTrainer.compute_loss) — only the sufficiency head is
        supervised on it. include/exclude rows are sufficiency=0.
        """
        lab = label.strip().lower() if isinstance(label, str) else label
        if lab in ("unclear", "uncertain", "insufficient"):
            # The decision/reason loss is MASKED for unclear rows (sufficiency==1);
            # the completion is only a fixed-shape placeholder. Default to a valid
            # code when the row carries no content reason so it always tokenizes to
            # the 3-token "no R<x>" shape — a None here yields "no None" (2 tokens)
            # and breaks the fixed-width reshape in ReasonCodeSFTTrainer.compute_loss.
            code = _reason_code(reason_to_exclude) or "R2"
            return f"no {code}", 1
        if lab in (1, "1", "include", "yes"):
            return "yes Rn", 0
        # exclude / 0 / "no": needs a content-codebook reason or it is dropped
        code = _reason_code(reason_to_exclude)
        if code is None:
            return None
        return f"no {code}", 0

    def _make_rows(df):
        rows = []
        for x, y in zip(df.to_dict(orient="records"), df[target]):
            tgt = _build_target(y, x["If 0, reason to reject?"])
            if tgt is None:
                continue
            completion, sufficiency = tgt
            rows.append(
                {
                    "prompt": get_prompt(
                        x["Title"],
                        x["Abstract"],
                        journal=x["Journal"],
                        max_abstract_len=max_abstract_len,
                    ),
                    "completion": completion,
                    "sufficiency": sufficiency,
                }
            )
        return rows

    train_df = pd.read_csv(train_file_path)
    val_df = pd.read_csv(val_file_path)
    test_df = pd.read_csv(test_file_path)

    # Initialize tokenizer

    # Create train/validation split
    # splitter = DataSplitter(TrainTestSplit(test_size=val_size, random_state=seed))
    # X_train, X_val, y_train, y_val = splitter.split(train_df, target)
    # train_dataset = [{"prompt":get_prompt(x['Title'], x['Abstract'], refs=None, journal=x['Journal']), "completion": label_map[y]} for x, y in zip(X_train.to_dict(orient='records'), y_train)]
    # val_dataset = [{"prompt":get_prompt(x['Title'], x['Abstract'], refs=None, journal=x['Journal']), "completion": label_map[y]} for x, y in zip(X_val.to_dict(orient='records'), y_val)]
    # train_dataset = Dataset.from_list(train_dataset)
    # val_dataset = Dataset.from_list(val_dataset)

    train_dataset = Dataset.from_list(_make_rows(train_df))
    val_dataset = Dataset.from_list(_make_rows(val_df))

    train_dataset = DatasetDict({"train": train_dataset, "test": val_dataset})
    logging.info(
        f"Train-val split: Train={len(train_dataset['train'])}, Val={len(train_dataset['test'])}"
    )

    test_dataset = Dataset.from_list(_make_rows(test_df))
    logging.info(f"Test dataset size: Test={len(test_df)}")

    return train_dataset, test_dataset
