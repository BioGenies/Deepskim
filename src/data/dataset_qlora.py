import pandas as pd
from transformers import AutoTokenizer
import logging

from data.data_splitter import DataSplitter, TrainTestSplit
from data.dataset_builder import TrainTestConverter, DataFrameConverter

# from .prompts.prompt_filter import get_prompt
# from .prompts.Mar13_prompt import get_prompt
from .prompts.Apr14_prompt import get_prompt

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

    def _create_completion(label, reason_to_exclude):
        """Creates a decision + reason-code completion for the LLM.
        Returns 'yes Rn' for included papers, 'no <code>' for excluded,
        or None for invalid entries (excluded papers with no reason given).
        """
        if label == 1:
            return "yes Rn"
        if type(reason_to_exclude) == str:
            reasons = reason_to_exclude.split(",")
            reason = sorted(reasons)[0].strip()
            return f"no {exclusion_reason_map[reason]}"
        return None  # Excluded but no reason — filter out

    exclusion_reason_map = {
        "There are no interactions described": "R0",
        "The interactor is not an Ab": "R1",
        "Not enough experimental data": "R2",
        "The interactee is not an amyloid protein": "R3",
        "(Pre)Clinical trials. No interaction or amyloid data": "RA",
        "Non-English paper": "RA",
        "In silico information only": "R2",
        "Pre-print": "RA",
        "Review article": "R4",
        "Unknown antibody type": "RA",
        "Other": "RA",
    }
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

    train_dataset = [
        {
            "prompt": get_prompt(
                x["Title"],
                x["Abstract"],
                journal=x["Journal"],
                max_abstract_len=max_abstract_len,
            ),
            "completion": _create_completion(y, x["If 0, reason to reject?"]),
        }
        for x, y in zip(train_df.to_dict(orient="records"), train_df[target])
    ]

    val_dataset = [
        {
            "prompt": get_prompt(
                x["Title"],
                x["Abstract"],
                journal=x["Journal"],
                max_abstract_len=max_abstract_len,
            ),
            "completion": _create_completion(y, x["If 0, reason to reject?"]),
        }
        for x, y in zip(val_df.to_dict(orient="records"), val_df[target])
    ]

    train_dataset = Dataset.from_list([d for d in train_dataset if d["completion"] is not None])
    val_dataset = Dataset.from_list([d for d in val_dataset if d["completion"] is not None])

    train_dataset = DatasetDict({"train": train_dataset, "test": val_dataset})
    logging.info(
        f"Train-val split: Train={len(train_dataset['train'])}, Val={len(train_dataset['test'])}"
    )

    test_dataset = [
        {
            "prompt": get_prompt(
                x["Title"],
                x["Abstract"],
                journal=x["Journal"],
                max_abstract_len=max_abstract_len,
            ),
            "completion": _create_completion(y, x["If 0, reason to reject?"]),
        }
        for x, y in zip(test_df.to_dict(orient="records"), test_df[target])
    ]

    test_dataset = Dataset.from_list([d for d in test_dataset if d["completion"] is not None])
    logging.info(f"Test dataset size: Test={len(test_df)}")

    return train_dataset, test_dataset
