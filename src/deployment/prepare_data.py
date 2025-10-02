import pandas as pd
from transformers import AutoTokenizer
import logging

from data_splitter import DataSplitter, TrainTestSplit
from dataset_builder import TrainTestConverter, DataFrameConverter


def prepare_dataset(
    train_file_path: str,
    test_file_path: str,
    target: str,
    test_size: float,
    model_name: str,
    max_length: int,
    keywords: list[str],
    special_tokens: list[str],
    seed: int,
):
    """
    Transforms a pd.DataFrame into tokenized datasets for training and validation.
    The datasets are huggingface Dataset objects.

    Args:
        file_path: Path to CSV file with data
        target: Target column name
        model_name: HuggingFace model name for tokenizer
        max_length: Maximum sequence length for tokenization
        keywords: List of keywords to mark in text
        special_tokens: Additional special tokens for tokenizer
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Create train/validation split
    splitter = DataSplitter(TrainTestSplit(test_size=test_size, random_state=seed))
    X_train, X_val, y_train, y_val = splitter.split(train_df, target)

    # Convert to datasets
    converter = TrainTestConverter()

    # Create train/validation dataset
    train_dataset = converter.convert(
        (X_train, X_val, y_train, y_val), tokenizer, keywords, max_length
    )

    logging.info(f"Train-val split: Train={len(X_train)}, Val={len(X_val)}")

    # Create test dataset
    df_converter = DataFrameConverter()
    test_dataset = df_converter.convert(test_df, tokenizer, keywords, max_length)
    logging.info(f"Test dataset size: Test={len(test_df)}")

    return train_dataset, test_dataset
