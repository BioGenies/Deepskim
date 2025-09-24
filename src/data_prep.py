import pandas as pd
from transformers import AutoTokenizer
import logging

from data_splitter import DataSplitter, TrainTestSplit, StratifiedKFoldSplit
from dataset_builder import TrainTestConverter, FoldsConverter


def prepare_dataset(
    file_path: str,
    target: str,
    split_strategy: str,
    model_name: str,
    max_length: int,
    keywords: list[str],
    special_tokens: list[str],
    seed: int,
):
    """
    Prepare datasets for training and testing with proper train/validation/test splits.

    Args:
        file_path: Path to CSV file
        target: Target column name
        split_strategy: Either "train_test" or "cv"
        model_name: HuggingFace model name for tokenizer
        max_length: Maximum sequence length for tokenization
        keywords: List of keywords to mark in text
        special_tokens: Additional special tokens for tokenizer
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Load and preprocess data
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["Abstract"]).reset_index(drop=True)
    df["rejection"] = df["rejection"].map({"Rejected": 0, "Useful": 1})

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    if split_strategy == "train_test":
        # Create train/validation/test split (70/15/15)
        splitter = DataSplitter(TrainTestSplit(test_size=0.3, random_state=seed))
        X_train, X_temp, y_train, y_temp = splitter.split(df, target)

        # Split the remaining 30% into validation (15%) and test (15%)
        temp_df = X_temp.copy()
        temp_df[target] = y_temp.values

        splitter_temp = DataSplitter(TrainTestSplit(test_size=0.5, random_state=seed))
        X_val, X_test, y_val, y_test = splitter_temp.split(temp_df, target)

        # Convert to datasets
        converter = TrainTestConverter()

        # Create train/validation dataset
        train_dataset = converter.convert(
            (X_train, X_val, y_train, y_val), tokenizer, keywords, max_length
        )

        # Create validation/test dataset (for final evaluation)
        test_dataset = converter.convert(
            (X_val, X_test, y_val, y_test), tokenizer, keywords, max_length
        )

        logging.info(
            f"Train-test split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}"
        )

    elif split_strategy == "cv":
        # Hold out 15% for final testing
        splitter = DataSplitter(TrainTestSplit(test_size=0.15, random_state=seed))
        X_temp, X_test, y_temp, y_test = splitter.split(df, target)

        # Prepare temp dataframe for CV
        temp_df = X_temp.copy()
        temp_df[target] = y_temp.values

        # Create 5-fold cross-validation splits
        splitter.set_strategy(
            StratifiedKFoldSplit(n_splits=5, shuffle=True, random_state=seed)
        )
        train_folds = splitter.split(temp_df, target)

        # Convert folds to datasets
        converter = TrainTestConverter()
        cv_converter = FoldsConverter(converter)

        # Create test dataset (same data for both train and test since it's held out)
        test_dataset = converter.convert(
            (X_test, X_test, y_test, y_test), tokenizer, keywords, max_length
        )

        # Create CV datasets
        train_dataset = cv_converter.convert(
            train_folds, tokenizer, keywords, max_length
        )

        logging.info(f"CV split: {len(train_folds)} folds, Test holdout={len(X_test)}")

    else:
        raise ValueError(
            f"Unknown split_strategy: {split_strategy}. Use 'train_test' or 'cv'"
        )

    return train_dataset, test_dataset


# file_path = "your_dataset_file.csv"
# target = 'rejection'
# split_strategy = 'train_test'
# model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
# max_length = 20
# keywords = ['amyloid']
# special_tokens = ['[KEY]']
# seed = 42

# result = prepare_dataset(
#     file_path=file_path,
#     target=target,
#     split_strategy=split_strategy,
#     model_name=model_name,
#     max_length=max_length,
#     keywords=keywords,
#     special_tokens=special_tokens,
#     seed=seed
# )
