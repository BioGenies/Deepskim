import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Abstract Base Class for Data Splitting Strategy
class SplitStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Splits data according to the strategy.

        Returns:
            - TrainTestSplit: Tuple (X_train, X_test, y_train, y_test)
            - CVSplit: List of such tuples for each fold
        """
        pass


# Train-test split
class TrainTestSplit(SplitStrategy):
    def __init__(self, test_size=0.2, random_state=42, stratify=True):
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split_data(self, df: pd.DataFrame, target_column: str):
        logging.info(
            f"Performing {'stratified' if self.stratify else 'simple'} train-test split."
        )
        X = df.drop(columns=[target_column])
        y = df[target_column]

        stratify_arg = y if self.stratify else None
        splits = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_arg,
        )
        logging.info("Train-test split completed.")
        return splits  # (X_train, X_test, y_train, y_test)


# Stratified K-Fold cross-validation split
class StratifiedKFoldSplit(SplitStrategy):
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        logging.info(f"Performing Stratified K-Fold split ({self.n_splits} folds)")
        X, y = df.drop(columns=[target_column]), df[target_column]
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )

        folds = []
        for train_idx, test_idx in skf.split(X, y):
            folds.append(
                (
                    X.iloc[train_idx],
                    X.iloc[test_idx],
                    y.iloc[train_idx],
                    y.iloc[test_idx],
                )
            )
        logging.info("Stratified K-Fold split completed.")
        return folds  # list of (X_train, X_test, y_train, y_test)


# Context for splitting
class DataSplitter:
    def __init__(self, strategy: SplitStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: SplitStrategy):
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str):
        logging.info("Splitting data using the selected strategy.")
        return self._strategy.split_data(df, target_column)


# Usage example
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "processed" / "amyloid-02-07-2025.csv"

    df = pd.read_csv(data_path)
    print(df.head())
    target = "rejection"

    # Split
    splitter = DataSplitter(StratifiedKFoldSplit(n_splits=5))
    folds = splitter.split(df, target)
