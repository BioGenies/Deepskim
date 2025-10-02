from __future__ import annotations
from typing import Optional, Iterable
import pandas as pd


class Preprocessor:
    """
    Preprocessor working on a pandas DataFrame.

    Attributes:
    - original_df: original passed DataFrame (not modified)
    - df: working copy of the DataFrame
    - dropped_df: rows dropped by dropna()
    - train_df, test_df: results of split_dataset()
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.original_df: pd.DataFrame = df.copy()
        self.df: pd.DataFrame = df.copy()
        self.dropped_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def dropna(self, subset: Optional[Iterable[str]] = None) -> None:
        """
        Drop rows that contain NaN.
        - subset: iterable of column names to check; if None -> check all columns.
        Effect:
        - self.dropped_df <- copied rows that were dropped
        - self.df <- DataFrame after dropping those rows
        """
        self.df = self.original_df.copy()  # reset to original before dropping
        if self.df is None:
            raise ValueError("No DataFrame in self.df")

        if subset is not None:
            # ensure columns exist
            missing = [c for c in subset if c not in self.df.columns]
            if missing:
                raise KeyError(f"Columns in subset do not exist in df: {missing}")
            mask = self.df[list(subset)].isna().any(axis=1)
        else:
            mask = self.df.isna().any(axis=1)

        dropped = self.df.loc[mask].copy()
        kept = self.df.loc[~mask].copy()

        dropped_count = len(dropped)
        print(
            f"Dropped {dropped_count} rows containing NaN. Dropped rows are stored in self.dropped_df."
        )

        self.dropped_df = (
            dropped if not dropped.empty else pd.DataFrame(columns=self.df.columns)
        )
        self.df = kept

    def map_labels(self, label_col: str, mapping: dict) -> None:
        """
        Map values in column `label_col` according to `mapping`.
        - mapping should map original values to 0 or 1 (e.g. {"Rejected": 0, "Useful": 1}).
        - This implementation applies a direct .map(); values not in mapping become NaN.
        Effect: updates self.df[label_col] in place.
        """
        if label_col not in self.df.columns:
            raise KeyError(f"Label column '{label_col}' does not exist in df")

        # Simple mapping: unmapped values become NaN
        self.df[label_col] = self.df[label_col].map(mapping).fillna(-1).astype(int)

        value_counts = self.df[label_col].value_counts(dropna=False).to_dict()
        unmapped_count = int((self.df[label_col] == -1).sum())
        print(
            f"Mapped labels in column '{label_col}' using provided mapping. Value counts: {value_counts}. "
            f"Unmapped (set to -1): {unmapped_count} rows."
        )

    def split_dataset(self, label_col: str) -> None:
        """
        Split self.df into train/test:
        - train_df: rows where label_col is exactly 0 or 1
        - test_df: all other rows (e.g. NaN, other strings, other numbers)
        Sets self.train_df and self.test_df.
        """
        if label_col not in self.df.columns:
            raise KeyError(f"Label column '{label_col}' does not exist in df")

        working = self.df.copy()

        train = working.loc[working[label_col].isin([0, 1])].copy()
        test = working.loc[~working[label_col].isin([0, 1])].copy()

        self.train_df = train
        self.test_df = test

        print(
            f"Dataset split completed: {len(self.train_df)} rows in self.train_df, "
            f"{len(self.test_df)} rows in self.test_df. Data stored in these attributes."
        )

    def drop_values(self, column: str, value) -> None:
        """
        Drop rows where self.df[column] == value and add them to self.dropped_df.
        - If self.dropped_df does not exist (None) or is empty, create it from the dropped rows.
        - Otherwise append the dropped rows to existing self.dropped_df.
        - Prints how many rows were added to self.dropped_df.
        """
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' does not exist in df")

        mask = self.df[column] == value
        dropped = self.df.loc[mask].copy()
        kept = self.df.loc[~mask].copy()

        added_count = len(dropped)
        if added_count == 0:
            print(
                f"No rows found matching condition {column} == {value}. Nothing added to self.dropped_df."
            )
            self.df = kept

        # If dropped_df is None or empty, set it to dropped; otherwise append
        if self.dropped_df is None or self.dropped_df.empty:
            # make a fresh copy to avoid linking to the same object
            self.dropped_df = dropped.copy()
        else:
            # append preserving indices (do not reset index)
            self.dropped_df = pd.concat(
                [self.dropped_df, dropped], ignore_index=False
            ).copy()

        # update working df
        self.df = kept

        print(
            f"Dropped {added_count} rows (condition: {column} == {value}). Dropped rows are stored in self.dropped_df."
        )
