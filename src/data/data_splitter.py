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


# Stratified K-Fold jointly stratified on decision (include/exclude) and reason
class StratifiedKFoldByDecisionAndReason(SplitStrategy):
    """
    Stratified K-Fold split using a single composite key that captures both
    the include/exclude decision and the rejection reason. Because a reason
    only applies to excluded rows, the key is:

        '__INCLUDED__'   for included rows (target == included_value)
        <reason string>  for excluded rows

    Rows with *multiple* rejection reasons (comma-separated in the reason
    column) are dropped before splitting. Rows marked excluded but missing
    any reason are also dropped. Strata with fewer than ``n_splits`` members
    are pooled into a single ``__RARE__`` bucket so StratifiedKFold receives
    at least ``n_splits`` samples per class.

    Returns a list of ``(X_train, X_test, y_train, y_test)`` per fold, where
    ``y`` is the original decision column (not the stratification key).
    """

    INCLUDED_KEY = "__INCLUDED__"
    RARE_KEY = "__RARE__"

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        reason_column: str = "If so; reason to reject?",
        included_value=1,
        invalid_target_values=("?", "-1"),
        verbose: bool = True,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.reason_column = reason_column
        self.included_value = included_value
        self.invalid_target_values = set(str(v) for v in invalid_target_values)
        self.verbose = verbose

    def split_data(self, df: pd.DataFrame, target_column: str):
        logging.info(
            f"Performing Stratified K-Fold split on decision + reason "
            f"({self.n_splits} folds)"
        )

        n_total = len(df)
        reason_raw = df[self.reason_column]
        reason_str = reason_raw.astype(object).where(reason_raw.notna(), "")

        target_vals = df[target_column]
        target_str = target_vals.astype(str).str.strip()
        is_invalid_target = target_vals.isna() | target_str.isin(
            {"", *self.invalid_target_values}
        )

        is_multi = reason_str.apply(lambda s: isinstance(s, str) and "," in s)
        is_excluded = df[target_column] != self.included_value
        is_missing_reason = is_excluded & reason_str.astype(str).str.strip().eq("")

        n_invalid_target = int(is_invalid_target.sum())
        n_multi = int((is_multi & ~is_invalid_target).sum())
        n_missing = int((is_missing_reason & ~is_invalid_target & ~is_multi).sum())

        keep_mask = ~(is_invalid_target | is_multi | is_missing_reason)
        df_f = df.loc[keep_mask].reset_index(drop=True)

        reason_clean = df_f[self.reason_column].fillna("").astype(str).str.strip()
        key = reason_clean.copy()
        key[df_f[target_column] == self.included_value] = self.INCLUDED_KEY

        counts = key.value_counts()
        rare_strata = counts[counts < self.n_splits].index
        n_rare_rows = int(key.isin(rare_strata).sum())
        pooled_key = key.where(~key.isin(rare_strata), self.RARE_KEY)

        if self.verbose:
            self._print_prefilter_stats(
                n_total, n_invalid_target, n_multi, n_missing, len(df_f),
                counts, rare_strata, n_rare_rows,
            )

        X = df_f.drop(columns=[target_column])
        y = df_f[target_column]
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )

        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, pooled_key)):
            folds.append(
                (
                    X.iloc[train_idx],
                    X.iloc[test_idx],
                    y.iloc[train_idx],
                    y.iloc[test_idx],
                )
            )
            if self.verbose:
                self._print_fold_stats(fold_idx, pooled_key, y, train_idx, test_idx)

        if self.verbose:
            self._print_cross_fold_table(pooled_key, folds)

        logging.info("Stratified K-Fold (decision + reason) split completed.")
        return folds  # list of (X_train, X_test, y_train, y_test)

    def _print_prefilter_stats(
        self, n_total, n_invalid_target, n_multi, n_missing, n_kept,
        counts, rare_strata, n_rare_rows,
    ):
        print("=" * 78)
        print(
            f"StratifiedKFoldByDecisionAndReason  "
            f"(n_splits={self.n_splits}, seed={self.random_state})"
        )
        print("=" * 78)
        print(f"  input rows                            : {n_total}")
        print(
            f"  dropped (invalid target)              : {n_invalid_target}  "
            f"(NaN, '', {sorted(self.invalid_target_values)})"
        )
        print(f"  dropped (multi-reason)                : {n_multi}")
        print(f"  dropped (excluded but missing reason) : {n_missing}")
        print(f"  rows kept for k-fold                  : {n_kept}")
        print(f"  distinct strata (pre-pooling)         : {len(counts)}")
        print(
            f"  strata pooled into '{self.RARE_KEY}'       : "
            f"{len(rare_strata)} strata covering {n_rare_rows} rows"
        )
        print()
        print("  Stratum counts (after pooling):")
        pooled_counts = counts.copy()
        if len(rare_strata):
            pooled_counts = pooled_counts.drop(index=rare_strata)
            pooled_counts[self.RARE_KEY] = n_rare_rows
        pooled_counts = pooled_counts.sort_values(ascending=False)
        total = int(pooled_counts.sum())
        for k, c in pooled_counts.items():
            print(f"    {c:5d} ({c / total:5.1%})  {k}")
        print()

    def _print_fold_stats(self, fold_idx, pooled_key, y, train_idx, test_idx):
        print(f"--- Fold {fold_idx + 1} / {self.n_splits} ---")
        print(f"  sizes           : train={len(train_idx)}  test={len(test_idx)}")

        def _pct(vc):
            t = int(vc.sum())
            return ", ".join(f"{k}={int(v)} ({int(v) / t:.1%})" for k, v in vc.items())

        print(f"  decision (train): {_pct(y.iloc[train_idx].value_counts())}")
        print(f"  decision (test) : {_pct(y.iloc[test_idx].value_counts())}")
        print("  strata (test) :")
        test_dist = pooled_key.iloc[test_idx].value_counts()
        t = int(test_dist.sum())
        for k, v in test_dist.items():
            print(f"     {int(v):4d} ({int(v) / t:5.1%})  {k}")
        print()

    def _print_cross_fold_table(self, pooled_key, folds):
        print("=" * 78)
        print("Cross-fold test-set stratum share (%)")
        print("=" * 78)
        strata = list(pooled_key.value_counts().index)
        header = "  {:55s} ".format("stratum") + "  ".join(
            f"f{i + 1:>2}" for i in range(len(folds))
        )
        print(header)
        for s in strata:
            row_vals = []
            for _, X_test, _, _ in folds:
                test_key = pooled_key.loc[X_test.index]
                t = len(test_key)
                row_vals.append(test_key.eq(s).sum() / t if t else 0.0)
            name = s if len(s) <= 55 else (s[:52] + "...")
            print(
                "  {:55s} ".format(name)
                + "  ".join(f"{v * 100:4.1f}" for v in row_vals)
            )
        print()


# Stratified single train/val/test split jointly on decision and reason
class StratifiedSplitByDecisionAndReason(SplitStrategy):
    """
    Single stratified train/val/test split using the same composite
    decision + reason key as ``StratifiedKFoldByDecisionAndReason``.

    Same pre-processing as the k-fold variant:
      - rows with multiple comma-separated rejection reasons are dropped
      - rows marked excluded but missing a reason are dropped
      - strata with fewer than ``min_per_stratum`` members are pooled
        into a single ``__RARE__`` bucket so both splits stay stratifiable

    The split is performed as two sequential stratified splits:
      1. full -> (train+val) / test
      2. (train+val) -> train / val

    ``val_size`` is expressed as a fraction of the *original* dataset (to
    match ``test_size``), not as a fraction of the train+val remainder.

    Returns ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """

    INCLUDED_KEY = "__INCLUDED__"
    RARE_KEY = "__RARE__"

    def __init__(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        reason_column: str = "If so; reason to reject?",
        included_value=1,
        invalid_target_values=("?", "-1"),
        min_per_stratum: int = 5,
        verbose: bool = True,
    ):
        if test_size <= 0 or val_size < 0 or test_size + val_size >= 1.0:
            raise ValueError(
                f"Invalid split sizes: test_size={test_size}, val_size={val_size}"
            )
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.reason_column = reason_column
        self.included_value = included_value
        self.invalid_target_values = set(str(v) for v in invalid_target_values)
        self.min_per_stratum = min_per_stratum
        self.verbose = verbose

    def split_data(self, df: pd.DataFrame, target_column: str):
        logging.info(
            f"Performing stratified train/val/test split on decision + reason "
            f"(test_size={self.test_size}, val_size={self.val_size})"
        )

        n_total = len(df)
        reason_raw = df[self.reason_column]
        reason_str = reason_raw.astype(object).where(reason_raw.notna(), "")

        target_vals = df[target_column]
        target_str = target_vals.astype(str).str.strip()
        is_invalid_target = target_vals.isna() | target_str.isin(
            {"", *self.invalid_target_values}
        )

        is_multi = reason_str.apply(lambda s: isinstance(s, str) and "," in s)
        is_excluded = df[target_column] != self.included_value
        is_missing_reason = is_excluded & reason_str.astype(str).str.strip().eq("")

        n_invalid_target = int(is_invalid_target.sum())
        n_multi = int((is_multi & ~is_invalid_target).sum())
        n_missing = int((is_missing_reason & ~is_invalid_target & ~is_multi).sum())

        keep_mask = ~(is_invalid_target | is_multi | is_missing_reason)
        df_f = df.loc[keep_mask].reset_index(drop=True)

        reason_clean = df_f[self.reason_column].fillna("").astype(str).str.strip()
        key = reason_clean.copy()
        key[df_f[target_column] == self.included_value] = self.INCLUDED_KEY

        counts = key.value_counts()
        rare_strata = counts[counts < self.min_per_stratum].index
        n_rare_rows = int(key.isin(rare_strata).sum())
        pooled_key = key.where(~key.isin(rare_strata), self.RARE_KEY)

        if self.verbose:
            self._print_prefilter_stats(
                n_total, n_invalid_target, n_multi, n_missing, len(df_f),
                counts, rare_strata, n_rare_rows,
            )

        X = df_f.drop(columns=[target_column])
        y = df_f[target_column]

        X_trainval, X_test, y_trainval, y_test, key_trainval, _ = train_test_split(
            X, y, pooled_key,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=pooled_key,
        )

        if self.val_size > 0:
            val_relative = self.val_size / (1.0 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval,
                test_size=val_relative,
                random_state=self.random_state,
                stratify=key_trainval,
            )
        else:
            X_train, y_train = X_trainval, y_trainval
            X_val = X_trainval.iloc[0:0]
            y_val = y_trainval.iloc[0:0]

        if self.verbose:
            self._print_split_stats(pooled_key, y, X_train, X_val, X_test)

        logging.info("Stratified train/val/test split completed.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _print_prefilter_stats(
        self, n_total, n_invalid_target, n_multi, n_missing, n_kept,
        counts, rare_strata, n_rare_rows,
    ):
        print("=" * 78)
        print(
            f"StratifiedSplitByDecisionAndReason  "
            f"(test_size={self.test_size}, val_size={self.val_size}, "
            f"seed={self.random_state})"
        )
        print("=" * 78)
        print(f"  input rows                            : {n_total}")
        print(
            f"  dropped (invalid target)              : {n_invalid_target}  "
            f"(NaN, '', {sorted(self.invalid_target_values)})"
        )
        print(f"  dropped (multi-reason)                : {n_multi}")
        print(f"  dropped (excluded but missing reason) : {n_missing}")
        print(f"  rows kept for split                   : {n_kept}")
        print(f"  distinct strata (pre-pooling)         : {len(counts)}")
        print(
            f"  strata pooled into '{self.RARE_KEY}'       : "
            f"{len(rare_strata)} strata covering {n_rare_rows} rows"
        )
        print()
        print("  Stratum counts (after pooling):")
        pooled_counts = counts.copy()
        if len(rare_strata):
            pooled_counts = pooled_counts.drop(index=rare_strata)
            pooled_counts[self.RARE_KEY] = n_rare_rows
        pooled_counts = pooled_counts.sort_values(ascending=False)
        total = int(pooled_counts.sum())
        for k, c in pooled_counts.items():
            print(f"    {c:5d} ({c / total:5.1%})  {k}")
        print()

    def _print_split_stats(self, pooled_key, y, X_train, X_val, X_test):
        total = len(X_train) + len(X_val) + len(X_test)
        print("--- Split sizes ---")
        print(
            f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  "
            f"(total={total})"
        )

        def _pct(vc):
            t = int(vc.sum())
            return ", ".join(f"{k}={int(v)} ({int(v) / t:.1%})" for k, v in vc.items())

        print(f"  decision (train): {_pct(y.loc[X_train.index].value_counts())}")
        if len(X_val):
            print(f"  decision (val)  : {_pct(y.loc[X_val.index].value_counts())}")
        print(f"  decision (test) : {_pct(y.loc[X_test.index].value_counts())}")
        print()

        print("=" * 78)
        print("Stratum share (%) per split")
        print("=" * 78)
        strata = list(pooled_key.value_counts().index)
        print(
            "  {:55s}  {:>6s}  {:>6s}  {:>6s}".format(
                "stratum", "train", "val", "test"
            )
        )
        train_k = pooled_key.loc[X_train.index]
        val_k = pooled_key.loc[X_val.index] if len(X_val) else pooled_key.iloc[0:0]
        test_k = pooled_key.loc[X_test.index]
        for s in strata:
            t_train = train_k.eq(s).sum() / len(train_k) if len(train_k) else 0
            t_val = val_k.eq(s).sum() / len(val_k) if len(val_k) else 0
            t_test = test_k.eq(s).sum() / len(test_k) if len(test_k) else 0
            name = s if len(s) <= 55 else (s[:52] + "...")
            print(
                f"  {name:55s}  {t_train * 100:6.1f}  {t_val * 100:6.1f}  {t_test * 100:6.1f}"
            )
        print()


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
