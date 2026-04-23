from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import logging

from data.data_splitter import (
    DataSplitter,
    StratifiedKFoldByDecisionAndReason,
    StratifiedSplitByDecisionAndReason,
)

REASON_COL_SRC = "If so; reason to reject?"
REASON_COL_DST = "If 0, reason to reject?"


def _prepare_df(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    df = X.copy()
    df["labels"] = y.values
    cols = [c for c in df.columns if c != REASON_COL_SRC] + [REASON_COL_SRC]
    return df[cols].rename(columns={REASON_COL_SRC: REASON_COL_DST})


def split_data_kfold(
    raw_path: str,
    target: str,
    n_splits: int,
    save_dir: str,
    seed: int = 42,
    invalid_target_values=("?", "-1"),
):
    df = pd.read_csv(raw_path)
    strategy = StratifiedKFoldByDecisionAndReason(
        n_splits=n_splits,
        random_state=seed,
        reason_column=REASON_COL_SRC,
        invalid_target_values=invalid_target_values,
    )
    folds = DataSplitter(strategy).split(df, target_column=target)

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
        fold_dir = save_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _prepare_df(X_train, y_train).to_csv(fold_dir / "train_data.csv", index=False)
        _prepare_df(X_test, y_test).to_csv(fold_dir / "test_data.csv", index=False)
        logging.info(
            f"Fold {fold_idx}: train={len(X_train)} test={len(X_test)} -> {fold_dir}"
        )

    print(
        f"Raw data split into {len(folds)} stratified folds under "
        f"{save_root}/fold_0..fold_{len(folds) - 1}/ "
        f"(each with train_data.csv and test_data.csv)."
    )


def split_data_tune_and_report(
    raw_path: str,
    target: str,
    dev_size: float,
    n_splits: int,
    save_dir: str,
    seed: int = 42,
    invalid_target_values=("?", "-1"),
):
    """Data methodology: a held-out DEV split for hyperparameter
    tuning, plus an independent k-fold split over the full dataset for
    frozen-config reporting.

    Layout produced under save_dir/:
        tuning/   train_data.csv  +  dev_data.csv
        reporting/fold_{0..k-1}/  train_data.csv  +  test_data.csv

    Both splits are stratified jointly on decision and reason via the
    existing StratifiedSplitByDecisionAndReason / StratifiedKFoldByDecisionAndReason
    strategies, so the same pre-filtering (invalid targets, multi-reason,
    missing reason) is applied consistently.
    """
    df = pd.read_csv(raw_path)
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    print("\n>>> TUNING SPLIT (train / dev) <<<\n")
    tune_strategy = StratifiedSplitByDecisionAndReason(
        test_size=dev_size,
        val_size=0.0,
        random_state=seed,
        reason_column=REASON_COL_SRC,
        invalid_target_values=invalid_target_values,
    )
    X_train, _, X_dev, y_train, _, y_dev = DataSplitter(tune_strategy).split(
        df, target_column=target
    )
    tuning_dir = save_root / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    _prepare_df(X_train, y_train).to_csv(tuning_dir / "train_data.csv", index=False)
    _prepare_df(X_dev, y_dev).to_csv(tuning_dir / "dev_data.csv", index=False)
    logging.info(f"Tuning split: train={len(X_train)} dev={len(X_dev)} -> {tuning_dir}")

    print("\n>>> REPORTING SPLIT (k-fold CV over full dataset) <<<\n")
    kfold_strategy = StratifiedKFoldByDecisionAndReason(
        n_splits=n_splits,
        random_state=seed,
        reason_column=REASON_COL_SRC,
        invalid_target_values=invalid_target_values,
    )
    folds = DataSplitter(kfold_strategy).split(df, target_column=target)

    reporting_dir = save_root / "reporting"
    reporting_dir.mkdir(parents=True, exist_ok=True)
    for fold_idx, (X_tr, X_te, y_tr, y_te) in enumerate(folds):
        fold_dir = reporting_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _prepare_df(X_tr, y_tr).to_csv(fold_dir / "train_data.csv", index=False)
        _prepare_df(X_te, y_te).to_csv(fold_dir / "test_data.csv", index=False)
        logging.info(
            f"Fold {fold_idx}: train={len(X_tr)} test={len(X_te)} -> {fold_dir}"
        )

    print(
        f"\nDone. Tuning under {tuning_dir} (train_data.csv, dev_data.csv). "
        f"Reporting under {reporting_dir}/fold_0..fold_{len(folds) - 1}/ "
        f"(each with train_data.csv and test_data.csv).\n"
        f"Tuning phase: train on tuning/train_data.csv, evaluate configs on "
        f"tuning/dev_data.csv, pick best, FREEZE hyperparameters.\n"
        f"Reporting phase: re-train the frozen config on each fold's "
        f"train_data.csv and predict on test_data.csv; aggregate per-fold "
        f"metrics (mean ± std) and pool the {len(folds)} fold predictions "
        f"for a confusion matrix + bootstrap CIs over the full dataset."
    )


def split_data_threeway(
    raw_path: str,
    target: str,
    test_size: float,
    val_size: float,
    save_dir: str,
    seed: int = 42,
    invalid_target_values=("?", "-1"),
):
    df = pd.read_csv(raw_path)
    strategy = StratifiedSplitByDecisionAndReason(
        test_size=test_size,
        val_size=val_size,
        random_state=seed,
        reason_column=REASON_COL_SRC,
        invalid_target_values=invalid_target_values,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter(strategy).split(
        df, target_column=target
    )

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    _prepare_df(X_train, y_train).to_csv(save_root / "train_data.csv", index=False)
    _prepare_df(X_val, y_val).to_csv(save_root / "val_data.csv", index=False)
    _prepare_df(X_test, y_test).to_csv(save_root / "test_data.csv", index=False)

    print(
        f"Raw data split into train={len(X_train)} val={len(X_val)} "
        f"test={len(X_test)} under {save_root} "
        f"(train_data.csv, val_data.csv, test_data.csv)."
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--raw_path", type=str, required=True, help="Path to the raw dataset file"
    )
    parser.add_argument(
        "--target", type=str, default="Rejection?", help="Target column name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["kfold", "split", "tune_report"],
        default="kfold",
        help=(
            "kfold: stratified k-fold CV; "
            "split: single train/val/test split; "
            "tune_report: held-out dev split for tuning + k-fold for reporting"
        ),
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds (mode=kfold or tune_report)",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.15, help="Test fraction (mode=split)"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.15, help="Val fraction (mode=split)"
    )
    parser.add_argument(
        "--dev_size",
        type=float,
        default=0.15,
        help="Dev fraction (mode=tune_report)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="dataset",
        help="Directory to save the split data",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--invalid_target_values",
        type=str,
        default="?,-1",
        help="Comma-separated target values to treat as invalid and drop "
        "(NaN and empty string are always dropped). Default: '?,-1'",
    )

    args = parser.parse_args()
    invalid_target_values = tuple(
        v.strip() for v in args.invalid_target_values.split(",") if v.strip()
    )

    if args.mode == "kfold":
        split_data_kfold(
            raw_path=args.raw_path,
            target=args.target,
            n_splits=args.n_splits,
            save_dir=args.save_dir,
            seed=args.seed,
            invalid_target_values=invalid_target_values,
        )
    elif args.mode == "tune_report":
        split_data_tune_and_report(
            raw_path=args.raw_path,
            target=args.target,
            dev_size=args.dev_size,
            n_splits=args.n_splits,
            save_dir=args.save_dir,
            seed=args.seed,
            invalid_target_values=invalid_target_values,
        )
    else:
        split_data_threeway(
            raw_path=args.raw_path,
            target=args.target,
            test_size=args.test_size,
            val_size=args.val_size,
            save_dir=args.save_dir,
            seed=args.seed,
            invalid_target_values=invalid_target_values,
        )
