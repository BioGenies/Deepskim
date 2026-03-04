from argparse import ArgumentParser

import pandas as pd
import logging

from data.data_splitter import DataSplitter, TrainTestSplit
from deployment import Integrator, Preprocessor

def split_data(    raw_path: str,
    target: str,
    test_size: float,
    val_size: float=0.1,
    seed: int=42,):
        # Create train/validation split
    
    df = pd.read_csv(raw_path)
    splitter = DataSplitter(TrainTestSplit(test_size=test_size, random_state=seed))
    X_train, X_test, y_train, y_test = splitter.split(df, target)
    tmp_train_df = X_train.copy()
    tmp_train_df["Rejection?"] = y_train.values
    test_df = X_test.copy()
    test_df["labels"] = y_test.values

    val_splitter = DataSplitter(TrainTestSplit(test_size=val_size, random_state=seed))
    X_train, X_val, y_train, y_val = val_splitter.split(tmp_train_df, target)
    train_df = X_train.copy()
    train_df["labels"] = y_train.values
    val_df = X_val.copy()
    val_df["labels"] = y_val.values


    col_to_move = 'If so; reason to reject?'
    cols = [c for c in train_df.columns if c != col_to_move] + [col_to_move]
    train_df = train_df[cols]
    val_df = val_df[cols]

    test_df = test_df[cols]
    train_df = train_df.rename(columns={'If so; reason to reject?': 'If 0, reason to reject?'})
    val_df = val_df.rename(columns={'If so; reason to reject?': 'If 0, reason to reject?'})
    test_df = test_df.rename(columns={'If so; reason to reject?': 'If 0, reason to reject?'})


    train_df.to_csv(f'{args.save_dir}/train_data.csv', index=False)
    val_df.to_csv(f'{args.save_dir}/val_data.csv', index=False)
    test_df.to_csv(f'{args.save_dir}/test_data.csv', index=False)
    logging.info(f"Train-val-test split: Train={len(X_train)}, Val={len(X_val)} Test={len(X_test)}")
    print("Raw data split into train, val and test sets. Stored as 'train_data.csv', 'val_data.csv' and 'test_data.csv'.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True, help="Path to the raw dataset file")
    parser.add_argument("--target", type=str, default="Rejection?", help="Target column name")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test set")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of validation set from training data")
    parser.add_argument("--save_dir", type=str, default="dataset", help="Directory to save the split data")

    args = parser.parse_args()
    split_data(
        raw_path=args.raw_path,
        target=args.target,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=42,
    )