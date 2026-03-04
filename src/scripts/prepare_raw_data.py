from argparse import ArgumentParser

import pandas as pd
import logging

from data.data_splitter import DataSplitter, TrainTestSplit
from deployment import Integrator, Preprocessor

def prepare_raw(
    raw_path: str,
    fetched_save_path:str,
    preprocessed_save_path: str,
):
    """
    End-to-end data preparation function. This function fetches data from PubMed, preprocesses it, and splits into train and test sets.
    Args:
        raw_path: Path to CSV file with data
        seed: Random seed for reproducibility
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    integrator = Integrator(raw_path=raw_path, pmid_col="PMID", email="test@gmail.com")
    # email is used for Entrez API and is not necessary

    # integrator.reduce_columns(keep_columns = ['PMID', "Rejection?", "If so; reason to reject?"])
    integrator.fetch_pubmed(save_path=fetched_save_path)
    integrator.merge()
    preprocessor = Preprocessor(integrator.merged_df.copy())

    preprocessor.dropna(subset=["Abstract"])
    preprocessor.drop_values(column="If so; reason to reject?", value="Review article")

    preprocessor.map_labels(label_col="Rejection?", mapping={"Rejected": 0, "Useful": 1})
    preprocessor.drop_values(column="Rejection?", value=-1)

    #Final cleanup of spurious columns
    to_drop = [col for col in preprocessor.df.columns if "Unnamed" in col]
    preprocessor.df = preprocessor.df.drop(columns=to_drop)
    preprocessor.df = preprocessor.df.drop(columns=["Year.1"])



    preprocessor.df.to_csv(preprocessed_save_path, index=False)

    return




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, required=True, help="Path to the raw dataset file")
    parser.add_argument("--fetched_save_path", type=str, default="dataset/data_fetched.csv")
    parser.add_argument("--preprocessed_save_path", type=str, default="dataset/data_preprocessed.csv", help="Path to save the preprocessed dataset")
    args = parser.parse_args()
    prepare_raw(
        raw_path=args.raw_path,
        fetched_save_path=args.fetched_save_path,
        preprocessed_save_path=args.preprocessed_save_path
    )


