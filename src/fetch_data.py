import argparse
import warnings
import pandas as pd

from fetchers import fetch_references
from fetchers import fetch_pubmed_data


def load_pmid_list(path: str, column: str) -> list[str]:
    """
    Load PMIDs from a dataset file, preserving order and emitting a warning if duplicates are found.
    """
    # Automatically detect delimiter for CSV/TSV based on file extension
    delim = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, delimiter=delim)
    pmids = df[column].dropna().astype(str).tolist()
    if len(pmids) != len(set(pmids)):
        warnings.warn(
            f"Found {len(pmids) - len(set(pmids))} duplicate PMID(s) in column '{column}'."
        )
    return pmids


def main(args):
    if args.mode == "pubmed":
        ids = load_pmid_list(args.dataset, args.column)
        df = fetch_pubmed_data(ids, args.email)
    elif args.mode == "references":
        ids = load_pmid_list(args.dataset, args.column)
        df = fetch_references(ids)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} records to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pubmed", "references"], required=True)
    p.add_argument("-d", "--dataset", required=True)
    p.add_argument("-c", "--column", default="ID")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--email")  # dla pubmed
    args = p.parse_args()
    main(args)


# python src/fetch_data.py \
#   --mode pubmed \
#   --dataset ./data/raw/amyloid-raw-13-08-2025.csv \
#   --column PMID \
#   --output ./data/fetched/pubmed-amyloid-13-08-2025.csv \
#   --email test@example.com
