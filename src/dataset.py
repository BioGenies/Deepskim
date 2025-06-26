import pandas as pd

from utils import fetch_pubmed_data
from config import RAW_DIR, PROCESSED_DIR, EMAIL


class DatasetPreprocessor:
    def __init__(self, filepath, pmid_col, label_col, columns_to_keep):
        """
        filepath: path to the CSV file
        pmid_col: name of the column containing the publication identifier (PMID)
        label_col: name of the column containing the label (e.g. 'rejection')
        columns_to_keep: list of columns to keep (including pmid and label)
        """
        self.filepath = filepath
        self.pmid_col = pmid_col
        self.label_col = label_col
        self.columns_to_keep = columns_to_keep
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        return self

    def rename_columns(self, rename_dict):
        self.df = self.df.rename(columns=rename_dict)
        return self

    def select_columns(self):
        self.df = self.df[self.columns_to_keep]
        return self

    def filter_labels(self, allowed_labels):
        self.df = self.df[self.df[self.label_col].isin(allowed_labels)]
        return self

    def exclude_by_reason(self, reason_col, excluded_reasons):
        if reason_col in self.df.columns:
            self.df = self.df[~self.df[reason_col].isin(excluded_reasons)]
        return self

    def get_dataframe(self):
        return self.df


# Constants for preprocessing
PMID_COL = "PMID"
LABEL_COL = "rejection"
COLUMNS_TO_KEEP = ["PMID", "rejection", "reason", "decision"]
RENAME_MAP = {
    "Rejection?": "rejection",
    "If so; reason to reject?": "reason",
    "Decided by what?": "decision",
}
FILTER_LABELS = ["Rejected", "Useful"]
EXCLUDE_REASONS = ["Review article"]


def main():
    # File names (manually defined)
    raw_file = RAW_DIR / "amyloid-raw-19-06-2025.csv"
    processed_file = PROCESSED_DIR / "amyloid-19-06-2025.csv"

    # Check if processed file already exists
    if processed_file.exists():
        print(f"Warning: {processed_file} already exists. Skipping processing.")
        return

    # Ensure processed directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load and preprocess raw data
    print(f"Loading raw data from {raw_file}")
    preprocessor = (
        DatasetPreprocessor(raw_file, PMID_COL, LABEL_COL, COLUMNS_TO_KEEP)
        .load_data()
        .rename_columns(RENAME_MAP)
        .select_columns()
        .filter_labels(FILTER_LABELS)
        .exclude_by_reason("reason", EXCLUDE_REASONS)
    )
    filtered_df = preprocessor.get_dataframe()
    print(f"Filtered down to {len(filtered_df)} records")

    # Fetch metadata from PubMed
    print(f"Fetching PubMed data for {len(filtered_df)} PMIDs")
    metadata_df = fetch_pubmed_data(filtered_df[PMID_COL], email=EMAIL)

    # Merge and save
    filtered_df[PMID_COL] = filtered_df[PMID_COL].astype(str)
    metadata_df[PMID_COL] = metadata_df[PMID_COL].astype(str)
    merged_df = pd.merge(filtered_df, metadata_df, on=PMID_COL, how="left")

    print(f"Saving processed data to {processed_file}")
    merged_df.to_csv(processed_file, index=False)


if __name__ == "__main__":
    main()
