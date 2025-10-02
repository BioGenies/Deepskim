"""
Integrator (class): keeps the original DataFrame and the reduced DataFrame in memory
(after loading at initialization and after calling reduce_columns). Other methods
operate on these objects in memory (fetching, merge).
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional

import warnings
import pandas as pd

from fetchers import fetch_pubmed_data
from fetch_data import load_pmid_list


class Integrator:
    """
    Class integrating a dataset with data fetched from PubMed.

    After initialization, self.original_df contains the entire loaded DataFrame.
    After calling reduce_columns, the result is stored in self.reduced_df.
    After calling fetch_pubmed, the result is stored in self.fetched_df.
    After calling merge, the result is stored in self.merged_df.

    """

    def __init__(
        self,
        raw_path: str | Path,
        pmid_col: str = "PMID",
        email: Optional[str] = "test@gmail.com",
    ):
        self.raw_path = Path(raw_path)
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw dataset not found: {self.raw_path}")
        self.pmid_col = pmid_col
        self.email = email

        # DataFrames stored in memory
        self.original_df: pd.DataFrame = self._read_df(self.raw_path)
        self.reduced_df: Optional[pd.DataFrame] = None
        self.fetched_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None

        # optional save paths (if the user saves results)
        self.reduced_df_path: Optional[Path] = None
        self.fetched_df_path: Optional[Path] = None
        self.merged_df_path: Optional[Path] = None

    def _detect_delim(self, path: Path) -> str:
        return "	" if path.suffix.lower() in {".tsv", ".txt"} else ","

    def _read_df(self, path: Path) -> pd.DataFrame:
        delim = self._detect_delim(path)
        return pd.read_csv(path, delimiter=delim)

    def reduce_columns(
        self,
        keep_columns: Iterable[str],
        save_path: Optional[str | Path] = None,
        drop_duplicates: bool = False,
    ) -> None:
        """
        Creates a reduced DataFrame based on self.original_df and stores it in self.reduced_df.
        Optionally saves to a CSV file.
        """
        cols = list(keep_columns)
        missing = [c for c in cols if c not in self.original_df.columns]
        if missing:
            raise KeyError(f"Missing columns in raw dataset: {missing}")
        reduced = self.original_df.loc[:, cols].copy()
        if drop_duplicates:
            reduced = reduced.drop_duplicates()
        self.reduced_df = reduced
        if save_path:
            save_p = Path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            reduced.to_csv(save_p, index=False)
            self.reduced_df_path = save_p

    def fetch_pubmed(
        self,
        dataset_path_for_ids: Optional[str | Path] = None,
        pmid_column: Optional[str] = None,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """
        Fetches PubMed data for a list of PMIDs. By default uses self.original_df

        If dataset_path_for_ids is provided as a path, the load_pmid_list function
        will be used to load PMIDs from the file.
        """
        if fetch_pubmed_data is None:
            raise RuntimeError(
                "Function fetch_pubmed_data is not available (check imports)."
            )

        pmid_col = pmid_column or self.pmid_col

        # Obtaining the list of PMIDs:
        if dataset_path_for_ids:
            # if the user provided a path -> use load_pmid_list (preserves previously defined behavior)
            if load_pmid_list is None:
                raise RuntimeError(
                    "Function load_pmid_list is not available (check imports)."
                )
            ids = load_pmid_list(str(dataset_path_for_ids), pmid_col)
        else:
            # use the DataFrame stored in memory
            source_df = self.original_df
            if pmid_col not in source_df.columns:
                raise KeyError(
                    f"PMID column '{pmid_col}' does not exist in the original DataFrame."
                )
            ids = source_df[pmid_col].dropna().astype(str).tolist()

        if not ids:
            warnings.warn("The list of PMIDs is empty — nothing fetched.")
            fetched_df = pd.DataFrame()
        else:
            fetched_df = fetch_pubmed_data(ids, self.email)

        self.fetched_df = fetched_df
        if save_path:
            save_p = Path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            fetched_df.to_csv(save_p, index=False)
            self.fetched_df_path = save_p

    def merge(self, out_path: Optional[str | Path] = None) -> None:
        """
        Merge the fetched_df into the reduced/original dataset using a left join,
        where the left side is the reduced dataset (if available) or the original dataset.

        - Takes only an optional out_path argument.
        - Does not return a DataFrame; the result is stored in self.merged_df.
        - If self.reduced_df exists, it is used as the left table; otherwise self.original_df is used.
        - The join is performed on self.pmid_col.
        """
        if self.fetched_df is None:
            raise ValueError(
                "self.fetched_df is None — nothing to merge. Call fetch_pubmed first."
            )

        pmid = self.pmid_col

        # choose left-side DF: reduced if available else original
        reduced = self.reduced_df if self.reduced_df is not None else self.original_df

        if pmid not in reduced.columns:
            raise KeyError(
                f"PMID column '{pmid}' not found in the left dataset (reduced/original)."
            )
        if pmid not in self.fetched_df.columns:
            raise KeyError(f"PMID column '{pmid}' not found in fetched_df.")

        # ensure PMIDs are strings for stable merging
        reduced = reduced.copy()
        fetched = self.fetched_df.copy()
        reduced[pmid] = reduced[pmid].astype(str)
        fetched[pmid] = fetched[pmid].astype(str)

        # perform the requested single-line merge (left = reduced, right = fetched)
        merged_df = pd.merge(reduced, fetched, on=pmid, how="left")
        merged_df[pmid] = merged_df[pmid].astype(int)

        # update in-memory fetched_df with the merged result
        self.merged_df = merged_df

        # optionally save to disk
        if out_path:
            out_p = Path(out_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            self.merged_df.to_csv(out_p, index=False)
            self.merged_df_path = out_p
