import requests
import pandas as pd
from typing import List, Union
from tqdm import tqdm


def fetch_references(pmids: Union[List[str], pd.Series]) -> pd.DataFrame:
    """
    Fetches a list of references from OpenAlex for the given PMIDs.

    Parameters
    ----------
    pmids : Union[List[str], pd.Series]
        A list or pandas Series with PMID identifiers.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
        - pmid: original PMID
        - references: list of reference work identifiers (e.g. "W3196964953")
    """
    # Ensure we have a list of strings
    if isinstance(pmids, pd.Series):
        pmid_list = pmids.astype(str).tolist()
    else:
        pmid_list = [str(x) for x in pmids]

    results = []

    for pmid in pmid_list:
        url = f"https://api.openalex.org/works/pmid:{pmid}"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()

            # Extract the list of references
            oa_ids = data.get("referenced_works", [])
            # Keep only the part after the last slash
            refs = [oa_id.rsplit("/", 1)[-1] for oa_id in oa_ids]
        except requests.HTTPError as e:
            print(f"HTTP error for PMID {pmid}: {e}")
            refs = []
        except Exception as e:
            print(f"Other error for PMID {pmid}: {e}")
            refs = []

        results.append({"pmid": pmid, "references": refs})

    df = pd.DataFrame(results)
    return df


def fetch_references_by_openalex(
    openalex_ids: Union[List[str], pd.Series],
) -> pd.DataFrame:
    """
    Fetches corresponding PMIDs, DOIs, and a list of references for given OpenAlex IDs.

    Parameters
    ----------
    openalex_ids : Union[List[str], pd.Series]
        A list or pandas Series of OpenAlex identifiers (e.g., "W2741809807").

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
        - openalex_id: original OpenAlex ID
        - pmid: corresponding PMID (str) or None if not available
        - doi: corresponding DOI (str) or None if not available
        - references: list of reference work IDs (e.g., ["W3196964953", ...])
    """
    oa_list = (
        openalex_ids.astype(str).tolist()
        if isinstance(openalex_ids, pd.Series)
        else [str(x) for x in openalex_ids]
    )

    results = []

    for oid in tqdm(oa_list, desc="Fetching data from OpenAlex", unit="wk"):
        url = f"https://api.openalex.org/works/{oid}"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()

            pmid_uri = data.get("ids", {}).get("pmid")
            pmid = pmid_uri.rsplit("/", 1)[-1] if pmid_uri else None
            doi = data.get("doi")
            refs = [r.rsplit("/", 1)[-1] for r in data.get("referenced_works", [])]

        except requests.HTTPError as e:
            print(f"HTTP error for OpenAlex ID {oid}: {e}")
            pmid, doi, refs = None, None, []
        except Exception as e:
            print(f"Other error for OpenAlex ID {oid}: {e}")
            pmid, doi, refs = None, None, []

        results.append(
            {"openalex_id": oid, "pmid": pmid, "doi": doi, "references": refs}
        )

    return pd.DataFrame(results)
