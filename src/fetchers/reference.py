import requests
import pandas as pd
from typing import List, Union
from tqdm import tqdm


def fetch_references(pmids: Union[List[str], pd.Series]) -> pd.DataFrame:
    """
    Pobiera listę referencji z OpenAlex dla podanych PMIDs.

    Parameters
    ----------
    pmids : Union[List[str], pd.Series]
        Lista lub pandas Series z identyfikatorami PMID.

    Returns
    -------
    pd.DataFrame
        Ramka danych z kolumnami:
        - pmid: oryginalny PMID
        - references: lista identyfikatorów prac referencyjnych (np. "W3196964953")
    """
    # Upewnij się, że mamy listę stringów
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

            # Pobieramy listę referencji
            oa_ids = data.get("referenced_works", [])
            # Wyciągamy tylko część po ostatnim ukośniku
            refs = [oa_id.rsplit("/", 1)[-1] for oa_id in oa_ids]
        except requests.HTTPError as e:
            print(f"Błąd HTTP dla PMID {pmid}: {e}")
            refs = []
        except Exception as e:
            print(f"Inny błąd dla PMID {pmid}: {e}")
            refs = []

        results.append({"pmid": pmid, "references": refs})

    df = pd.DataFrame(results)
    return df


def fetch_references_by_openalex(
    openalex_ids: Union[List[str], pd.Series],
) -> pd.DataFrame:
    """
    Fetches corresponding PMIDs, DOIs, and a list of references for given OpenAlex IDs.
    Always displays a progress bar using tqdm.

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
