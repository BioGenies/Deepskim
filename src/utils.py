import html
import ssl
from xml.etree import ElementTree as ET

import pandas as pd
from Bio import Entrez


def fetch_pubmed_data(pmids, email, batch_size=200, api_key=None):
    """
    Fetches from PubMed for a list of PMIDs: title, abstract, authors, journal, and list of references.

    Parameters:
    - pmids: pandas Series or list of PMIDs (str or int)
    - email: Your email required by NCBI Entrez
    - batch_size: number of PMIDs sent at once (default 200)
    - api_key: optional NCBI API key

    Returns:
    - pandas DataFrame with columns: PMID, Title, Abstract, Authors (list), Journal, References (list of PMIDs)
    """

    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    # Make sure it's a list of unique IDs as strings
    ids = [str(int(x)) for x in pd.Series(pmids).dropna().unique()]
    results = []

    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        handle = Entrez.efetch(db="pubmed", id=",".join(batch), retmode="xml")
        tree = ET.parse(handle)
        root = tree.getroot()

        for article in root.findall(".//PubmedArticle"):
            medline = article.find("MedlineCitation")
            pmid = medline.findtext("PMID")

            art = medline.find("Article")

            # Title - improved extraction with HTML entities and tags handling
            title_elem = art.find("ArticleTitle")
            if title_elem is not None:
                # Use tostring with method='text' to extract all text without tags
                title = ET.tostring(
                    title_elem, method="text", encoding="unicode"
                ).strip()
                # Decode HTML entities (e.g. &#x3b5; -> ε)
                title = html.unescape(title)
                # Clean up extra whitespace
                title = " ".join(title.split())
            else:
                title = ""

            # Abstract - also with improved entities handling
            abstract_nodes = art.findall("Abstract/AbstractText")
            if abstract_nodes:
                abstract_parts = []
                for n in abstract_nodes:
                    text = ET.tostring(n, method="text", encoding="unicode").strip()
                    text = html.unescape(text)
                    if text:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)
                abstract = " ".join(abstract.split())  # Normalize whitespace
            else:
                abstract = ""

            # Authors
            authors = []
            for a in art.findall("AuthorList/Author"):
                lastname = a.findtext("LastName")
                initials = a.findtext("Initials")
                if lastname and initials:
                    authors.append(f"{lastname} {initials}")
                elif a.findtext("CollectiveName"):
                    authors.append(a.findtext("CollectiveName"))

            # Journal
            journal = art.findtext("Journal/Title") or ""

            # References
            refs = []
            for ref in article.findall(".//ReferenceList/Reference"):
                for aid in ref.findall(".//ArticleId"):
                    if aid.get("IdType") == "pubmed" and aid.text:
                        refs.append(aid.text)

            results.append(
                {
                    "PMID": pmid,
                    "Title": title,
                    "Abstract": abstract,
                    "Authors": authors,
                    "Journal": journal,
                    "References": refs,
                }
            )

        handle.close()

    return pd.DataFrame(results)
