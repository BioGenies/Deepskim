# pip install chromadb sentence-transformers pandas

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from numpy.linalg import norm


class RAGHelper:
    """
    Vector-backed example store for few-shot RAG in the biomedical domain.
    Each DF row should have: title, abstract, label
    Optional cols: references (list[str] or str), id, source, pub_year
    """

    def __init__(
        self,
        collection_name: str = "biomed_examples",
        persist_dir: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        metadata_fields_to_index: Optional[List[str]] = None,
    ):
        """
        persist_dir:
          - None -> in-memory (ephemeral) client
          - string path -> persistent client on disk
        """
        self.embed_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
        self.client = (
            chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.Client()
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
        )

    # -----------------------------
    # Formatting helpers
    # -----------------------------
    @staticmethod
    def _fmt_embed_text(item: dict) -> str:
        parts = [f"Title: {item["Title"]}", f"Abstract: {item["Abstract"]}"]
        if item.get("References"):
            if isinstance(item["references"], str):
                # Allow a single string with separators; store as-is
                parts.append(f"References: {item["References"]}")
            else:
                parts.append("References: " + "; ".join(item["References"]))
        return "\n".join(parts)

    @staticmethod
    def _fmt_prompt_example(item: dict) -> str:
        resp_dict = {1: "YES", 0: "NO"}
        ref = ""
        if item.get("References"):
            if isinstance(item["References"], str):
                ref = f"\nReferences: {item["References"]}"
            else:
                ref = "\nReferences: " + "; ".join(item["References"])
        return (
            f"### Example\n"
            f"Title: {item["Title"]}\n"
            f"Abstract: {item["Abstract"]}{ref}\n"
            f"References: {ref if ref else "None"}\n"
            f"Answer: {resp_dict[item["Label"]]}"
        )
    


    ### Answer

    def upsert_data(self, data: list[dict]) -> int:
        """
        Converts each dictionary to an entry and upserts into Chroma.
        Returns the total count of vectors after upsert.
        """
        ids, documents, metadatas = [], [], []

        for idx, item in enumerate(data):
            ids.append(str(idx))
            documents.append(self._fmt_embed_text(item))  # what gets embedded
            metadatas.append(
                {
                    "label": item["Label"],              # keep labels ONLY in metadata/prompt exemplar
                    "prompt_example": self._fmt_prompt_example(item),  # used to assemble few-shot
                    "title": item["Title"],              # convenient for audits
                }
            )

        # Chroma will embed documents via embedding_function, then index
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return self.collection.count()

    # -----------------------------
    # Retrieval (with light MMR + label balancing)
    # -----------------------------
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (norm(a) * norm(b))
        return float(a @ b / denom) if denom else 0.0

    def _mmr_select(
        self,
        query_vec: np.ndarray,
        cand_vecs: np.ndarray,
        cand_indices: List[int],
        k: int,
        lambda_mult: float = 0.7,
    ) -> List[int]:
        """Simple greedy MMR selection on candidate embeddings."""
        selected = []
        remaining = set(cand_indices)
        # Precompute similarity to query
        sim_to_q = {i: self._cosine(query_vec, cand_vecs[j]) for j, i in enumerate(cand_indices)}

        while remaining and len(selected) < k:
            best_i, best_score = None, -1e9
            for i in list(remaining):
                # max similarity to already selected (diversity term)
                if not selected:
                    div = 0.0
                else:
                    div = max(
                        self._cosine(cand_vecs[cand_indices.index(i)], cand_vecs[cand_indices.index(s)])
                        for s in selected
                    )
                score = lambda_mult * sim_to_q[i] - (1 - lambda_mult) * div
                if score > best_score:
                    best_score, best_i = score, i
            selected.append(best_i)
            remaining.remove(best_i)
        return selected

    def retrieve_similar_examples(
        self,
        new_item: Dict[str, Any],
        k: int = 6,
        pool: int = 30,
        balance_labels: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        new_item: dict with keys title, abstract, optional references
        Returns a list of metadatas (each includes 'prompt_example' and 'label').
        """
        item_dct = new_item
        item_dct.update({"id": "tmp", "label": "NA"}, )
        q_embed_text = self._fmt_embed_text(item_dct)

        out = self.collection.query(
            query_texts=[q_embed_text],
            n_results=pool,
            include=["documents", "metadatas", "embeddings", "distances"],
        )

        ids = out["ids"][0]
        metas = out["metadatas"][0]
        embs = np.array(out["embeddings"][0]) if "embeddings" in out and out["embeddings"] else None
        # Query vector (recompute via embed_fn for MMR)
        q_vec = np.array(self.embed_fn([q_embed_text])[0])

        # De-duplicate by exact title if present
        seen_titles, uniq_idx = set(), []
        for j, m in enumerate(metas):
            t = (m or {}).get("title")
            if t and t in seen_titles:
                continue
            seen_titles.add(t)
            uniq_idx.append(j)

        # Light MMR to diversify
        pick_idx = uniq_idx
        if embs is not None and len(uniq_idx) > k:
            cand_indices = uniq_idx
            selected = self._mmr_select(q_vec, embs, cand_indices, k=min(len(cand_indices), max(k, k*2)))
            pick_idx = selected

        # Optional class balance (post-order)
        chosen = [metas[j] for j in pick_idx]
        if balance_labels:
            yes = [m for m in chosen if m.get("label") == "Yes"]
            no  = [m for m in chosen if m.get("label") == "No"]
            half = k // 2
            balanced = yes[:half] + no[: (k - len(yes[:half]))]
            if len(balanced) < k:
                # backfill from remaining (either class)
                remaining = [m for m in chosen if m not in balanced]
                balanced.extend(remaining[: (k - len(balanced))])
            chosen = balanced[:k]
        else:
            chosen = chosen[:k]

        return chosen

    # # -----------------------------
    # # Prompt assembly
    # # -----------------------------
    # @staticmethod
    # def build_prompt(
    #     new_item: Dict[str, Any],
    #     examples: List[Dict[str, Any]],
    #     criteria_text: str = "Decide if the answer is Yes or No only based on biomedical relevance.",
    #     ask_reason: bool = False,
    # ) -> str:
    #     ex_text = "\n\n".join(m["prompt_example"] for m in examples if m.get("prompt_example"))
    #     ref = ""
    #     if new_item.get("references"):
    #         if isinstance(new_item["references"], str):
    #             ref = f"\nReferences: {new_item['references']}"
    #         else:
    #             ref = "\nReferences: " + "; ".join(new_item["references"])

    #     query_block = (
    #         f"### Query\n"
    #         f"Title: {new_item['title']}\n"
    #         f"Abstract: {new_item['abstract']}{ref}"
    #     )
    #     system = (
    #         "You are a biomedical reviewer. Output exactly 'Yes' or 'No'.\n"
    #         f"Criteria: {criteria_text}"
    #     )
    #     suffix = "\n\nAnswer (Yes/No only):" if not ask_reason else "\n\nAnswer: <Yes|No>\nReason (1–2 sentences):"
    #     return f"{system}\n\n{ex_text}\n\n{query_block}{suffix}"

    # -----------------------------
    # Utilities
    # -----------------------------
    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        self.collection.delete(where={})  # delete all
