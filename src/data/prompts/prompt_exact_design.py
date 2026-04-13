def get_prompt(title, abstract, journal=None, max_abstract_len=450):
    # len_abstract = len(abstract.split())
    # limited_abstract = ' '.join(abstract.split()[:max_abstract_len])
    # if len_abstract > max_abstract_len:
    #     print(f"Abstract truncated from {len_abstract} to {max_abstract_len} words.")
    #     print("Current length of abstract:", len(limited_abstract.split()))
    return f"""Task: Decide if this PubMed article should be INCLUDED in AmyloGraphAB, a curated database of studies reporting the effects of antibodies on amyloid formation.

Background (context for the task):
- Amyloids are fibrous protein aggregates with β-sheet structure that assemble into long, unbranched fibrils.
- Their accumulation in tissues and organs can impair normal function and lead to diseases known as amyloidosis.
- AmyloGraphAB collects and systematizes data on interactions between amyloids and antibodies, focusing on studies that report how antibodies affect amyloid formation.
- Articles were retrieved by keyword search; retrieval does not imply relevance. Decide only using the criteria below.

Inclusion criteria (answer "yes" only if ALL are satisfied):
1) The study reports the effect of antibodies on amyloid formation (i.e., antibodies are experimentally investigated as modulators of amyloid aggregation/fibrillation/formation).
2) The study includes experimental data on amyloid fibrillation measured using at least one of the following techniques:
   - AFM (atomic force microscopy)
   - PET (positron emission tomography)
   - ThT (Thioflavin T assay/fluorescence)
   - TEM (transmission electron microscopy)

Exclusion criteria (answer "no" if ANY apply):
- The article is a pre-print (not a peer-reviewed publication).
- The article is a duplicate of a paper already included.
- The paper is not in English.
- The article is a review, commentary, perspective, or editorial (non-primary research).

Output format:
Respond with exactly one word: yes or no (lowercase, no punctuation, no extra text).

Input:
Journal: {journal if journal else "Unknown"}
Title: {title}
Abstract: {' '.join(abstract.split()[:max_abstract_len])}

Question:
Based on the criteria above, is this article relevant for inclusion in AmyloGraphAB? The answer is """
