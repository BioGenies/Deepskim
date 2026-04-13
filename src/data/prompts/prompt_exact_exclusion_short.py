def get_prompt(title, abstract, journal=None, max_abstract_len=450):
    # len_abstract = len(abstract.split())
    # limited_abstract = ' '.join(abstract.split()[:max_abstract_len])
    # if len_abstract > max_abstract_len:
    #     print(f"Abstract truncated from {len_abstract} to {max_abstract_len} words.")
    #     print("Current length of abstract:", len(limited_abstract.split()))
    return f"""Task: Decide if this PubMed article should be INCLUDED in AmyloGraphAB, a curated database of studies reporting the effects of antibodies on amyloid formation.

Inclusion criteria (answer "yes" only if ALL are satisfied):
1) The study reports the effect of antibodies on amyloid formation (i.e. antibodies are experimentally investigated as modulators of amyloid aggregation/fibrillation/formation).
2) The study includes experimental data on amyloid fibrillation measured using at least one of the following techniques:
   - AFM (atomic force microscopy)
   - PET (positron emission tomography)
   - ThT (Thioflavin T assay/fluorescence)
   - TEM (transmission electron microscopy)

Exclusion criteria (answer "no" if ANY apply):
- The article is a pre-print (not a peer-reviewed publication).
- The paper is not in English.
- The article is a review, commentary, perspective, or editorial (non-primary research).

When including (the answer is "yes"), the reason is ALWAYS Rn.

When excluding (the answer is "no"), choose one reason (label between R0-RA) from the following codebook:
R0 — There are no interactions described
R1 — The interactor is not an antibody
R2 — Not enough experimental data
R3 — The interactee is not an amyloid protein
R4 — (Pre)Clinical trials. No interaction or amyloid data
R5 — Non-English paper
R6 — In silico information only
R7 — Pre-print
R8 - Review article
R9 - Unknown antibody type
RA — Other
Choose RA only if no other exclusion label fits.

Output format:
Respond with a single yes/no and a reason. Examples: 
* "no, reason: R0"
* "yes, reason: Rn"


Input:
Journal: {journal if journal else "Unknown"}
Title: {title}
Abstract: {' '.join(abstract.split()[:max_abstract_len])}

Question:
Based on the criteria above, is this article relevant for inclusion in AmyloGraphAB? The answer is """
