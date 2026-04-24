def get_prompt(title, abstract, journal=None, max_abstract_len=450):
    return f"""Task: Decide if this PubMed article should be INCLUDED in AmyloGraphAB, a curated database of studies reporting the effects of antibodies on amyloid formation. Select a reason from the reason codebook.

Reason codebook:
R1 — The interactor is not an antibody.
R3 — The interactee is not an amyloid protein.
R4 — Review, commentary, perspective, or editorial (non-primary research).
R0 — Primary research, but no antibody–amyloid interaction is described.
R2 — Antibody–amyloid interaction is described, but amyloid fibrillation is not quantified by amyloid-quantification means (e.g., ThT, AFM, TEM, PET, or another fibrillation-specific assay). Includes in silico / computational only, binding assays only (SPR, ITC, ELISA) with no fibrillation quantification, clinical/preclinical efficacy without fibrillation quantification
Rn — Meets all criteria.
Decision rule:
- If the article should be INCLUDED, the reason is Rn
- If the article should be EXCLUDED, the reason is R0–R4

Output format:
Respond with a decision (yes/no) followed by the reason code. Examples:
* "no R0"
* "yes Rn"

Input:
Journal: {journal if journal else "Unknown"}
Title: {title}
Abstract: {' '.join(abstract.split()[:max_abstract_len])}

Question:
Based on the criteria above, classify this article using the reason codebook. """
