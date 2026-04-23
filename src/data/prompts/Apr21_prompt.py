def get_prompt(title, abstract, journal=None, max_abstract_len=450):
    return f"""Task: Decide if this PubMed article should be INCLUDED in AmyloGraphAB, a curated database of studies reporting the effects of antibodies on amyloid formation. Select a reason from the reason codebook.

Reason codebook:
R0 — No antibody-amyloid interaction described
R1 — The interactor is not an antibody
R2 — Amyloid fibrillation not measured by ThT, AFM, TEM, or PET
R3 — The interactee is not an amyloid protein
R4 — Review, commentary, perspective, or editorial (non-primary research)
Rn — Meets all criteria: primary research; antibody modulates amyloid aggregation AND fibrillation measured by ThT/AFM/TEM/PET

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
