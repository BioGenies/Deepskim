def get_prompt(title, abstract, journal=None, max_abstract_len=450):
    return f"""Task: Decide if this PubMed article should be INCLUDED in AmyloGraphAB, a curated database of studies reporting the effects of antibodies on amyloid formation. Select a reason from the reason codebook.

Reason codebook:
R0 — No antibody-amyloid interaction described
R1 — The interactor is not an antibody
R2 — Amyloid fibrillation not measured by ThT, AFM, TEM, or PET
R3 — The interactee is not an amyloid protein
R4 — Review, commentary, perspective, editorial, or pre-print (non-primary research)
RA — Other exclusion reason
Rn — Meets all criteria: primary research; antibody modulates amyloid aggregation AND fibrillation measured by ThT/AFM/TEM/PET

Decision rule:
- If reason is Rn, the article should be INCLUDED
- If reason is R0–RA, the article should be EXCLUDED
- Choose RA only if no other label fits.

Output format:
Respond with a single reason code. Examples:
* "R0"
* "Rn"

Input:
Journal: {journal if journal else "Unknown"}
Title: {title}
Abstract: {' '.join(abstract.split()[:max_abstract_len])}

Question:
Based on the criteria above, classify this article using the reason codebook. """
