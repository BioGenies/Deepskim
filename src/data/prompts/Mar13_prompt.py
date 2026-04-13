def get_prompt(title, abstract, journal=None, max_abstract_len=450):
    return f"""Task: Decide if this PubMed article should be INCLUDED in AmyloGraphAB, a curated database of studies reporting the effects of antibodies on amyloid formation.

Reason codebook:
Rn — Meets all criteria: antibody modulates amyloid aggregation AND fibrillation measured by ThT/AFM/TEM/PET
R0 — No antibody-amyloid interaction described
R1 — The interactor is not an antibody
R2 — Amyloid fibrillation not measured by ThT, AFM, TEM, or PET
R3 — The interactee is not an amyloid protein
R4 — Review, commentary, perspective, editorial, or pre-print (non-primary research)
RA — Other exclusion reason

Decision rule:
- If reason is Rn, then the answer is "yes"
- If reason is R0–RA, then the answer is "no"
- Choose RA only if no other label fits.

Output format:
Respond with yes/no then a reason code. Examples:
* "no, reason: R0"
* "yes, reason: Rn"

Input:
Journal: {journal if journal else "Unknown"}
Title: {title}
Abstract: {' '.join(abstract.split()[:max_abstract_len])}

Question:
Based on the criteria above, is this article relevant for inclusion in AmyloGraphAB? """
