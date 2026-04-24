def get_prompt(title, abstract, journal=None, max_abstract_len=450):
    return f"""Task: Decide if this PubMed article should be INCLUDED in AmyloGraphAB, a curated database of studies reporting the effects of antibodies on amyloid formation. Select a reason from the reason codebook.

Definitions:
- Interactor = the agent being tested for its effect on amyloid (the perturbing agent in the study). If the study tests a small-molecule drug, peptide, or nanoparticle, the interactor is that agent — not any antibody used for detection or quantification. Antibodies used only for IHC/WB/ELISA readouts are NOT the interactor.
- Amyloid proteins include: Aβ (amyloid-β), tau, α-synuclein, transthyretin (TTR), IAPP/amylin, prion protein, and other proteins known to form amyloid fibrils. They do NOT include: APP, APLP1/APLP2 (precursors/homologues), ApoE, nicastrin, NEP, or other proteins that are associated with amyloid pathology but do not themselves form amyloid fibrils.

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
- When multiple reasons could apply, use this priority: R4 → R1 → R3 → R0 → R2 → Rn. (i.e., if it is a review, stop at R4; else check if the interactor is an antibody; else check if the interactee is an amyloid protein; etc.)

Examples:
- Paper uses antibodies against CD3, CD8, and beta-amyloid for IHC in a muscle disease study → R2 (antibody–amyloid interaction exists; no fibrillation assay)
- Paper tests fasudil (drug) on tau mice; uses anti-pTau antibody only for readout → R1 (drug is the interactor, not an antibody)
- Paper solves crystal structure of anti-Aβ antibody bound to Aβ peptide → R2 (antibody–amyloid interaction; no fibrillation assay, just structure)

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
