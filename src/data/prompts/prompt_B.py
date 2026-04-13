def get_prompt(title, abstract, refs, journal):
    return f"""
    ### Instruction
    You are a screening biologist. Decide whether a given publication is useful for research on interactions between amyloid proteins and antibodies.
    Respond with EXACTLY one token: YES or NO. No punctuation or extra text.
    If any required information is missing or you are unsure, answer NO.

    ### Inclusion Criteria (ALL must hold)
    1) Published in a peer-reviewed journal.
    2) Reports new original experimental data (wet-lab or computational with experimental validation).
    3) Studies interactions between explicitly named amyloid proteins (e.g., Aβ, α-synuclein, tau) and antibodies.
    4) Describes the effect of this interaction on amyloid aggregation (kinetics, fibril formation, or toxicity).

    ### Exclusion Criteria (ANY implies NO)
    1) Not focused on amyloid aggregation or related mechanisms.
    2) Review, meta-analysis, theoretical, clinical, epidemiological, or purely in silico study.
    3) Not written in English.
    4) Published before 1995.
    5) Antibody or amyloid species unspecified/unknown.
    6) Focus on non-amyloid diseases/conditions.
    7) References do not include at least one primary research article supporting the claims.

    ### Input
    Journal: {journal}
    Title: {title}
    Abstract: {abstract}
    References: {refs if refs else "None"}

    ### Answer"""
