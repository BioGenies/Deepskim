def get_prompt(title, abstract, refs, journal):
    return f"""
    ### Instruction
    You are a screening biologist. Decide whether a given publication is useful for research on amyloid proteins. Examples of scenarios include:
        1) Amyloids in neurodegenerative diseases (e.g., Alzheimer's, Parkinson's).
        2) Antibodies as therapeutic agents targeting amyloid aggregation.
        3) Antibodies as emerging therapuetic agents in neurodegenerative diseases.   
    Respond with EXACTLY one token: yes or no. No punctuation or extra text.
    Answer only if the study investigates an interaction between an antibody and an explicitly named amyloid protein. Otherwise answer no.

    ### Inclusion Criteria (ALL must hold)
    The study is published in a peer-reviewed journal.
    The study reports new, original experimental data (wet-lab or computational with experimental validation).
    The study investigates interactions between explicitly named amyloid proteins (e.g., Aβ, α-synuclein, tau, TDP-43) and antibodies (monoclonal, polyclonal, or engineered).
    The study describes the effect of this antibody–amyloid interaction on aggregation, fibril formation, seeding, clearance, or cellular toxicity.

    ### Exclusion Criteria (ANY implies no)
    The interaction partner is not an antibody (e.g., small molecules, chaperones, Nucleobindin, peptides, or other proteins).
    The study is a review, meta-analysis, clinical, epidemiological, or purely computational (without experimental validation).
    The study is not written in English.
    Published before 1995.
    The antibody or amyloid species is unspecified or unclear.
    The focus is outside neurodegenerative amyloids, such as prion diseases, bacterial amyloids, or systemic amyloidoses unless directly related to Alzheimer’s, Parkinson’s, ALS, or FTD.

    ### Input
    Journal: {journal}
    Title: {title}
    Abstract: {abstract}
    References: {refs if refs else "None"}
    Answer: """
