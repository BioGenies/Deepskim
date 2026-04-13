def get_prompt(title, abstract, journal):
    return f"""You are screening PubMed articles.

Output exactly one word: yes or no (lowercase).

Answer "yes" ONLY if the abstract clearly shows ALL of the following:
1) Neurodegeneration context: amyloid refers to Aβ (amyloid-beta) or other neurodegenerative amyloid species (not serum amyloid A/SAA, AA amyloidosis, or non-neurodegenerative inflammation).
2) Animal model in vivo: experiments in non-human animals relevant to neurodegeneration (e.g., AD/amyloid mouse models).
3) Antibody intervention: an antibody targeting amyloid is administered/expressed/tested as the intervention in the animal study.
4) Outcome reported: effects on amyloid pathology/aggregation and/or neurodegeneration outcomes are reported in vivo.

Answer "no" if ANY is true:
- antibodies are used only for detection/assays (IHC, immunostaining, WB, ELISA) or “tested” only as staining reagents
- the antibody is only mentioned (e.g., donanemab/lecanemab) without in vivo antibody treatment experiments
- in vitro/cell-only, human-only, computational-only, review/commentary
- “amyloid” refers to SAA/SAA1 or unrelated targets

If the abstract does not explicitly state an in vivo amyloid-targeting antibody intervention, answer "no".

Journal: {journal}
Title: {title}
Abstract: {abstract}

Is the article relevant based on the criteria above? The answer is"""