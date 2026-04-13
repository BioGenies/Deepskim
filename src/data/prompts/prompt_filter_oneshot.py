def get_prompt(title, abstract, refs, journal):
    return f"""Task: Decide if this PubMed article reports experimental antibody research on amyloid aggregation in neurodegenerative diseases (Alzheimer's, Parkinson's, ALS, or FTD).

Respond only with one word: yes or no (lowercase, no punctuation).

Criteria for "yes":
- The other species is an antibody directly studied in experiments (not just mentioned as a future tool or potential immunogen).
- The antibody directly interacts with amyloid species modulating neurodegeneration.
- Includes experimental results (not purely computational).

Respond "no" if at least one of the criteria above is not satisfied.

Input:
Journal: {journal}
Title: {title}
Abstract: {abstract}
References: {refs if refs else "None"}

Is the article relevant based on the criteria above? The answer is"""