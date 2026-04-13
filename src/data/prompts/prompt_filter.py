def get_prompt(title, abstract, refs, journal):
    return f"""Task: Decide if this PubMed article reports *experimental antibody research* on *amyloid aggregation* in *neurodegenerative diseases* (Alzheimer's, Parkinson's, ALS, or FTD).

Respond only with one word: yes or no (lowercase, no punctuation).

Criteria for "yes":
- The other species is an antibody directly studied in experiments (not just mentioned as a future tool or potential immunogen).
- The antibody directly interacts with amyloid species modulating neurodegeneration.
- Includes experimental results (not purely computational).

Respond "no" if at least one of the criteria above is not satisfied.

Examples:
Antibody inhibits Aβ aggregation in Alzheimer's models: yes
Docking of small molecules blocking amyloid fibrils: no
Amyloid aggregation of prion protein with chaperones": no
Amyloid aggregation inhibited by interaction with another protein: no
Amyloid-binding chaperone NUCB1 inhibits fibrillization but no antibodies tested: no


Input:
Journal: {journal}
Title: {title}
Abstract: {abstract}
References: {refs if refs else "None"}

Is the article relevant based on the criteria above? The answer is"""