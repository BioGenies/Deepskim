# NOTE: "Not enough experimental data" is DELIBERATELY ABSENT (R2 retired, Jul13).
# It is an ABSENCE of evidence, not a content reason for exclusion — the handbook's own
# term for the undecidable case. Rows carrying it belong in `unclear` -> the `maybe`
# token, never in `no R{code}`. It had ZERO decided-exclude rows in v4 and v5 anyway.
# Leaving it mapped here would let any future such row silently emit `no R2`, a code the
# model has no training signal for — the exact laundering this pipeline exists to prevent.
# An exclude row with an unmapped reason is DROPPED by _create_completion, loudly.
exclusion_reason_map = {
    "There are no interactions described": "R0",
    "The interactor is not an Ab": "R1",
    # In-silico-only papers are a decidable auto-exclusion — the handbook lists
    # "Only in silico data" alongside reviews, and the codebook's R4 explicitly
    # covers "a paper consisting only of computational, in-silico or PK-PD modelling".
    "In silico information only": "R4",
    "The interactee is not an amyloid protein": "R3",
    "Review article": "R4",
}
