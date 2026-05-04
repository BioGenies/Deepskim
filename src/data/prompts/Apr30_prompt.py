def get_prompt(title, abstract, journal=None, max_abstract_len=450):
    return f"""Task: Decide if this PubMed article should be INCLUDED in AmyloGraphAB, a curated database of studies reporting the effects of antibodies on amyloid formation. Select a reason from the reason codebook.
"Antibody" covers monoclonal/polyclonal antibodies, antibody fragments (e.g., scFv, Fab, F(ab')2, nanobody), pooled immunoglobulin preparations, vaccination-elicited antibodies when the paper credits them with the effect, antibody-conjugated delivery systems where the antibody is the targeting/effector moiety, and bispecific or bifunctional antibodies provided at least one arm targets an amyloid protein.

Each code has a characteristic abstract concept pattern; classify by which pattern best fits.

R4 — Review / commentary / perspective / editorial / viewpoint / news / drug-approval summary, or a paper consisting only of computational or PK-PD modelling.
Pattern: the abstract summarises, discusses, models, or comments on prior work; no primary wet-lab experiment characterises an antibody's effect on amyloid.

R1 — A non-antibody agent is administered or applied as the perturbing agent, its effect is the subject of the study, and no antibody is being tested as a perturbing agent alongside it.
Pattern: the abstract reports administering / treating with / dosing / applying a small molecule, peptide (non-antibody), gene/mRNA construct, cell therapy, plain particle, or other non-antibody compound, and measures its effects. Antibodies, if present, appear only as readout reagents (IHC/WB/ELISA/imaging) or as background framing of the disease area.
Boundary cue: mere mention of non-antibody compounds (clinical context, comparators, controls) is NOT R1 — there must be an actively tested non-antibody perturbation, and no antibody perturbation tested alongside it. If both a non-antibody agent and an antibody are tested as perturbations, do not assign R1; let the article flow to R3 / R2 / Rn.

R3 — An antibody is administered or tested as the perturbing agent, but its direct antigen is not an amyloid protein; OR an immunoglobulin / light chain is itself the amyloid-forming species and no antibody is being tested as a perturbation against it.
Pattern A (non-amyloid target): the abstract describes administering / treating with / dosing an antibody (or vaccination eliciting antibodies) whose direct target is a non-amyloid receptor, processing enzyme, regulator, or signalling molecule (e.g., BACE1, γ-secretase, RAGE, microglial Fc or complement receptors). Amyloid effects, if measured, are downstream of the direct binding event.
Pattern B (Ig-as-amyloid): the abstract concerns light-chain (AL) amyloidosis or related conditions in which an immunoglobulin or its fragment is the amyloid-forming species, and no antibody is being tested as a perturbing agent against that amyloid.
Boundary cue: R3 requires either Pattern A or Pattern B above. If neither holds and antibodies appear only as instruments, the paper is R0.

R0 — The antibody serves as a research instrument, not as a perturbing agent.
Pattern: the abstract reports using antibodies to detect, visualise, image, quantify, stain, label, characterise, or determine the structure of something — the antibody is a tool the experimenter uses, not the agent whose effect is tested. Includes diagnostic studies, imaging-probe / radioligand development (including radioisotope-conjugated antibodies used as imaging probes), structural biology of antibody–antigen complexes used as probes, biosensor development, epitope-mapping work. No perturbation by any agent is the subject of the study.
Boundary cue: characteristic verbs are "we used / we developed / we generated / we determined the structure of / we characterised / we detected / we imaged"; absent any "we administered / we treated / we dosed".

R2 — An antibody is tested against an amyloid protein, but its impact on the amyloid forming is not quantified by an aggregation-specific assay.
Sub-patterns (any one is sufficient for R2):
  • binding-only — affinity, kinetics, thermodynamics, or structure of the antibody–antigen complex with no aggregation data;
  • clinical-only — clinical-trial outcomes (cognitive, behavioural, functional scores) or soluble biomarkers reported without amyloid burden quantification;
  • computational-only — in silico modelling without wet-lab aggregation data;
  • indirect-only — cytokine, behavioural, or pathology-marker readouts that are not aggregation-specific.

Rn — Primary research in which an antibody is administered or tested against an amyloid protein, and the abstract reports an aggregation-specific readout.
Aggregation-specific readouts include: direct biophysical assays of fibrils (kinetics, mass, morphology); quantitative imaging or histology of amyloid deposits / plaques / fibrils; conformation- or aggregation-state-specific detection used as a pathology readout; fibril disaggregation, dissolution, or seeding-amplification assays; quantitative amyloid imaging used as an outcome measure.

Apply the codebook to the abstract content alone. Do not defer to any prior label, source tag, or external annotation. Judge entities by the role they play in the study (perturbing agent vs. readout reagent; antigen vs. downstream protein; aggregation readout vs. non-aggregation readout) rather than by recognising specific names.

Decision priority when more than one code could apply: R4 → R1 → R3 → R0 → R2 → Rn.

Output format: a decision (yes/no) followed by the reason code.
Examples: "no R0", "yes Rn".

Input:
Journal: {journal if journal else "Unknown"}
Title: {title}
Abstract: {' '.join(abstract.split()[:max_abstract_len])}

Question: Based on the criteria above, classify this article using the reason codebook. """
