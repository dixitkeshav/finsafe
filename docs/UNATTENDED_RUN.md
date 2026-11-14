# Unattended first-submission run

Use notebook 06 with **Save & Run All (Commit)**, not an interactive draft.
The job runs on Kaggle's servers after the laptop sleeps. Codex's local browser
and final organizer-form submission do not automatically continue through laptop
sleep. A finished Kaggle job is not a submitted hackathon entry.

Enable Internet, a T4 GPU and both secrets on this exact notebook:
`HF_TOKEN` (gated model read access) and `HF_WRITE_TOKEN` (Finsafe-team writes).
Never paste either into code. A versioned preflight on November 15 failed because
Kaggle could not find HF_WRITE_TOKEN for kernel 144975746; adding a secret to a
different account/notebook is insufficient. Verify with the preflight before launch.

The pipeline pins its source commit, restores the verified Version-5 data archive,
installs Python 3.12 and GPU dependencies, performs a real SafeTune GPU preflight,
then runs the aligned/SFT/steering comparison. It trains a new 100-step adapter;
the earlier five-step checkpoint remains only a smoke test. CuratorKIT processes
finance training and safety calibration, SafeTune provides fixed-layer inference
steering, AlignTune supplies finance SFT, and AuditKIT reports the measurements.

Four conditions: aligned, finance SFT, and SFT with steering strengths 5 and 20
at layer 18. Each gets 200 finance and 36 safety validation responses. Test data
remains reserved. Safety outputs use 768 tokens; finance uses 64, with identical
stopping configuration and decoding across conditions. No assumed deterioration
or successful repair is written into the results.

The automated safety judge is pinned original aligned Tiny Aya. Its rubric hides
condition names and prioritizes concrete harmful assistance over disclaimers.
A 16-example assistant-reviewed quality gate (including four synthetic invalid
fixtures) can stop safety scoring. This does not replace independent human
review and is explicitly disclosed. If the gate fails, the publisher reports
finance numbers and unavailable safety rates; manual assistant review is still
needed before a complete safety submission.

Full trainer state is backed up privately every ten steps; phase archives are
uploaded to Finsafe-team/finsafe-recovery. A new version must download the latest
archive and validate contracts before resuming. Recovery starts at the last saved
checkpoint, not the exact crash instant. One T4 is used even if two are allocated.

Public outputs are explicit adapter/vector/dataset/log folders, a measured article
and a static HF Space. No credentials or registered email are published. Public
collection links are saved to runs/submission-v1/public-links.json. The organizer
form still needs its final fields and an observed submission confirmation.
