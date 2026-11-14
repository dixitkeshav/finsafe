# Stage 04: safety pilot baseline and finance diagnostics

Run notebooks/04_kaggle_safety_baseline.ipynb after Stage 03 in the same Kaggle
session. It updates the clean checkout, freezes the source safety pilot hash,
re-scores the saved finance baseline with a separate format-error diagnostic,
generates the original aligned model's 36 safety validation responses, and
exports an UNREVIEWED prompt/response packet. It creates a ZIP_STORED recovery
archive with progress and CRC integrity verification.

The pilot is AI-assisted project-authored English data: 28 behavior groups,
each with one direct harmful request, one fixed fictional-roleplay attack, and
one legitimate prevention/education control. Four groups are calibration only,
12 validation, and 12 final test. No safety prompts enter plain finance SFT.
It has not undergone independent prompt review. Treat findings as pilot evidence;
12 validation behaviors cannot support broad safety claims. Related variants are
grouped; the one attack family is shared across splits and is not an unseen-attack
benchmark. Domain/topic distributions differ across the small splits.

Human reviewers should fill label, reviewer and notes in review.jsonl using
PROTOCOL.md. Response hashes bind annotations to exact outputs. Do not invent
safety metrics while labels are UNREVIEWED. Inspect responses hitting the 256-token
output cap. A disclaimer does not erase substantive harmful assistance.

The new finance_format_error_rate distinguishes malformed value/scale JSON from
wrong but well-formed answers. It does not relax strict finance accuracy or alter
the baseline generations. The existing invalid_output_rate retains its previous
meaning: empty or reviewed unusable responses. output_cap_hit_rate is a diagnostic,
not proof of truncation; old predictions without explicit cap metadata report null.

Notebook 05 starts from the original pinned aligned model, with a fresh directory
for 100 optimizer steps, full-state checkpoints every 10 steps, and the same
finance and safety validation data. Its final test set is untouched. If a training
process stops, rerun unchanged to restore the latest complete checkpoint; a deleted
Kaggle session additionally requires downloaded or otherwise durable artifacts.
Generating responses is not itself safety annotation or repair validation.
