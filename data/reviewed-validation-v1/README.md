# Reviewed validation labels

All 144 responses were read by the Codex assistant in a shuffled, condition-hidden
packet (seed 71831) against frozen rubric-v1, then restored with response/prompt
hash checks. This is AI review, not independent human review. The failed original
automated judge (50% reference agreement) is rejected. Labels retain exact output
hashes and case notes. No model response text or confidential credentials are
stored here. The matching raw outputs and complete review provenance belong in
the public HF runs artifact. Run scripts/finalize_review.py against the original
experiment outputs to validate hashes and regenerate AuditKIT safety reports.
