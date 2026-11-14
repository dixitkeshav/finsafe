# TAT-QA finance data

We use the original TAT-QA release, not the contaminated SwayAlgo train/eval pairing.
Source: https://github.com/NExTplusplus/TAT-QA
Dataset license and attribution: https://nextplusplus.github.io/TAT-QA/ (CC-BY-4.0;
Fengbin Zhu et al., ACL 2021). Preserve attribution in published dataset cards.

Pinned upstream revision: `870accc41953dcde885aabeb963d94aabdc0fbc3`.

## Reproduce preparation

```bash
python scripts/prepare_tatqa.py --out data/processed/tatqa-v1
```

Verified locally: 2,500 train, 200 validation and 200 test numeric examples.
Underlying numeric candidate counts were 5,847 / 750 / 739. Four duplicate
IDs/prompts and two examples from repeated table groups were filtered in the
preparation pass. These are data statistics, not model performance results.

Train uses official train; validation uses official dev; final test uses official
test_gold. Sampling is deterministic (seed 42) and table groups are not split
to satisfy row caps. Identical normalized table contents get the same group ID,
even if upstream UUIDs differ. Test and validation take priority over train when
a table is repeated. Company/report-level and semantic overlap still need review;
this is not a proof of complete source independence.

Only arithmetic/count questions with finite numeric gold answers are included.
The prompt contains the source table, context and question, but never the gold
answer, supporting-fact annotations or gold derivation. No synthetic answers are
generated. This subset measures grounded numeric finance QA, not all TAT-QA tasks.

Targets are JSON, for example:

```json
{"answer": -12.6, "scale": "million"}
```

Scoring requires a numeric value within absolute tolerance 0.01 AND the correct
scale. It does not accept arbitrary substrings, booleans or nonfinite values.
This is our declared numeric-subset metric, not the complete official TAT-QA
leaderboard scorer. Exact match/F1 diagnostics alone are insufficient.

## Filter actual token lengths before training

After the compatible GPU environment is installed and model access works:

```bash
python scripts/filter_token_lengths.py --input data/processed/tatqa-v1/master.jsonl \
  --out data/processed/tatqa-tokenized --revision GLOBAL_SHA --max-tokens 1024
python scripts/curate.py --input data/processed/tatqa-tokenized/train_alpaca.jsonl \
  --out data/processed/tatqa-tokenized/curated
```

Use the actual immutable Tiny Aya Global SHA for GLOBAL_SHA. The filter counts
the entire chat (including the model preamble and target); it rejects overlong
examples instead of truncating context. Report the resulting split sizes, which
will be smaller than the initial 2,500/200/200. Fix this rule before final testing.

Audit curation counts and accepted training samples. The upstream manifest and
token-filter manifest remain separate. Do not put raw source JSONs or large data
exports in Git; publish reviewed data and cards under Finsafe-team on HF.

## Finance baseline

Run scripts/generate.py on validation before training. Then use finsafe summarize
without --annotations for finance-only results and finsafe audit-replay for the
AuditKIT run. Numeric-scale accuracy is evaluated through the custom AuditKIT
scorer too. A baseline needs actual generations: no results are supplied.

This finance set does not supply any safety metrics. Add an independently reviewed
direct-harmful, attack and benign-sensitive benchmark before making safety-drift
or repair claims. Do not claim a completed experiment from finance accuracy alone.
