# Initial submission checklist

Source of requirements: https://hackathon.lexsi.ai/ (checked November 14, 2026).
Check organizer announcements for changes. Initial deadline: **November 15, 2026,
11:59 PM AoE = November 16, 2026, 5:29 PM IST**. Submit early. November 22 is
the shortlisted-finalist extension, not the initial deadline. Registration is
listed as already closed; confirm the team was registered.

## Minimum credible project package

The initial submission is a working public project with real preliminary evidence,
not only this scaffold or a proposal. Do not wait for the finalist stage to run
the first comparison. If incomplete, disclose exactly what is pending; organizers
may reject incomplete submissions.

- [ ] Registered team email and all three correct HF usernames.
- [ ] Track 2b and problem numbers explicitly stated; claim only completed methods.
- [ ] CuratorKIT, SafeTune and AuditKIT actually used together; AlignTune control.
- [ ] Allowed dense Cohere model, exact revision, model/license details.
- [ ] Baseline, finance SFT, and at least one intervention evaluated on the same held-out set.
- [ ] More than one measured before/after metric. Recommended: finance accuracy,
      direct harmful compliance, ASR, over-refusal and invalid rate.
- [ ] Original benchmark contribution and domain policy definitions described.
- [ ] Curation before/after counts, source licenses and train/test leakage checks.
- [ ] Fixed generation settings and attack budgets; reviewed safety labels.
- [ ] Headline table, trade-off plot, representative outputs and limitations.
- [ ] Public GitHub repo with tested run instructions and dependency details.
- [ ] Public, ungated HF model collection containing model/adapter artifacts.
- [ ] Public HF dataset collection containing datasets, splits, logs, annotations,
      benchmark outputs, configurations and run/environment metadata.
- [ ] HF article; GitHub README is accepted if HF articles are unavailable.
- [ ] Abstract below 500 words and a measured headline before/after number.
- [ ] Discussion of real use case, method choice, dataset/model rationale and why
      the resulting model is meaningfully better (or the limitations of the finding).
- [ ] Compute used and extension plan for additional credits; label forecasts as forecasts.
- [ ] Teammates tagged, contributions recorded, outside-hackathon work declared.
- [ ] Links checked while signed out; no tokens/private data in uploaded artifacts.
- [ ] If shipping a quantized model, evaluate that exact deployed format too.
- [ ] Submission form completed using the registered email; save confirmation.

## Useful, optional additions

- [ ] Space/demo showing safe comparisons or fixed examples.
- [ ] Independent closed-model baseline using the same workload.
- [ ] Measured per-1,000-reply ROI, GPU hours, provider price and measured throughput.
- [ ] Multiple seeds, additional repair method, paired hardening comparison.
- [ ] Video/slides; especially useful for the finalist stage.

Do not sacrifice measured results and reproducibility for an elaborate UI.
No separate finance or multilingual bonus prize was advertised on the checked page.

## HF artifact organization

Suggested names under the team's real HF namespace:

| Repository | Type | Contents |
|---|---|---|
| finsafe-finance-adapter | model | Finance LoRA, base revision, training card/config |
| finsafe-repaired | model | Selected repaired checkpoint, or steering artifacts/code and clear usage contract |
| finsafe-benchmark | dataset | Versioned calibration/validation/test, labels, policy/rubric |
| finsafe-training | dataset | Licensed raw/curated data as permitted, manifests, splits |
| finsafe-runs | dataset | Generations, annotations, metrics, plots, environments and cost logs |

Do not mislabel a steering wrapper as independently repaired weights. Publish the
required adapter/checkpoint, calibration data and exact runtime instructions.
The original gated Cohere model is a dependency: publish your permitted artifacts
publicly with correct lineage and license; do not relicense upstream weights.

Create one model collection and one dataset collection, add the relevant repos,
and use their URLs in the form. A collection does not replace its individual repos.

## Suggested first-submission scope

One model, one finance task, one language, plain SFT, one repair, one fixed attack
protocol, over-refusal checks, and real before/after numbers. Explain sample-size,
judge and single-seed limitations. A second repair and multilingual extension can
be finalist work if shortlisted.

Use templates/ARTICLE.md as a starting outline. It contains placeholders and is
not a finished article. Replace every placeholder with measured evidence or an
explicitly disclosed omission before publication. Do not submit fabricated values.
