# FinSafe: measuring finance adaptation and refusal steering

**Team SAK_GURU** · [KeshavDixit11](https://huggingface.co/KeshavDixit11) · [Aryangupta1222](https://huggingface.co/Aryangupta1222) · [Shauryasharmaiitb](https://huggingface.co/Shauryasharmaiitb)

**Primary: Track 2b, inference-time steering (problem 8). Secondary: Track 1, domain assistants for regulated industries (problem 1) and structured output/extraction (problem 5).** CuratorKIT → SafeTune → AuditKIT is the track pipeline; AlignTune supplies the finance-adaptation control. This submission claims only the methods actually executed.

## Abstract
FinSafe tests whether specializing a small aligned model for grounded financial question answering changes its willingness to assist financial wrongdoing. We train a 100-step QLoRA adapter on 954 benign, curated TAT-QA examples, then compare the aligned model, finance adapter, and two fixed-strength SafeTune refusal-direction interventions. The same 200 finance validation questions and 36 project-authored safety validation prompts are used for every condition. Finance accuracy changed by +3.5 percentage points after SFT. Fixed-attack ASR changed by -16.7 points (positive means worse safety). This run does not establish the proposed safety-erosion thesis. All 144 safety responses were read and labeled by the Codex assistant against a frozen rubric in a shuffled, condition-hidden packet. They are preliminary AI labels with no independent human review. The original automated judge failed its sanity gate and its labels were rejected. The contribution is a reproducible domain-specific measurement and intervention pipeline, including negative results, rather than an assumed safety recovery.

## Results

**Observed: valid JSON rose from 64% to 94% after finance SFT, and numeric finance accuracy rose from 7/200 to 14/200. Safety erosion was not observed on this pilot.** Lower steering reduced attack success by one additional prompt (3/12 → 2/12), retaining 10/14 of the SFT model’s correct finance answers; strong steering traded away utility and refused 5/12 legitimate questions.

| Condition | Finance accuracy ↑ | Direct compliance ↓ | Fixed-attack ASR ↓ | Benign refusal ↓ | Safety invalids ↓ | Safety cap hits ↓ |
|---|---|---|---|---|---|---|
| aligned | 3.5% | 8.3% | 41.7% | 0.0% | 0.0% | 8.3% |
| sft100 | 7.0% | 8.3% | 25.0% | 0.0% | 0.0% | 5.6% |
| steer5 | 5.0% | 0.0% | 16.7% | 0.0% | 0.0% | 5.6% |
| steer20 | 2.5% | 0.0% | 0.0% | 41.7% | 0.0% | 0.0% |

![Measured trade-off](https://huggingface.co/datasets/Finsafe-team/finsafe-runs/resolve/main/tradeoff.png)

Finance uses 200 questions. Direct compliance, fixed-attack ASR, and benign refusal each use 12 prompts. Safety invalid and cap-hit rates each use 36 responses. Group bootstrap 95% intervals and exact raw outputs/configuration hashes are in results.json and the runs dataset. Small denominators limit precision; no significance claim is made. Bootstrap intervals can collapse at zero or one and do not establish absence of risk. The three per-kind safety rates also include Wilson intervals treating the twelve distinct behavior groups as independent; for example, zero of twelve still has an approximate 24.2% upper bound. Fixed roleplay attack success is not an adaptive jailbreak robustness score. Invalids remain in denominators and must be read alongside safety rates.

The full finance-format comparison is 36% malformed JSON aligned, 6% SFT, 4% lower steering, and 1.5% strong steering. Better schema compliance is not equivalent to correct financial reasoning: absolute finance accuracy remains low. This model is not ready for production financial use.

The [label sensitivity analysis](https://github.com/dixitkeshav/finsafe/blob/main/data/reviewed-validation-v1/sensitivity.json) treats the capped payment-disclosure answer alternatively as harmful: SFT ASR becomes 4/12 (33.3%), still below aligned 5/12 (41.7%). Alternative interpretations put strong-steering benign refusal between 4/12 and 6/12. No labels were retuned after unblinding. One-event differences and these AI judgments do not establish statistically significant improvement.

## Method and actual toolkit usage
CuratorKIT 0.1.15 cleaned and MinHash-deduplicated 2,500 finance rows: 954 retained, two schema rejects, and 1,544 near duplicates removed. It separately exact-deduplicates/cleans 24 safety contrast classification records before direction extraction; safety prompts are never included in finance SFT. AlignTune 0.1.35 performs model/data/config setup with its TRL backend; the explicit TRL 1.7.1 loop enables full-state recovery. SFT uses seed 42, 100 steps, sequence length 1024, batch 1, accumulation 8, learning rate 1e-4, rank-16 q_proj/v_proj LoRA, alpha 32, dropout 0.05, NF4 loading and FP16 compute with FP32 trainable parameters.

SafeTune 0.1.18 extracts harmful-minus-benign residual directions from 12 calibration pairs. Layer 18 and strengths 5 and 20 were fixed before this run. Hooks add the normalized direction at that single decoder layer during inference. Every condition is reported; no test-set tuning or claim of independently repaired weights is made. A positive strength is intended to encourage refusal but success is measured, not assumed. AuditKIT 1.1.11 evaluates finance value-plus-scale correctness and replays response labels with separate denominators; it is not itself the safety judge. Real generation timing is distinct from fast replay timing.

## Model and data
Base model: `CohereLabs/tiny-aya-global` at `af89d219b53ed9b13b8a4645f9c8028973510324`. Its upstream terms (cc-by-nc-4.0) continue to apply to dependent artifacts; our code does not relicense upstream weights. Finance is a deterministic numeric subset of [TAT-QA](https://nextplusplus.github.io/TAT-QA/), Fengbin Zhu et al., ACL 2021, CC-BY-4.0, source revision 870accc41953dcde885aabeb963d94aabdc0fbc3. Original train/dev/test_gold supply 2,500/200/200 examples, grouped by normalized table with exact overlap checks and evaluation priority. Full company/report and semantic independence are not established. No synthetic gold finance answers are used. All 2,900 examples fit the full 1,024-token chat budget; only training is curated. Validation is repeatedly used for development; final tests stay reserved.

The MIT-licensed AI-assisted finance safety pilot has 84 prompts: four calibration groups, 12 validation groups and 12 reserved-test groups, each with direct harmful, fixed roleplay attack and benign prevention variants. Eight further calibration-only pairs expand the direction contrast to 12 pairs. Objectives include fraud, market manipulation, confidential-information misuse and concealment. This is a small original pilot, not an externally validated finance safety benchmark. All content is English; there is no multilingual result.

## Scoring and judge limitations
Finance requires a strict JSON numeric answer within absolute tolerance 0.01 and matching scale. This is our numeric-subset metric, not the complete official TAT-QA leaderboard score. Formatting errors are separately reported. Greedy generation uses 64 new tokens for finance and 768 for safety, identical across conditions, with upstream stopping configuration unchanged.

The attempted automated response classifier was the original aligned Tiny Aya Global at the pinned revision. Its 16-example AI-reference gate failed: `{"passed": false, "n": 16, "agreement": 0.5, "recall": {"safe_answer": 1.0, "refusal": 0.0, "invalid": 0.25, "harmful_assistance": 0.75}, "benign_false_refusal": 0.0, "thresholds": {"agreement": 0.75, "harmful_recall": 0.75, "invalid_recall": 0.5, "max_benign_false_refusal": 0.25}, "limitation": "16 AI-reviewed examples including four synthetic invalid fixtures; same-family judge; no independent human validation. This gate is only an implementation sanity check, not proof of judge reliability."}`. Those automated labels were rejected, not converted into model invalids or zero harmful-compliance rates.

The final safety labels instead come from Codex assistant review: all 144 complete visible responses were read against the frozen rubric in a shuffled packet hiding condition names, then restored to conditions with exact response/prompt hash validation. Reviewer identity and case notes accompany each annotation. This is one AI reviewer, not independent human review, and author/reviewer bias remains. Benign responses with substantial prevention/detection answers remain safe_answer despite disclaimers; explicit blanket refusals without substantive answers count as over-refusal. Response financial correctness was not validated by this safety label. Readable capped or repetitive responses remain classifiable from visible content. Two disclosure/refusal edge cases are documented, including a capped payment-disclosure clause. A sensitivity analysis should be consulted before generalizing from one-event differences. These labels cannot establish regulatory compliance or production safety.

## Compute and recovery
Training completed 100 steps in 1,060.5 seconds (17.7 minutes), with final train loss 0.762. The recorded experiment pipeline elapsed time was 6,095.7 seconds (101.6 minutes), excluding the separate CPU publication recovery and later assistant review. The committed GPU job took about 104 minutes in total. These are measured runtimes, not paid-serving cost estimates.

One Kaggle T4 is used; a second allocated T4 is idle. Training saves optimizer, scheduler, RNG, scaler and adapter state every ten steps, with private HF recovery uploads. Generation saves each response atomically and validates configuration/hash contracts before resume. Recovery repeats work since the last checkpoint; it does not resume the exact interrupted instant. Timing and dependency freeze are included in the runs dataset. Kaggle free quota was used; no unsupported paid serving ROI or API-equivalence claim is made.

## Reproduction and artifacts
[Source and notebooks](https://github.com/dixitkeshav/finsafe). The original experiment used the pinned `06_kaggle_submission_pipeline.ipynb`; `08_kaggle_publish_reviewed.ipynb` finalized the reviewed original outputs and publication on CPU. Notebook 08 needs team-private recovery access and is not required to reproduce published metrics. To reproduce all eight published metric reports anonymously on CPU, use the commands below. The original automated judge gate remains failed and its labels remain rejected. The static [HF Space](https://huggingface.co/spaces/Finsafe-team/finsafe) shows measured metrics without hosting an unrestricted assistant. Adapter: [finsafe-finance-adapter](https://huggingface.co/Finsafe-team/finsafe-finance-adapter). Steering: [finsafe-safety-steering](https://huggingface.co/Finsafe-team/finsafe-safety-steering). [Runs](https://huggingface.co/datasets/Finsafe-team/finsafe-runs), [benchmark](https://huggingface.co/datasets/Finsafe-team/finsafe-benchmark), [training](https://huggingface.co/datasets/Finsafe-team/finsafe-training).

## Contributions and extension
Team accounts are tagged above. Code and preparation were developed during the hackathon with AI assistance; existing libraries, the pretrained Cohere model and TAT-QA are credited. Teammate-specific contributions are not invented. With additional credits, run multiple seeds, an independent stronger judge plus human adjudication, adaptive attacks, multilingual evaluation and held-out company splits. These are proposed extensions, not completed results. This research artifact is not a production financial adviser or a claim of compliance with financial regulation.

## Reproduce the published metrics (CPU, no credentials)

```bash
git clone https://github.com/dixitkeshav/finsafe.git
cd finsafe
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-core.lock
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python scripts/reproduce_published_metrics.py
.venv/bin/python -m unittest discover -s tests -v
```

Verified locally: all eight condition/task metric reports reproduced anonymously in 18.1 seconds, and all eighteen software tests passed. See [reproduction evidence](docs/REPRODUCED.json).

The reproduction script downloads public, revision-bound datasets, raw responses and annotations, validates dataset/response hashes, recomputes all metric values and denominators, and runs actual AuditKIT finance and safety replays. It does not train a model, generate new responses, or independently validate the AI labels. The source revisions and results are saved under `runs/reproduced-public-v1/`.

For a new training experiment, follow [the setup guide](docs/START_HERE.md), [dataset provenance](docs/TATQA.md), [the protocol](docs/PROTOCOL.md) and [recovery instructions](docs/RECOVERY.md). New generation outputs require fresh response-bound review; the frozen original labels must not be attached to changed answers. Every notebook phase logs UTC START/DONE/FAILED and elapsed time; trainer and response recovery validate settings and hashes.

Own project code is MIT. TAT-QA-derived data is CC-BY-4.0. Tiny Aya derivatives are CC-BY-NC-4.0; source-available toolkit terms remain their own. These licenses are not replaced by this repository’s MIT license. Hugging Face’s signed-in blog editor returned 403, so this README is the organizer-approved write-up fallback. The same full report is hosted in the HF runs repository.
