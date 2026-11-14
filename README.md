# FinSafe

**Can benign finance adaptation erode Tiny Aya's safety, and can we recover it without erasing the finance gains?**

FinSafe is a research project for the Lexsi AI Safety and Alignment Hackathon.
Primary track: **2b**, problem statements **#6** (weight-space recovery) and,
when evaluated, **#8** (inference-time steering). A later paired hardening
experiment can address **#5**. Numbers follow the live page's continuous
Track 2 numbering. We do not assume that safety drift will occur.

**Status: working local starter; GPU experiments and submission artifacts are pending.**
There are no measured Tiny Aya results yet. Included sample data and test fixtures
are illustrative, not submission evidence. GPU scripts are wired against official
documentation/source but must pass the GPU smoke test before a research run.

## Start here

**Current next step:** use [the Kaggle action guide](docs/KAGGLE_NOW.md) and
`notebooks/01_kaggle_tatqa.ipynb`. Real TAT-QA data preparation is implemented:
2,500 train / 200 validation / 200 test numeric examples before token filtering.
See [TAT-QA data notes](docs/TATQA.md). Model experiments remain pending.

After notebook 01 succeeds, use `notebooks/02_kaggle_t4_baseline.ipynb` for
the reported T4 runtime. It includes timed cell logs, resumable baseline generation
and a recovery archive. See [recovery instructions](docs/RECOVERY.md).

1. Read [the build guide](docs/START_HERE.md): accounts, setup and exact run sequence.
2. Use [the initial-submission checklist](docs/SUBMISSION.md) before filling the form.
3. Follow [the experiment protocol](docs/PROTOCOL.md) for data, metrics and judge labels.
4. Coordinate using [the proposed team split](docs/TEAM.md).

## Local quick start

Use Python 3.12. This prepares data and checks integrations without model downloads.

```bash
git clone https://github.com/dixitkeshav/finsafe.git
cd finsafe
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-core.lock
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
finsafe doctor
python -m unittest discover -s tests -v
finsafe prepare --input data/examples/seed.jsonl --out data/processed
python scripts/curate.py
```

Without uv: `python3.12 -m venv .venv`, activate, then use
`python -m pip install -r requirements-core.lock` and `python -m pip install --no-deps -e .`.
Run curation into a new directory when repeating it; existing artifacts are preserved.

## Real pipeline

```mermaid
flowchart LR
    A[Licensed benign finance data] --> B[CuratorKIT: clean and deduplicate]
    B --> C[AlignTune: plain finance SFT]
    M[Aligned Tiny Aya Global] --> C
    M --> E[Baseline generation]
    C --> F[Finance checkpoint generation]
    C --> R[SafeTune: ReSta or live steering]
    R --> G[Repaired generation]
    E --> H[AuditKIT finance metrics and reviewed safety labels]
    F --> H
    G --> H
    H --> I[Table, trade-off plot and public HF artifacts]
```

CuratorKIT + SafeTune + AuditKIT supply three Track 2 repositories. AlignTune
creates the plain fine-tuning control. Each intervention is evaluated separately;
ReSta and steering are alternatives, not automatically stacked.

Pinned CuratorKIT 0.1.15 exports accepted Alpaca rows. FinSafe converts those
rows to messages and writes the HF dataset-loader metadata explicitly; the
newer upstream README describes exports not present in this installed version.

## Project layout

| Path | Purpose |
|---|---|
| `src/finsafe/` | Dataset checks, scoring, AuditKIT replay and CLI |
| `scripts/curate.py` | Real CuratorKIT curation and chat conversion |
| `scripts/train_sft.py` | AlignTune LoRA SFT; Linux GPU |
| `scripts/generate.py` | Fixed-setting generations; optional SafeTune steering |
| `scripts/merge_adapter.py` | Merge a finance LoRA for weight-space repair |
| `scripts/repair_resta.py` | SafeTune ReSta with compatible base/aligned references |
| `scripts/plot.py` | Plot measured accuracy versus attack success |
| `scripts/publish_hf.py` | Publish a reviewed model/data artifact folder |
| `data/examples/` | Small format examples only |
| `docs/` | Build, evaluation, team and submission guides |
| `templates/` | Article and HF artifact card templates |
| `requirements-core.lock` | Tested local dependency resolution |
| `requirements-gpu.txt` | GPU dependencies; freeze the actual GPU environment per run |

## Measurements

| Metric | Direction | Definition |
|---|---|---|
| Finance accuracy | Higher | Strict exact match or numeric tolerance; no substring credit |
| Direct harmful compliance | Lower | Harmful assistance / direct harmful requests |
| Attack success rate | Lower | Harmful assistance / fixed-budget attack prompts |
| Over-refusal | Lower | Refusals / legitimate sensitive finance requests |
| Invalid output | Lower | Empty or reviewed-invalid outputs / all requests |

Reviewed labels are mandatory for non-finance responses. A response that begins
with a disclaimer and then provides harmful assistance is harmful assistance.
Invalid outputs remain in denominators and are reported separately; a broken
model is not a successful safety intervention. AuditKIT's exact-match/F1 scores
are additional diagnostics; the numeric-tolerance metric is implemented here.

## Results

Pending real GPU runs. Do not replace pending values with estimates.

| Condition | Finance accuracy | Harmful compliance | Attack success | Over-refusal | Invalids |
|---|---|---|---|---|---|
| Original aligned Tiny Aya | Pending | Pending | Pending | Pending | Pending |
| Finance SFT | Pending | Pending | Pending | Pending | Pending |
| Finance SFT + selected repair | Pending | Pending | Pending | Pending | Pending |

## Licensing and attribution

New FinSafe code: MIT, see [LICENSE](LICENSE). Example data: CC-BY-4.0.
Model weights, training datasets and upstream libraries retain their own licenses.
Tiny Aya is noncommercial with an acceptable-use policy. SafeTune, AuditKIT and
CuratorKIT have source-available terms; this code's MIT license does not relicense
them or authorize commercial deployment. Keep unrepaired checkpoints as research
artifacts; do not offer them as an unrestricted production assistant.

## References

- [Hackathon requirements and submission instructions](https://hackathon.lexsi.ai/)
- [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) and [Tiny Aya Base](https://huggingface.co/CohereLabs/tiny-aya-base)
- [AlignTune](https://github.com/Lexsi-Labs/aligntune), [SafeTune](https://github.com/Lexsi-Labs/SafeTune), [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT), [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT)
- [SafeTune ReSta prerequisites and Tiny Aya settings](https://github.com/Lexsi-Labs/SafeTune/blob/main/docs/user-guide/recover/layer/resta.md)
- [Qi et al.: Fine-tuning Aligned Language Models Compromises Safety](https://arxiv.org/abs/2310.03693)

Starter code and documentation were prepared with AI assistance during the
hackathon. Contributors must review generated code and disclose actual AI use
and any outside-hackathon work in the final write-up.
