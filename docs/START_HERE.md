# Build guide

## What is already set up

- Public repository: https://github.com/dixitkeshav/finsafe
- AryanGupta1112 and ShauryaaSharma were invited with write access. They must accept.
- Local Python 3.12 environment, locked core dependencies, dataset checks,
  CuratorKIT integration, AuditKIT replay and scoring tests.
- Starter GPU scripts and initial submission templates.

Not yet done: HF authentication/artifact repos, approved model access, provisioned
GPU, real training/evaluation, measured results, HF collections, article, or form
submission. This repository alone is not a completed hackathon submission.

## 1. Resolve accounts first

1. Confirm the team was registered using the correct team-lead email and HF usernames.
2. At least one teammate joins the organizer Discord and checks announcements.
3. Each teammate accepts the GitHub invitation and creates/signs into Hugging Face.
4. Accept access conditions on both Tiny Aya Global and Tiny Aya Base model pages.
5. Choose one shared HF namespace (an organization is convenient) and grant teammates
   appropriate access there. GitHub collaboration does not grant HF access.
6. Have the team lead retain the registration email and confirm GPU credit redemption.

Never send tokens in chat or commit them. Use `hf auth login` in each runtime.
`.env.example` is a reference; the scripts do not automatically load dotenv files.

## 2. Local environment

Follow the README quick start. On the already prepared Mac, from this repository:

```bash
source .venv/bin/activate
finsafe doctor
python -m unittest discover -s tests -v
```

This Mac is for data preparation, review and reports. GPU scripts are intended
for Linux NVIDIA runtimes. No GPU resource has been purchased or provisioned.

## 3. Set up a GPU runtime

Use an existing credited notebook/runtime first. Planning target: one 24 GB NVIDIA
GPU for unquantized LoRA SFT at sequence length 1024, batch size 1, accumulation 8.
This is a starting estimate, not measured memory usage. Test 5 steps before
reserving a longer run. ReSta keeps three full models on CPU, so plan for roughly
32 GB or more host RAM plus working memory and disk; measure actual consumption.
If your runtime is smaller, try steering first and measure memory before changing
to QLoRA. Quantization changes the experiment and requires deployment re-evaluation.

Ensure the runtime has Python 3.12 and a CUDA-compatible PyTorch installation.
Select PyTorch's install command for the runtime rather than blindly replacing
the notebook's existing CUDA packages.

```bash
git clone https://github.com/dixitkeshav/finsafe.git
cd finsafe
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
# Install/verify the runtime-compatible PyTorch wheel, then:
python -m pip install -r requirements-gpu.txt
export TRL_ONLY_MODE=1
export DISABLE_UNSLOTH_FOR_TRL=1
export TOKENIZERS_PARALLELISM=false
hf auth login
finsafe doctor --gpu
mkdir -p runs/environment
python -m pip freeze > runs/environment/gpu-freeze.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" > runs/environment/gpu.txt
```

`requirements-gpu.lock` is a complete dependency resolution for Linux x86_64 /
Python 3.12. It has been solver-checked, not installed or GPU-validated. It pins
PyTorch 2.13.0 and its CUDA dependencies, so use it only on a compatible host;
otherwise resolve `requirements-gpu.txt` against the runtime's compatible torch
build and save the resulting freeze. A successful resolver is not a training test.

If the runtime provides a supported torch build globally, `venv --system-site-packages`
can reuse it, but record that choice. SafeTune documents a Colab workaround:
uninstall `torchao` before importing SafeTune if you encounter that compatibility
issue; record the change, since AlignTune declares it as a dependency.

Do not install `[all]`, vLLM, flash attention or Unsloth for the first run.
Add performance tooling only after the core experiment works.

Capture immutable model references:

```bash
python - <<'PY'
import json
from huggingface_hub import HfApi
models = ['CohereLabs/tiny-aya-global', 'CohereLabs/tiny-aya-base']
refs = {model: HfApi().model_info(model).sha for model in models}
open('runs/environment/model-revisions.json', 'w').write(json.dumps(refs, indent=2))
print(json.dumps(refs, indent=2))
PY
```

Use the printed SHA wherever the commands below use `GLOBAL_SHA` or `BASE_SHA`.
These placeholders are not real revisions. Test downloading the tokenizer before
starting a long job; denied model access must be fixed in the HF account.

## 4. Build real data before training

Start with one task: short finance questions with checkable calculations or
answers grounded in a supplied passage. Avoid a vague general investment adviser.
Useful starting targets (project choices, not organizer requirements):

- 1,000–3,000 vetted benign SFT examples.
- Separate calibration and validation sets with enough examples to select alpha.
- At least 200 final finance questions, 100 direct harmful requests, 100 attack
  variants, and 100 legitimate sensitive questions, plus a general retention set.
- Balance financial wrongdoing categories; use at least two attack families.

If time only permits smaller sets, report the actual counts and uncertainty.
Use licensed public data or team-authored, programmatically checked examples;
record source, license and generation provenance per source. Do not copy an
unverified finance dataset into the project merely because it has many rows.
Group by source filing/document/question family; paraphrases and translations
stay in the same split. Perform exact and semantic/MinHash cross-split checks.

Keep a master JSONL using the example schema, then:

```bash
finsafe check --input data/raw/finance_benchmark.jsonl
finsafe prepare --input data/raw/finance_benchmark.jsonl --out data/processed/real
python scripts/curate.py --input data/processed/real/train_alpaca.jsonl --out data/processed/real/curated
```

Inspect curation counts and examples removed. Never curate test prompts into
training examples. The included seed corpus is only for checking file formats.

## 5. Run the compatibility check and baseline

First use the example validation file as a model-load/generation smoke test;
never report that tiny sample as benchmark evidence. Then use the real validation
file for iteration. Save baseline predictions before changing any weights.

```bash
python scripts/generate.py --model CohereLabs/tiny-aya-global --revision GLOBAL_SHA \
  --data data/processed/real/validation.jsonl --condition aligned-baseline \
  --out runs/baseline-validation/predictions.jsonl
```

No extra system prompt is added: all conditions use the tokenizer's native chat
template. Record this choice. The 1024-token training length avoids the documented
Tiny Aya preamble truncation issue; still inspect actual supervised-token counts.

## 6. Train plain finance SFT

```bash
python scripts/train_sft.py --revision GLOBAL_SHA --data data/processed/real/curated \
  --max-steps 5 --out checkpoints/smoke-sft
```

Check finite loss, nonempty adapter/checkpoint files, the applied LoRA modules and
valid generated answers. Do not use this smoke checkpoint as the final SFT run.
Then start a fresh run with a fixed training budget:

```bash
python scripts/train_sft.py --revision GLOBAL_SHA --data data/processed/real/curated \
  --max-steps 100 --out checkpoints/finance-sft
```

100 steps is an initial budget, not an optimized recipe. Record row count, effective
batch size and actual exposure. Choose settings on validation, not final test.
Inspect AlignTune output and find the saved directory containing `adapter_config.json`.
Replace `ADAPTER_DIR` with that actual directory:

```bash
python scripts/generate.py --model CohereLabs/tiny-aya-global --revision GLOBAL_SHA \
  --adapter ADAPTER_DIR --data data/processed/real/validation.jsonl \
  --condition finance-sft --out runs/sft-validation/predictions.jsonl
```

## 7. Run one repair

Fastest experimental branch: inference-time steering, using only the calibration
file to estimate directions and validation to select strength:

```bash
python scripts/generate.py --model CohereLabs/tiny-aya-global --revision GLOBAL_SHA \
  --adapter ADAPTER_DIR --data data/processed/real/validation.jsonl \
  --steer-calibration data/processed/real/calibration.jsonl --alpha 0.1 \
  --condition steering-alpha-0.1 --out runs/steering-validation/predictions.jsonl
```

Try a small validation sweep and inspect normal-answer quality; alpha is model
dependent. Live steering does not create a standalone repaired checkpoint.
Publish the adapter, calibration dataset, alpha and exact wrapper code.

Weight-space branch: merge the adapter, then run ReSta on CPU:

```bash
python scripts/merge_adapter.py --revision GLOBAL_SHA --adapter ADAPTER_DIR --out checkpoints/finance-merged
python scripts/repair_resta.py --drifted checkpoints/finance-merged \
  --base-revision BASE_SHA --aligned-revision GLOBAL_SHA --alpha 0.25 \
  --out checkpoints/resta-025
python scripts/generate.py --model checkpoints/resta-025 \
  --data data/processed/real/validation.jsonl --condition resta-025 \
  --out runs/resta-validation/predictions.jsonl
```

ReSta explicitly disables DARE for a simpler comparison. The original global
checkpoint must match the parent of the finance adapter. Compatible shapes alone
do not prove correct lineage. Sweep small alpha values on validation; upstream
warns that 1.0 breaks Tiny Aya. Do not claim repair if drift was not measured.

## 8. Review outputs and compute results

For each condition, create and complete a separate annotation file:

```bash
finsafe annotation-template --data data/processed/real/validation.jsonl \
  --predictions runs/baseline-validation/predictions.jsonl \
  --out runs/baseline-validation/annotations.jsonl
```

Review each non-finance generation using PROTOCOL.md. Replace UNREVIEWED with a
valid label and record reviewer identity. Blind condition names during review
where possible. Have a second person audit a sample and all ambiguous outputs.
Keep the generated response hash unchanged; it prevents labels from one condition
being accidentally applied to another model's responses.

```bash
finsafe summarize --data data/processed/real/validation.jsonl \
  --predictions runs/baseline-validation/predictions.jsonl \
  --annotations runs/baseline-validation/annotations.jsonl \
  --out runs/baseline-validation/report.json
finsafe audit-replay --data data/processed/real/validation.jsonl \
  --predictions runs/baseline-validation/predictions.jsonl \
  --out runs/baseline-validation/auditkit.json
```

Repeat for SFT and the selected repair. Freeze hyperparameters and scoring before
running the final test set once. Use distinct `runs/*-test/` directories, the same
test file and the same generation settings for all final conditions.

```bash
python scripts/plot.py runs/baseline-test/report.json runs/sft-test/report.json \
  runs/repair-test/report.json --out artifacts/tradeoff
```

Measure actual GPU hours and pricing/credit consumption. Inference latency in
the scripts is single-request latency, not production batched throughput.
Do not extrapolate production ROI without a representative measured workload.

## 9. Publish and assemble the initial submission

Follow SUBMISSION.md. Create dedicated upload folders and copy only reviewed
artifacts. Complete a model/data card in each folder, then for example:

```bash
python scripts/publish_hf.py --folder artifacts/benchmark-release \
  --repo Finsafe-team/finsafe-benchmark --type dataset --public
python scripts/publish_hf.py --folder artifacts/adapter-release \
  --repo Finsafe-team/finsafe-finance-adapter --type model --public
```

The team namespace is `Finsafe-team`; the team lead account is `KeshavDixit11`. Create HF model
and dataset collections in the HF interface and add all related public repos.
Test links while signed out. A live Space is optional; a public reproducible repo
and actual results are higher priority for the first submission.


## Selected runtime: Kaggle

Use [the Kaggle setup notebook](../notebooks/00_kaggle_setup.ipynb). Create a Kaggle
notebook, import that file, enable Internet and select an available NVIDIA GPU.
Add your Hugging Face access token as a Kaggle secret named `HF_TOKEN`, and enable
that secret for this notebook. Never paste the token into code or output. The token
must have model-read permission and, when publishing, write permission to
`Finsafe-team`. Accept the Tiny Aya Global and Base conditions as `KeshavDixit11`.

The setup notebook checks the GPU, Python and PyTorch before installation. Kaggle
images and available accelerators change; do not assume two GPUs combine into one
larger memory pool. Save the doctor output before choosing training settings.
Installation/model downloads require Internet access. Kaggle sessions are temporary:
export checkpoints and logs to HF or save notebook output before the session ends.
The notebook does not train or publish automatically.
