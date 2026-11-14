# Logs, checkpoints and recovery

All new notebook code cells must log START, DONE or FAILED with timestamps and
elapsed time. Stage 02 uses UTC in persistent log files; its session timer resets
when the helper is re-created, while append-only log timestamps retain history.
Saved per-stage JSON files record the latest attempt's duration/status.

## Current next notebook

`notebooks/02_kaggle_t4_baseline.ipynb` matches the reported T4 / CUDA 13.0 runtime.
It installs in `.venv-gpu` with Python 3.12, preserves the host kernel, uses GPU 0,
pins the PyTorch 2.11 / torchvision 0.26 family, then filters token lengths,
curates real data and runs an eight-example 4-bit/FP16 baseline smoke test.
The dependency set solver-resolves; GPU execution must be verified on Kaggle.
It does not train or publish. Do not use a smoke score as final benchmark evidence.

Each notebook subprocess streams output into a timestamped log. The first bootstrap
cell prints its own start and saves timing after the project logger is loaded.
If bootstrap itself fails before that import, the visible notebook output is the
available log. Later cells persist failure markers automatically.

## Generation recovery

Use the same generate.py command with `--resume`. A `.run.json` contract records
data hashes, remote model revision/local weight hashes, adapter hashes, generation
settings, code hash and relevant package versions. Every response is written with
an atomic file replacement. Resume accepts only the saved ordered prefix and
rejects changed configuration/artifacts. Completed runs skip loading the model.
Partially generated current responses may be repeated; earlier saved responses
are reused. Artifact hashing can take time for large local checkpoints.

## Training recovery

The training runner uses AlignTune's model/data setup and training-argument builder,
then explicitly calls the underlying TRL trainer. Upstream wrapper.train() did not
forward resume_from_checkpoint; invoking that wrapper alone would not restore state.

```bash
# Five-step smoke run, with a checkpoint at step 5:
CUDA_VISIBLE_DEVICES=0 python scripts/train_sft.py --revision GLOBAL_SHA \
  --data data/processed/tatqa-tokenized/curated --load-in-4bit \
  --max-steps 5 --save-steps 5 --out checkpoints/smoke-sft

# After an interruption, rerun identical arguments and add:
# --resume
```

Use a fresh output directory for the 100-step research run, normally saving every
25 steps and retaining two checkpoints. Resume restores the latest complete weights,
optimizer, scheduler, RNG and trainer state. It resumes at a checkpoint boundary;
steps since that checkpoint may be repeated. If no complete checkpoint exists, it
logs a restart from step zero. Identical data/settings/version checks are required.
A completed run exits rather than retraining. GPU recovery is not locally tested.

Checkpoint files do not promise bit-for-bit determinism across different GPUs,
drivers or library versions. Record environment changes and disclose reruns.

## Protect against session loss

`/kaggle/working` alone is not durable backup. Stage 02 creates
`/kaggle/working/finsafe-stage02-recovery.zip` with processed data and run logs,
excluding tokens, virtual environments and model cache. Download it or retain
Kaggle output before ending the session. A ZIP created once does not include
future work: rerun the backup cell after important new outputs.

Training optionally accepts `--backup-repo Finsafe-team/finsafe-recovery` to upload
each complete checkpoint to a **private HF dataset repo** using a write-enabled
token. It logs durable-backup success or failure; a failed upload leaves only the
local checkpoint. This option requires explicit configuration and is off by default.
Private recovery storage is separate from the final public submission artifacts.

Restore a private training backup with Hugging Face snapshot_download, repo_type
`dataset`, allow_patterns for the run folder (e.g. `finance-sft/**`), and local_dir
`checkpoints`. Also restore the original data and package environment, then pass
the same training arguments with `--resume`. Never treat adapter-only weights as
a full training-state checkpoint.

## Runtime notes

The user's reported host uses Python 3.13.16, torch 2.11.0+cu130, driver 580.178.04
and two Tesla T4s. The project uses a separate Python 3.12 environment. T4 training
uses FP16 compute rather than native BF16. 4-bit loading is an explicit experiment
setting: use matching precision for compared deployed conditions and report it.
Unsloth is not required for this recipe.

Reference: https://pytorch.org/get-started/previous-versions/ and
https://github.com/pytorch/ao/issues/2919 for the matched torch/torchao families.
