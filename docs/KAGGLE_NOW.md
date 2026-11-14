# Current Kaggle stage

Use the existing private notebook: https://www.kaggle.com/code/keshavdixit11/lexsilabs/edit
Do not reset the running session until its recovery files are downloaded.

## Completed

- Notebook 01: real TAT-QA preparation and HF access.
- Notebook 02: matched Python 3.12 GPU environment, token filtering, CuratorKIT,
  and 4-bit Tiny Aya model-load/generation smoke test on one T4.
- Notebook 03: original 200-question finance baseline (7/200 correct), five-step
  QLoRA smoke training (loss 2.1082), checkpoint file verification and adapter
  generation. Stage-03 recovery ZIP passed its integrity check (211.6 MiB).

## Current workflow

Notebook 04 freezes the safety pilot and collects 36 validation responses. It
also diagnoses the saved finance responses: 36% malformed JSON, none at the known
64-token cap. The safety review packet contains prompt, generated output, exact
response hash, cap flag, UNREVIEWED label, reviewer and notes. Fill labels using
STAGE04.md and PROTOCOL.md before reporting harmful-compliance/ASR/over-refusal.
The pilot is small and AI-assisted; independent prompt review remains necessary.

Notebook 05 is ready for a fresh 100-step finance run and matching validation
responses. It has not yet executed. It uses a new output directory, full-state
checkpoints every 10 steps and resumable generation. It does not evaluate final
test data or produce safety rates before annotation.

Keep Internet, GPU and HF_TOKEN enabled. Tokens belong in Kaggle Secrets only.
Every notebook logs cell timing and failures. Download the latest verified
recovery ZIP before stopping the session; Quick Save alone is not proof that
working-directory files are durable. See RECOVERY.md for restore behavior.
