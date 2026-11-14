# Stage 03: five-step training smoke test

User-run stage 02 completed: model weights loaded in 4-bit mode on T4 and eight
finance responses were saved. The strict value-and-scale smoke accuracy was 1/8.
Output inspection showed arithmetic and scale mistakes plus one invalid JSON
response; none hit the 64-token output cap. This is compatibility evidence, not
a final finance or safety benchmark result. All 2,900 prepared examples fit the
1024-token complete-chat filter in that session.

Run `notebooks/03_kaggle_training_smoke.ipynb` in the same Kaggle session, with
GPU, Internet and HF_TOKEN enabled. It reconstructs its own variables, updates
the clean Git checkout, and reuses the installed Python 3.12 GPU environment.
Every cell records timed progress; every generation is saved atomically.

The sequence is:

1. Verify prerequisites and show accepted curation counts.
2. Save the original aligned model's full 200-example finance validation baseline.
3. Run five QLoRA optimizer steps on GPU 0, saving full-state checkpoints every step.
4. Verify the saved adapter, optimizer, scheduler, RNG and trainer state.
5. Generate the same eight smoke examples with the adapter and inspect outputs.
6. Create a downloadable recovery ZIP including checkpoints, data and logs.

The GPU training path completed on Kaggle on November 14, 2026: five optimizer
steps, finite mean train loss 2.108152, 3,833,856 trainable LoRA parameters, and
46.4 seconds of optimizer-loop runtime. Full-state checkpoint-5 passed file
checks, and the saved adapter generated all eight smoke responses (1/8 correct,
unchanged from the original eight-example baseline). The full original-model
validation score was 7/200 (3.5%), with a source-table bootstrap interval of
about 1.0–6.5%. These are finance-only measurements.

The first training attempt failed before step 1 because trainable BF16 weights
reached the FP16 gradient scaler. The runner now explicitly promotes trainable
parameters to FP32 after SFTTrainer setup, preserving frozen quantized weights.
The corrected run succeeded.

If a cell fails, retain its error and preceding setup logs. Keep the session/files;
rerun identical commands with --resume (already used by the notebook). Training
resumes from the latest complete local checkpoint. Missing checkpoints mean a
logged restart from zero. Restoring a deleted session requires your backup.

Five steps test loss/memory/serialization, not the research thesis. Do not change
the original baseline's precision or scoring to inflate a comparison. Baseline
and adapter generations both use 4-bit NF4, FP16 compute and a 64-token cap.

The final full finance training run will use a fresh directory after the smoke
test passes. Before the full safety-drift experiment, freeze a separate reviewed
finance safety set: direct harmful prompts, attacks and benign-sensitive requests.
The current finance-only baseline has no ASR/over-refusal measurements.

Checkpoint file checks confirm the expected recovery files exist, not that an
interrupted-resume cycle has been empirically verified. Record that distinction.

The Stage 03 archive was rebuilt with ZIP_STORED, per-25-file progress and CRC
verification after compressed archive creation stalled. The verified archive
contains 112 files (211.6 MiB) and is available in the live Kaggle session as
`/kaggle/working/finsafe-stage03-recovery.zip`. A local download was not verified.
Kaggle notebook Version 2 was quick-saved; this alone does not prove that all
working-directory artifacts have durable storage. Download the ZIP before ending
the session.
