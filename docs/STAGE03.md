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

The GPU training path is implemented but not yet executed on Kaggle. If a cell
fails, send the final error and preceding setup logs. Keep the session/files;
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
