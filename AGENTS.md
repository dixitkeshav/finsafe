# Project conventions

User requirements for all future notebooks:

- Log START, DONE/FAILED, timestamps and elapsed time for every code cell.
- Stream long-running subprocess output and retain logs under runs/.
- Add meaningful recovery points. Validate inputs/settings before reusing artifacts.
- Save each generated response atomically and support resuming a validated prefix.
- Save full trainer state, not adapter weights alone, for training recovery.
- Explain that recovery resumes at the last saved checkpoint, not the exact
  interrupted instant. Do not promise identical results across changed runtimes.
- Provide durable backup options; local Kaggle storage alone is insufficient.
- Never log or commit credentials. Notebook outputs must not display HF tokens.
- Clearly distinguish software tests, smoke tests and actual research results.

Use finsafe.progress.Progress for notebook logging. Consult docs/RECOVERY.md.
GPU code is not verified until exercised on the actual Kaggle environment.
