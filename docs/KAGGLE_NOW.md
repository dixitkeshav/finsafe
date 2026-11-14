# What to do on Kaggle now

1. Open https://www.kaggle.com/code and create a notebook.
2. Import `notebooks/01_kaggle_tatqa.ipynb` from this repo.
3. Enable Internet and select an available NVIDIA GPU in notebook settings.
4. Sign into Hugging Face as KeshavDixit11 and accept the conditions for
   CohereLabs/tiny-aya-global and CohereLabs/tiny-aya-base.
5. In Kaggle Secrets, add HF_TOKEN, enable it for this notebook, and keep the
   token out of code/output. Use a read token for now; publishing later needs
   permission to write to Finsafe-team.
6. Run cells in order. They inspect the runtime, clone code, install the core
   environment, verify model-file access, run software checks and prepare TAT-QA.
7. Save the output and share the first cell's GPU model, memory, driver,
   Python and PyTorch/CUDA versions. Do not share the token.

The expected raw preparation result is 2,500 train / 200 validation / 200 test.
The notebook ends before GPU-stack installation or training because the compatible
CUDA build depends on the actual Kaggle image. We have not executed it on Kaggle.
This first pass checks accounts and real data without downloading model weights.

Once runtime details are known, the next sequence is compatible GPU dependencies,
token-length filtering, curation, eight-example model-generation smoke test, the
real aligned baseline, five-step SFT smoke test, then the full training budget.

Do not use “Run All” on a later training notebook until its compute plan is set.
Save notebook output and retain source/run manifests before ending a Kaggle session.
See TATQA.md for preparation details and START_HERE.md for the full experiment.
