# Current Kaggle stage

**Version 12 is running** as a committed GPU T4 x2 batch job:
https://www.kaggle.com/code/keshavdixit11/lexsilabs?scriptVersionId=369665762

Use notebook 06 for the full run. It pins source commit
ef272c570f62c07d680fd295df4812167e018c1d, restores the private HF archive,
installs the pinned environment and runs the experiment/publication pipeline.
HF identity and organization writes were verified. The 222,115,140-byte recovery
archive SHA256 is db8e2cbf4e40ad92ebf48b99ad7e3492ac5d60c03757d415db1d10d04c360ce0.
The configured write secret is HF_WRITE; the bootstrap also accepts HF_WRITE_TOKEN.
Read access remains in HF_TOKEN.

The GPU job runs independently of the laptop. Organizer form submission remains
separate and must be confirmed. The software suite has 18 passing tests; actual
SafeTune GPU compatibility and efficacy are checked in the running job.

# Historical recovery and baseline evidence

Completed, retained recovery run:
https://www.kaggle.com/code/keshavdixit11/lexsilabs?scriptVersionId=369625059

Version 4 was cancelled with explicit user approval after remaining queued with
zero output. The editor's Save & Run All created Version 5, which completed in
44m 58s on T4 x2. The final recovery ZIP was downloaded locally and CRC-verified:
157 files, including prepared data, checkpoint-5, predictions and review packets.

## Observed results

- Finance baseline: 7/200 (3.5%), matching the earlier aggregate result; 36%
  malformed value/scale JSON; no finance responses hit the 64-token cap.
- Recreated five-step smoke: finite loss 1.8306249380111694, full checkpoint files
  verified. This is distinct from the earlier smoke run with loss 2.108152.
- Safety pilot: 36 saved validation responses, 21 reaching the 256-token cap.
  Safety rates remain unmeasured because labels are UNREVIEWED.
- Budget diagnostic: five selected capped examples rerun with 768 tokens. Four
  stopped naturally at 309–526 tokens; one still reached 768. Their original
  answers were exact text prefixes of the longer answers. None emitted tokenizer
  EOS 6; the four completed on generation EOS 3. No stop-rule change is justified
  by these observations. The subset is not a safety-rate estimate.

## Next notebook

Use `notebooks/05_kaggle_finance_sft.ipynb`, with Version-5 retained output attached
as notebook input. The bootstrap restores finsafe-stage04b-recovery.zip when
needed and reinstalls missing GPU dependencies. Keep GPU, Internet and HF_TOKEN
enabled. Submit through the editor's Save & Run All and verify Notebook format
and actual logs; a saved version alone does not prove execution.

Notebook 05 collects a NEW original-model safety baseline at 768 tokens before
training. SFT safety evaluation also uses 768; finance remains at 64. It starts
a fresh 100-step finance run with full-state checkpoints every 10 steps. Budget
several GPU hours for all baseline/training/evaluation stages on the observed
runtime. It has not executed yet; no SFT or repair efficacy is claimed.

Review the NEW full-response packets using STAGE04.md and PROTOCOL.md; do not
copy response labels across changed response hashes. The condition-hidden review
tool can mix baseline/SFT packets after both are collected. Independent prompt
review remains necessary for this small AI-assisted pilot. Final test prompts
remain reserved. Save completed artifacts before ending an interactive session.
