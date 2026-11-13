# Team coordination

Repository owner: [dixitkeshav](https://github.com/dixitkeshav).
Invited collaborators: [AryanGupta1112](https://github.com/AryanGupta1112) and
[ShauryaaSharma](https://github.com/ShauryaaSharma), both write access.
Invitations are pending until accepted; this does not configure HF permissions.

Suggested division of work (proposal, not an assignment agreed by teammates):

| Person | Main responsibility | Review responsibility |
|---|---|---|
| Owner / team lead | GPU environment, model access, SFT and repair experiments | Data split and license review |
| AryanGupta1112 | Finance data, safety prompt pairs, curation and provenance | Training output sanity checks |
| ShauryaaSharma | Evaluation labels, reports, HF article and submission package | Independent review of safety labels |

All three should agree on benchmark policy and frozen test set before final runs.
For labels, rotate blinded reviews so one person does not judge only one condition.

## First working session

1. Accept GitHub invitations and run the local quick start.
2. Confirm registered team details, HF namespace, model access and available GPU.
3. Freeze one finance task and the annotation rubric.
4. Assign data preparation, GPU compatibility check and submission packaging in parallel.
5. Run baseline before training; run five SFT steps before the longer run.

## Collaboration workflow

Work on small branches, open PRs and have one teammate review changes. Example:

```bash
git switch -c data/finance-benchmark
# edit and validate files
git add data/ docs/
git commit -m "Add finance benchmark sources and split policy"
git push -u origin data/finance-benchmark
```

Keep models, raw data and logs out of Git. Publish reviewed artifacts on HF.
Commit only configurations, source, small examples and concise evidence summaries.
Avoid changing the final test set after comparing model performance on it.
