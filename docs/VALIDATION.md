# Validation status

Checked November 14, 2026.

| Component | Status |
|---|---|
| Project-local Python 3.12.14 on macOS ARM64 | Installed |
| Core environment | Installed; `finsafe doctor` passes |
| CuratorKIT 0.1.15 | Ran cleaning, exact/MinHash dedup and accepted-row export |
| Chat conversion and HF loader metadata | Produced from accepted curation output |
| AuditKIT 1.1.11 | Evaluated recorded test-fixture outputs through the real API |
| Scoring, leakage, logging and resume checks | Nine tests pass |
| Changed-output replay | Verified no stale cached result is reused |
| Annotation response binding | Verified changed-response hashes are rejected |
| All Python source files | Parsed/compiled successfully |
| GPU dependencies | 167 packages solver-resolved for Linux x86_64 / Python 3.12 |
| Full GPU environment installation | Not run on this Mac |
| Tiny Aya loading/training/repair | Not run; needs HF sign-in/model access and GPU runtime |
| HF artifacts and collections | Namespace selected: Finsafe-team; artifact repos and runtime sign-in pending |
| Actual safety/finance results | Not yet measured |
| Hackathon submission | Not submitted |

Unit-test fixtures are synthetic software checks, not model experiment results.
The GPU lock is a dependency resolution, not evidence of CUDA/runtime compatibility.
CI checks core code, example data and curation only; it does not certify GPU methods.

Integration adjustments made during setup:

- CuratorKIT 0.1.15 lacks the messages export described by newer main-branch docs.
  The project converts the accepted Alpaca rows explicitly and writes loader metadata.
- AuditKIT's current API uses `scorers=`, not `metrics=`. The integration was tested.
- Callable replay uses `cache=False`, because callable identity does not fingerprint
  the generated outputs. Reports separately retain dataset and predictions hashes.
- AuditKIT initializes a cache directory even when caching is disabled; its location
  defaults to the project's `.cache` when no XDG cache override is supplied.
- Replay throughput is not model throughput and must not be used for cost claims.

## TAT-QA integration

Implemented and locally exercised the pinned TAT-QA preparation: 2,500 training,
200 validation and 200 final-test numeric examples, before tokenizer filtering.
Numeric-scale scoring and source-table grouping tests pass. New Kaggle notebook
01_kaggle_tatqa.ipynb code cells compile; Kaggle execution is still pending.
No model weights were loaded and no model accuracy or safety results were measured.

## Kaggle status supplied by user

Notebook 01 executed successfully on the reported Kaggle session: HF identity and
both model config accesses verified, seven then-current tests passed, and TAT-QA
prepared 2,500/200/200 examples. Host: Python 3.13.16, torch 2.11.0+cu130,
CUDA driver 580.178.04, two Tesla T4s. The core environment is Python 3.12.3.

Stage 02 adds per-cell logs, a matched GPU dependency constraint set, token filtering,
real curation and resumable 4-bit baseline generation. Constraints solver-resolve;
notebook code cells compile. Generation resume/changed-contract and failed-stage
logging/redaction are locally tested. Training full-state recovery and optional
private HF backup are implemented but require actual GPU verification.
