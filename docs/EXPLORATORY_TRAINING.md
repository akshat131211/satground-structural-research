# Longer laptop training with incomplete data review

On 16 September 2026 the user explicitly requested continuing beyond the
preliminary 100-step cap. The pipeline now supports `data_review_mode:
exploratory` with a recorded reason. Reviewed mode and its data-QA requirements
remain the default. This does not turn unreviewed data into verified data.

The next experiment fixes the budget before training:

| Setting | Specification |
|---|---|
| Model | Pinned Sat3DGen, frozen DINOv3 encoder, 4,448-parameter adapter |
| Comparison | A1 RGB/LPIPS versus A3 plus building-region/boundary losses |
| Budget | 500 optimizer steps each, initialized from zero |
| Data | Same 100 training views and 24 validation views |
| Randomness | Training seed 17; rendering seed convention 101 |
| Resolution | 256 x 256; accumulation eight; cached float32 scene features |
| Checkpoints | Every 100 steps and completion |
| Selection | Fixed final step; no validation checkpoint search |
| Evaluation | Independent B1 segmenter and three accepted target-mask pairs |

The other 21 validation targets retain pseudo-labels. These are previously
inspected development images, not a blind audit. Inference uses training-only
illumination and no target conditioning. All previous runs remain unchanged.
Earlier published structural scores used different labels; do not attribute a
label-induced score change to model improvement.

Exploratory mode retains image availability, hashes, manifest membership, exact
cross-split duplicate checks, split isolation, pseudo-label provenance, finite
gradients and frozen base weights. Pending footprint, pairing and near-duplicate
review are recorded in run identities, checkpoints, summaries and generation
records. These runs cannot open the sealed audit or support a main-study server
decision. No QA approval is synthesized.

Run from the repository root:

```powershell
& .\.venv\python.exe -u scripts/run_exploratory_pair.py
```

The runner trains A1, generates and evaluates all 24 views, then repeats for A3.
Progress is in `runs/exploratory500-seed17/status.json`, with stage logs alongside
and per-step logs in each training directory. Failures stop the queue. Code,
configuration and input hashes are checked between stages. Do not edit
`src/satground/*.py` or the pinned runner/report/config files during this run.
Unrelated scripts and documentation can be updated independently.

For a clean interruption, `--resume` uses a compatible training checkpoint.
Partial evaluations are preserved and need a fresh output. A crash may leave
`runner.lock`: check its PID before handling that exact stale file. Never start
a concurrent copy of the job.

On completion, `reports/exploratory500/` receives a checked aggregate report.
It verifies matched cohorts, budgets, checkpoints, prediction hashes, label
coverage, code/environment, finite logs and aggregate arithmetic. Interpretation
remains exploratory even if the difference is favourable. Three seeds and the
full reviewed study remain necessary before larger-server training is justified.
