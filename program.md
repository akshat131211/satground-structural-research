# SatGround agent research program

Apply the experiment discipline from Karpathy's autoresearch to the existing
single-image VIGOR + Sat3DGen structural-supervision question. The imported
`external/autoresearch/program.md` describes a different language-model task;
use this program for SatGround. Follow the user's current scope and budgets.

## Resume from evidence

1. Read `README.md`, `reports/IMPLEMENTATION_STATUS.md`,
   `docs/AUTORESEARCH.md` and the latest completed interpretation.
2. Run `.venv/python.exe scripts/autoresearch.py status`. Inspect live processes
   before starting a GPU job. The drawing server is not a training job.
3. Read actual versioned training-review submissions. Pending responses remain
   pending; uncertain/mismatched responses remain uncertain/mismatched. Never
   create a human response, redraw an accepted mask or silently replace a scene.
4. Continue independent coding/diagnosis while human review is pending. Report
   the concrete remaining review work. No repeat approval is needed for work
   already authorized, but elapsed time cannot supply missing human evidence.

## Current bounded experiment

The 500-step contour comparison is complete. Preserve its checkpoints, labels,
configurations, reports and pinned scripts. Its small observed reduction does
not establish a structural improvement. Importing its result is bookkeeping,
not a new training run or a retrospective change in its decision rule.

Next is the previously proposed training-only fitting diagnosis: after verifying
all 12 review submissions and freezing the suitable subset, specify one fresh
A2/A3 pair, 200 optimizer steps each, seed 17, learning rate 0.00003. Preserve
uncertain/rejected cases in the review record and report the usable fraction.
If no adequately supported building examples remain, stop at that finding.
Do not silently pick replacements or train with pending/pseudo-human labels.
The runner and fitting configurations are not implemented by this integration;
implement and test them against the final reviewed subset before any launch.

This fitting test asks whether known training outlines are learnable. Scores
on the fitting examples cannot support generalization or the server gate.
Fix the fitting objective, label conversion, final checkpoint and success
criterion before launching. Include matched A2 and A3 budgets and the frozen
reference. Do not tune against the existing 21 development masks during fitting.

## Bounded agent loop

1. Write one concrete hypothesis and the evidence motivating it.
2. Before the experiment, create a dedicated `codex/autoresearch/<trial>` branch
   from the explicitly recorded baseline commit. Use an isolated checkout if
   needed; never reset or discard the user's working tree. Baseline means a
   reproducible reference, not a proven improvement. Keep the branch even on
   failure, following the supplied guide's history-preserving approach.
3. Freeze an allowlist of editable experiment files and hashes of the evaluator,
   camera handling, data/label manifests, style, model, protocol and configs.
   Historical files and `external/autoresearch` are read-only. Scope new changes
   narrowly; do not edit shared core files merely to make a score look better.
4. Run relevant CPU tests and startup/resume checks. Launch one GPU child at a
   time through a serial runner with a lock and fixed update budget. Store the
   Git commit, all input hashes, stdout/stderr, finite-loss checks, elapsed time,
   peak VRAM, checkpoints and exact command locally. Never run an open-ended loop.
5. If interrupted, inspect the real error and live processes. Preserve partial
   output; resume only an unchanged compatible checkpoint after confirming no
   runner/child remains. Do not treat a crash as a zero-valued metric.
6. Evaluate the predeclared final checkpoint on the fixed eligible cohort. Keep
   boundary error, IoU, LPIPS, target-building and empty-target errors together.
   Use geographic intervals for development comparisons; identify fitting-set
   measurements explicitly. No best-of-many images, poses or checkpoints.
7. Record measured results and one decision: supported within the stated scope,
   inconclusive, regressed, or failed. A smaller mean alone is not evidence of
   a real structural gain. Preserve negative results and numerical limitations.
8. Retain candidate branches and evidence. Merge a candidate only after its
   predefined checks pass and its scope is honestly described. Never merge merely
   because one scalar decreased. Infrastructure changes may be merged separately.
9. Stop after the bounded pair and its report. A new sweep, seed, longer budget
   or server allocation needs a new concrete research decision. This instruction
   overrides the imported example's indefinite-loop/reset behavior for this study.

## Fixed boundaries

- Keep Sat3DGen, VIGOR and the single-overhead-image structural question.
- Use one fixed training-derived illumination code; no target-derived style.
- Keep Seattle, audit and calibration imagery sealed.
- The 21 reviewed development masks are evaluation-only, never training input.
- Current data QA is incomplete. No novelty, audited generalization, conference
  acceptance or server-gate claim follows from the pilot or from automation.
- Publish only code, documentation and aggregate evidence. Keep imagery, masks,
  coordinates, sample IDs, human identities/responses and checkpoints local.
- Do not install the upstream language-model dependencies into `.venv`, execute
  its data downloader, or substitute its `val_bpb` metric for building structure.

## Existing executable support

`scripts/autoresearch.py status` verifies the imported revision, counts actual
local review progress and reports the current prerequisite. `import-contour`
re-audits the completed experiment on CPU and creates an immutable aggregate
ledger in a fresh local directory. `verify-ledger` checks its hashes, decisions
and source evidence. These commands perform zero training updates; the agent,
guided by this file, remains responsible for research changes and interpretation.
