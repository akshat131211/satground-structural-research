# Completed training submissions: two targeted checks before fitting

All 12 views were submitted and verified on 20 September 2026. Exact versioned
PNG masks, target identities, completion pixels, context revisions and their
immutable histories agree. The original submissions are preserved locally under
`data/review/training12-submitted-v1`; no submitted pixels were changed.

Eleven views have a supported-pairing selection and one has a mismatch selection.
Eleven have mostly-inside crop support and one is uncertain. Ten satisfy both
supported/mostly-inside choices and have both masks marked checked. These are
human judgments, not independent proof of exact geographic alignment.

## Disposition

- View 2: preserve the human mismatch decision; exclude from fitting.
- View 8: preserve uncertain crop support; exclude from fitting.
- View 1: assistant inspection flags apparent non-building ground beneath the
  parked car in the building mask, plus scored bin pixels in the validity mask.
  Request a targeted correction or exclude it. This is an assistant finding;
  it does not alter or replace the human submission.
- View 10: the submitted building mask appears reasonable, but zero positive
  target contour pixels survive the objective's 3-by-3 validity-neighborhood
  check. Review clear roofline and building/sidewalk transitions in Validity,
  scoring only confidently visible pixels on both sides of an edge. Do not
  include vehicles or hidden surfaces just to increase a count. Alternatively,
  exclude this view from the contour fitting diagnosis.

The other eight are candidates for the bounded diagnostic, pending final subset
verification. Retain all rejected/uncertain examples in the audit record and
report the usable fraction. No replacement examples are selected. If a view is
corrected, use Save & Next again and freeze a new snapshot; version 1 is immutable.

## Measured contour support

The ten human-supported/mostly-inside views contain 1,985 positive target boundary
pixels whose full 3-by-3 neighborhood is valid. One of the ten contributes zero.
Counts use the same hard morphological gradient and validity erosion as the
existing structural objective, including equivalent image-border behavior.
Six tests check exact snapshots, stale completions, artifact/history tampering,
private output boundaries and agreement with the PyTorch training operator.

These counts establish available supervision, not annotation accuracy or a
model improvement. The snapshots intentionally contain no training-eligible
index. No GPU training or inference ran during verification. The next fitting
test remains one fresh A2/A3 pair of 200 steps each, seed 17, learning rate
0.00003, after resolving the subset and fixing the fitting protocol. It would
measure learnability on training examples only. Dataset-wide QA is incomplete.

## Reproduce without overwriting

```powershell
.venv/python.exe scripts/verify_training_review.py --output data/review/training12-submitted-v2
```

Use a fresh output. The command fails if current masks or scene notes differ
from the completed versions. It preserves private submissions and does not
assign approval or eligibility. Only the aggregate `summary.json` is suitable
for publication; `submissions.json` contains private annotations and identifiers.

See [aggregate verification](../reports/TRAINING_REVIEW_VERIFICATION.json).
