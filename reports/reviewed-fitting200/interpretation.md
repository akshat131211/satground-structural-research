# Eight reviewed training views: final fitting diagnosis

The final 200-step A3 adapter does **not meet the predeclared fitting-response
criterion**. Its normalized boundary error is 3.158% lower than A2 and 5.059%
lower than frozen A0; the criterion required at least 10% against both. The
paired images do not show clear recovery of the major building-layout errors.
This bounded experiment does not justify longer training or server scaling.

All measurements below concern the same eight **training examples**, with one
seed. They provide no held-out generalization or novelty evidence. The original
single-image structural-supervision question, VIGOR and Sat3DGen are unchanged.

| Final model | Boundary error, lower is better | Building IoU, higher is better | LPIPS, lower is better | SSIM |
|---|---:|---:|---:|---:|
| Frozen A0 | 0.063514 | 0.556596 | 0.537679 | 0.175838 |
| A2: RGB/perceptual + region | 0.062268 | 0.572865 | 0.538090 | 0.176807 |
| A3: A2 + boundary | 0.060301 | 0.574493 | 0.538196 | 0.177103 |

A3 improves the primary boundary score on five of eight views against A2 and
seven against A0. IoU changes are +0.163 percentage points against A2 and +1.790
against A0; relative LPIPS increases are 0.0197% and 0.0961%. These satisfy the
declared quality limits. Both 10% boundary checks fail, so the unchanged decision
is `fitting_response_not_established`.

All eight targets and predictions contain buildings and nonempty primary
boundaries. There are no building-free targets, maximum empty-boundary penalties
or zero-building predictions in this cohort. The empty-target stratum is absent;
this run cannot assess building hallucinations on building-free scenes. Scored
false-positive area fractions are 0.032784, 0.034799 and 0.034472 for A0, A2 and
A3 respectively.

The separate [post-hoc boundary diagnostic](final-boundary-diagnostic.json)
excludes only the outermost image rows and columns from both boundary sets. Its
errors are 0.116382 for A0, 0.116397 for A2 and 0.116719 for A3: A3 is 0.277%
worse than A2. Every alternate boundary set is nonempty, so these changes do not
come from its empty-edge convention. This diagnostic was defined after A0
evaluation; it neither replaces nor rescues the primary criterion.

The [support audit](boundary-support-audit.json) explains why interpretation
requires care. Only 1,043 positive training contour pixels survive the actual
validity-neighborhood operator, occupying 0.249% of its scored support. Of the
2,263 primary target edges, 1,676 (74.1%) lie on the image perimeter. Training and
evaluation therefore emphasize different boundary sets for these crops.

Assistant inspection of all eight paired RGB and mask comparisons found small
changes rather than clear restoration of the missing or substituted buildings.
The arched building and openings remain missing in two views, and incorrect
skyline, placement and forecourt arrangements remain in others. A high semantic
extent IoU can coexist with an incorrect facade. The fourth fitting view supplies
82.7% of the net A3-versus-A2 primary gain, while its misplaced tree, stairs and
background building remain visually apparent. This concentration is descriptive,
not grounds to remove the view or change the criterion.

A possible pole/crossbar label in another accepted mask remains an annotation-QA
question at 256 pixels. Its approved pixels were preserved. Assistant inspection
is not independent human certification of label accuracy or building identity.
The unmodified [automatic aggregate](aggregate.json) retains its initial gallery-
review flag; the separate [completion audit](completion-audit.json) records the
actual assistant visual inspection and its limitations.

Both models completed exactly 200 fresh optimizer updates at learning rate
0.00003, seed 17, with the same eight-view sampling schedule and final-step
scoring. All 400 rows are finite. A2 took 954 seconds and A3 1,158 seconds; each
recorded a peak PyTorch allocated-memory value of 668.1 MiB. This is not total
device-memory usage. The eight periodic checkpoints and two final copies passed
adapter, optimizer and saved-RNG payload checks. Actual GPU checkpoint restore
and resume were not exercised.

All 141 frozen source/input pins, exact reviewed labels, 24 generated images,
24 evaluated masks, 24 target copies and 24 finite probability arrays passed
verification. The aggregate and local gallery reproduce byte-for-byte to a
fresh report path. Runner and child exit and lock release are verified. The
original 12 submissions, four excluded views, 21 development mask pairs and
earlier experiments are preserved. Seattle, audit and calibration imagery was
not opened; no development images were generated or evaluated by this run.

No additional training is queued. The next methodological priority is to resolve
the sparse-contour and measurement mismatch, including independent annotation
QA, before proposing another bounded comparison. A longer budget cannot by
itself establish that this objective learns the intended visible outlines.

The executable checks are `scripts/report_reviewed_fitting.py`,
`scripts/audit_reviewed_fitting_outputs.py` and
`scripts/diagnose_fitting_boundaries.py`. They operate on ignored local inputs;
only code, documentation and aggregate evidence are published.

The full local suite passed **221 tests**, with two optional CUDA tests skipped.
The new CPU completion auditor contributes 28 synthetic tests, including
tampered checkpoints/targets, malformed output sets and live-worker rejection.
