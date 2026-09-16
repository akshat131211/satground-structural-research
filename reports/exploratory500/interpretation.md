# Completed laptop experiment: mixed evidence, no scaling decision

Verified on 17 September 2026 in India (16 September UTC). The original question
remains: does building-region and boundary supervision improve structural
fidelity beyond ordinary RGB/perceptual adaptation? No height-estimation
research track has been added.

Both fresh adapters completed their preselected 500 optimizer steps, using
seed 17, 100 training views, 24 validation views, and 4,448 trainable parameters.
All validation views remain included. Three targets use the accepted reviewed
mask pairs; the other 21 use pseudo-labels. Geographic review remains incomplete.

## Results using the same evaluation labels

Metrics are equally weighted over ten geographic groups. Lower boundary error
and LPIPS are better; higher IoU and SSIM are better.

| Model | Boundary error | Building IoU | LPIPS | SSIM |
|---|---:|---:|---:|---:|
| A0, frozen reference | 0.195863 | 0.533314 | 0.565909 | 0.156559 |
| A1, ordinary adaptation | 0.310804 | 0.317779 | 0.600978 | 0.198647 |
| A3, structural supervision | 0.184594 | 0.567683 | 0.567869 | 0.162940 |

The prespecified A3/A1 comparison gives 40.61% relative boundary reduction,
24.99 percentage points higher IoU, and 5.51% lower LPIPS. SSIM worsens.
The paired geographic-bootstrap 95% interval for absolute boundary improvement
is [0.007997, 0.328013]. This describes this small development cohort, not
unseen-city generalization. The [original report](report.md) is preserved.

A0 was regenerated after the paired result to investigate A1 degradation. All
24 image hashes exactly match the original frozen predictions, and the current
evaluation uses the same labels, segmenter, rendering settings and pipeline
source as A1/A3. A3/A0 boundary reduction is 5.75%, with absolute-improvement
CI [0.003878, 0.020412]; IoU increases 3.44 points and LPIPS worsens 0.35%.
This is an explicitly post-run reference check, not a newly selected primary
comparison.

## Why the large primary difference is not enough

One geographic group, containing one vegetation-dominated view, accounts for
**79.23% of the net A3/A1 boundary improvement**. Its scored target contains zero
building pixels. A1 has 13 segmenter-predicted building pixels; A3 and A0 have
zero. The existing metric assigns error 1 when exactly one boundary set is empty
and error 0 when both are empty. Consequently, a tiny segmentation difference
causes a full-unit change. This measures a false positive under that convention;
it does not demonstrate repaired building outlines. The view and convention
remain in the primary result. Eight group differences favor A3, one is tied and
one favors A1; their magnitudes vary greatly.

The three previously reviewed targets also contradict a broad boundary claim:

| Reviewed view | A0 boundary | A1 boundary | A3 boundary | A1 IoU | A3 IoU |
|---|---:|---:|---:|---:|---:|
| 1 | 0.080931 | 0.072355 | 0.094215 | 0.691081 | 0.842628 |
| 2 | 0.061055 | 0.043043 | 0.065758 | 0.468693 | 0.614950 |
| 3 | 0.214713 | 0.081959 | 0.218851 | 0.136693 | 0.075907 |

A3 has worse boundary error on all three. IoU improves on the first two relative
to A1, but declines on the third. Assistant inspection of the fixed three-view
gallery finds sharper A3 textures without faithful recovery of all building
shapes; distant tall buildings remain missing in view 3. These observations do
not establish the cause, certified correspondence, or reviewed instance counts.
The reviewed targets were previously exposed and edited from proposals;
predicted masks still come from the evaluation segmenter. This is not a blind
expert benchmark.

## Verification and decision

Both training logs contain exactly 500 consecutive finite steps. The two final
adapter and optimizer states are finite, the initial RGB/LPIPS components match,
and generation/checkpoint/image hashes agree with evaluation. Recomputing the
paired report exactly reproduces its measurements. A1/A3 training-loop times
were 2,648.9/2,194.9 seconds, and peak allocated CUDA memory 597.5/668.1 MiB with
cached features. Desktop workloads and timing conditions were uncontrolled;
these are resource observations, not speed claims.

The laptop can execute this adaptation experiment. **A reproducible structural
improvement, calibrated reliability and a case for server scaling remain
unestablished.** One seed and 24 previously inspected views with incomplete data
QA cannot meet the original gate. Keep the same research question. The next
diagnosis should check segmentation/empty-building behavior and A1 degradation
on development data before defining another matched experiment. Any revised
metric or training choice must be specified prospectively and retain these
original results; no audit or Seattle data were opened.

Reproduce the completion audit from the repository root, choosing fresh output
paths. This rechecks actual local artifacts without retraining:

```powershell
.\.venv\python.exe scripts/audit_exploratory500.py --output runs/my-completion-audit.json --work runs/my-completion-check
```

[Machine-readable audit](completion-audit.json) and [primary measurements](summary.json).
The full target/prediction gallery and raw records remain in ignored local
`runs/exploratory500-completion-audit/`. No dataset imagery, masks, location
coordinates or sample identifiers are included in this published report.
