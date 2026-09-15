# Preliminary 100-step learning check

This is a real-data feasibility and learning check. It does not establish a research improvement or authorize server scaling.

100 training views; 24 fixed validation views in 10 geographic groups; seed 17; 256 x 256 outputs; 100 optimizer steps per adapter.

| Experiment | Building IoU (higher) | Boundary error (lower) | LPIPS (lower) | SSIM (higher) |
|---|---:|---:|---:|---:|
| A0 | 0.531613 | 0.203957 | 0.565909 | 0.156559 |
| A1 | 0.540302 | 0.200953 | 0.561346 | 0.173449 |
| A3 | 0.531960 | 0.200765 | 0.565367 | 0.156266 |

Values give each geographic group equal weight. Boundary distance is normalized by the image diagonal. If only one mask has an assessable boundary after ignoring uncertain pixels, the metric assigns error 1. This includes missing/extra building masks and can also affect tiny uncertain fragments.

A3 versus A1: relative boundary reduction **0.09%**; paired geographic-bootstrap 95% interval for A1 minus A3 boundary error **[-0.003246, 0.004538]**. Positive values favor A3. IoU change: -0.83 percentage points; relative LPIPS change: +0.72%.

| Training | Optimizer steps | Trainable parameters | Peak allocated VRAM (MiB) | Loop time (s) |
|---|---:|---:|---:|---:|
| A1 | 100 | 4,448 | 1971.8 | 719.2 |
| A3 | 100 | 4,448 | 668.1 | 605.2 |

Both logs contain every optimizer step with finite losses and gradient norms. A1 was deliberately stopped after step 10 and resumed to step 100. Cache and initialization differences prevent a controlled speed comparison.

Views receiving the saturated boundary penalty (A0: 3, A1: 3, A3: 3) should be checked against human masks. They are retained in all reported primary metrics; no difficult views were removed.

## Limits and next decision

- Single seed and 100 optimizer steps; 24 validation views, not the sealed audit.
- Reference building masks are independent-segmenter pseudo-labels, not human annotations.
- The pretrained model may have seen these training-city holdouts.
- The bootstrap is exploratory with only ten geographic groups.
- Training timing excludes initialization; A1 populated caches that A3 reused.

Complete image alignment, footprint, duplicate, and human boundary review before the main A1/A2/A3 study. Retain matched budgets and three seeds. Seattle and the final laptop audit have not been evaluated.

The summary includes source, checkpoint, manifest, training-log, and metric-file hashes. Images and per-location records remain in local ignored directories.
