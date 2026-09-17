# Exploratory 500-step laptop comparison

Each adapter trained 500 steps from zero initialization, seed 17. All 24 pre-existing validation views are retained; three use reviewed target masks.

**This is an exploratory development result with pending data review. It cannot pass the server gate.**

| Model | Boundary error | Building IoU | LPIPS |
|---|---:|---:|---:|
| A1 | 0.195721 | 0.452819 | 0.582288 |
| A3 | 0.191681 | 0.540232 | 0.567910 |

A3 versus A1: boundary reduction 2.06%; paired geographic-bootstrap absolute-improvement 95% CI [-0.004615, 0.015448]. IoU change +8.741 percentage points; LPIPS change -2.47%.

The final budget was fixed before training; no best-of-checkpoint selection was performed. Earlier 100-step experiments and their outputs are preserved. Label changes prevent direct comparison of their published structural aggregates with this mixed-label evaluation.

Single seed, previously inspected development images, mostly pseudo-labels.
Geographic alignment, footprint scale and near-duplicate review remain incomplete.
Validation bootstrap does not establish audited generalization or publication novelty.

[Aggregate evidence](summary.json). No images, coordinates or sample identifiers are published.
