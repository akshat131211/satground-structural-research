# Exploratory 500-step laptop comparison

Each adapter trained 500 steps from zero initialization, seed 17. All 24 pre-existing validation views are retained; three use reviewed target masks.

**This is an exploratory development result with pending data review. It cannot pass the server gate.**

| Model | Boundary error | Building IoU | LPIPS |
|---|---:|---:|---:|
| A1 | 0.310804 | 0.317779 | 0.600978 |
| A3 | 0.184594 | 0.567683 | 0.567869 |

A3 versus A1: boundary reduction 40.61%; paired geographic-bootstrap absolute-improvement 95% CI [0.007997, 0.328013]. IoU change +24.990 percentage points; LPIPS change -5.51%.

The final budget was fixed before training; no best-of-checkpoint selection was performed. Earlier 100-step experiments and their outputs are preserved. Label changes prevent direct comparison of their published structural aggregates with this mixed-label evaluation.

Single seed, previously inspected development images, mostly pseudo-labels.
Geographic alignment, footprint scale and near-duplicate review remain incomplete.
Validation bootstrap does not establish audited generalization or publication novelty.

[Aggregate evidence](summary.json). No images, coordinates or sample identifiers are published.
