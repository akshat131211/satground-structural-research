# Existing objective gradients on training views

100 unchanged training views at three adapter states; no optimizer updates. Weighted gradients before AdamW.

| A3 state | Boundary / region norm | Boundary / A2 norm | Cosine A2 vs A3 | Boundary vs A2 negative |
|---|---:|---:|---:|---:|
| 0 | 0.0492 | 0.0485 | 0.999300 | 48/100 |
| 300 | 0.0503 | 0.0498 | 0.999292 | 43/100 |
| 500 | 0.0498 | 0.0487 | 0.999291 | 47/100 |

Ratios and cosines are per-image medians, not ratios of aggregated losses. A2 here means the A2 objective at the SAME A3 parameters.
Undefined ratios/cosines remain unavailable. Full distributions, equal-group means and target-defined strata are in the summary.

[Summary](summary.json). [Interpretation](../interpretation.md).
