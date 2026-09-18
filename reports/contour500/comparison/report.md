# Fixed contour-correction comparison

500 fresh steps per model; seed 17. All 24 development views retained.

| Model | Boundary | Reviewed boundary | IoU | LPIPS |
|---|---:|---:|---:|---:|
| A0 | 0.089638 | 0.097656 | 0.513683 | 0.565909 |
| A1_legacy | 0.088043 | 0.094905 | 0.485935 | 0.582288 |
| A2_legacy | 0.089659 | 0.096788 | 0.521941 | 0.567256 |
| A3_legacy | 0.087750 | 0.094853 | 0.522212 | 0.567910 |
| A2_fresh | 0.089327 | 0.096426 | 0.520927 | 0.567564 |
| A3_corrected | 0.087724 | 0.094799 | 0.522016 | 0.567954 |

Fresh A2 exactly reproduces 0/24 historical prediction images.
The primary contrast is corrected A3 versus fresh A2. See all geographic intervals and target-defined strata in [summary](summary.json).
Interpretation and local visual inspection remain necessary; target coverage alone is not an improvement claim.
