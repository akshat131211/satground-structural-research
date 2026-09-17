# Expanded-label evaluation of unchanged predictions

21 reviewed targets and three remaining pseudo-labelled targets; all 24 development views retained.
These measurements use a new label version. The original experiment remains unchanged.

| Model | All-view boundary | Reviewed boundary | All-view IoU | Reviewed IoU | LPIPS |
|---|---:|---:|---:|---:|---:|
| A0 | 0.089638 | 0.097656 | 0.513683 | 0.491171 | 0.565909 |
| A1 | 0.088043 | 0.094905 | 0.485935 | 0.464063 | 0.582288 |
| A2 | 0.089659 | 0.096788 | 0.521941 | 0.500399 | 0.567256 |
| A3 | 0.087750 | 0.094853 | 0.522212 | 0.500356 | 0.567910 |

Each geographic group has equal weight within the reported cohort. LPIPS is unchanged by target-mask review.

[Detailed paired intervals and strata](summary.json). [Interpretation](interpretation.md).
