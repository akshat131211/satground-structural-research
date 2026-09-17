# Does longer training have supporting evidence?

All 15 saved model states checked; no new training. Same 24 views and 21 reviewed target-mask pairs.

| Model | Step | All boundary | Reviewed boundary | Reviewed IoU | LPIPS | Empty-target FP pixels |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 100 | 0.089820 | 0.098015 | 0.491043 | 0.565383 | 86 |
| A1 | 200 | 0.087157 | 0.095137 | 0.483006 | 0.564664 | 84 |
| A1 | 300 | 0.064661 | 0.065163 | 0.523398 | 0.562874 | 0 |
| A1 | 400 | 0.091332 | 0.099526 | 0.470246 | 0.571165 | 59 |
| A1 | 500 | 0.088043 | 0.094905 | 0.464063 | 0.582288 | 55 |
| A2 | 100 | 0.087313 | 0.095457 | 0.488183 | 0.566375 | 88 |
| A2 | 200 | 0.088231 | 0.095909 | 0.491780 | 0.565676 | 82 |
| A2 | 300 | 0.086099 | 0.093677 | 0.494493 | 0.566929 | 84 |
| A2 | 400 | 0.087864 | 0.094987 | 0.494662 | 0.569300 | 95 |
| A2 | 500 | 0.089659 | 0.096788 | 0.500399 | 0.567256 | 104 |
| A3 | 100 | 0.087307 | 0.095438 | 0.487855 | 0.566359 | 88 |
| A3 | 200 | 0.088371 | 0.096133 | 0.492392 | 0.565772 | 83 |
| A3 | 300 | 0.086093 | 0.093574 | 0.494304 | 0.566457 | 83 |
| A3 | 400 | 0.088159 | 0.095194 | 0.494478 | 0.569213 | 95 |
| A3 | 500 | 0.087750 | 0.094853 | 0.500356 | 0.567910 | 99 |

Means weight geographic groups equally. False-positive pixels are pooled counts on three empty targets.
Lower boundary and LPIPS are better; higher IoU is better. Earlier reports and final checkpoint choice are unchanged.

[Interpretation](interpretation.md). [All paired intervals and strata](summary.json).
