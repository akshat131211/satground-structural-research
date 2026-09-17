# Checkpoint trajectory: unchanged model and evaluation

Post-hoc development diagnosis. All 24 views retained; no new training or primary-checkpoint selection.

| Model | Step | All-view boundary | Building-target boundary | Building-target IoU | LPIPS | Empty-target FP area (%) |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0 | 0.195863 | 0.217625 | 0.481460 | 0.565909 | 1.607006 |
| A1 | 100 | 0.193471 | 0.214968 | 0.491255 | 0.561335 | 0.941597 |
| A1 | 200 | 0.198759 | 0.220843 | 0.407967 | 0.592269 | 1.066423 |
| A1 | 300 | 0.205919 | 0.228799 | 0.369716 | 0.599292 | 1.435985 |
| A1 | 400 | 0.309321 | 0.232579 | 0.361234 | 0.596561 | 0.918053 |
| A1 | 500 | 0.310804 | 0.234226 | 0.353088 | 0.600978 | 0.139961 |
| A3 | 100 | 0.192751 | 0.214168 | 0.481019 | 0.565346 | 1.310177 |
| A3 | 200 | 0.192109 | 0.213454 | 0.477608 | 0.568458 | 1.588331 |
| A3 | 300 | 0.190063 | 0.211181 | 0.501634 | 0.568202 | 1.236461 |
| A3 | 400 | 0.183635 | 0.204038 | 0.512753 | 0.567112 | 0.843310 |
| A3 | 500 | 0.184594 | 0.205105 | 0.519647 | 0.567869 | 1.856657 |

Target partition: 22 building-containing scored masks, 2 empty scored masks. Three reviewed targets; others are pseudo-labels.
Each stratum uses equal geographic-group weighting within that stratum. Stratum means do not sum to the all-view mean.
FP area is predicted building pixels divided by valid pixels in empty targets, then averaged by geographic group.
Original empty-boundary penalties are unchanged. A nonempty scored mask does not guarantee a nonempty evaluable boundary.

[Aggregate measurements and reviewed-view trajectories](summary.json).
