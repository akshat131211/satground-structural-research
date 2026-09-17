# One diagnosed revision: smaller shared learning rate

Specified on 17 September 2026 before starting these runs. This remains the
original single-overhead-image structural-supervision study.

The saved-checkpoint diagnosis shows that ordinary A1 adaptation loses perceptual
quality between steps 100 and 200, while its RGB training error decreases. Its
output-layer norm grows with training and the fixed-view gallery becomes blurred.
These observations motivate testing update strength; they do not prove that the
learning rate is the cause or that the new setting will succeed.

| Setting | Fixed choice |
|---|---|
| Learning rate | 0.00003 for A1, A2 and A3; previously 0.0001 |
| Budget | 500 steps from fresh zero-output adapters, seed 17 |
| A1 | RGB L1 + 0.1 LPIPS |
| A2 | A1 + 0.5 building-region loss |
| A3 | A2 + 0.5 boundary loss |
| Data | Existing 100 training views and all 24 validation views |
| Evaluation labels | Existing three reviewed mask pairs plus 21 pseudo-label targets |
| Other settings | Existing resolution, optimizer, accumulation, renderer, style and weights |
| Checkpoint evaluated | Fixed final step 500; intermediate states are retained |

No new masks are added to this comparison after it starts. The separate 18-view
review packet is pending human work; any later evaluation with those labels must
be reported as a distinct label-version diagnostic, never as changed predictions.

A3 versus A1 remains the primary comparison. A2 versus A1 and A3 versus A2 isolate
the region and boundary contributions. Report all-view scores, building-target
scores, empty-target false-positive area, and each reviewed view. Keep frozen A0
as a reference to detect degradation of the ordinary control. Do not claim
success from an empty-mask threshold event or a worse ordinary baseline alone.

This is the one diagnosed revision following the 500-step failure, not an open
hyperparameter search. Do not automatically increase the budget or try more rates.
Three-seed confirmation requires a promising structural result, acceptable image
quality against both A1 and A0, and support from the expanded reviewed set. The
original audit, data-QA and server gates remain unfulfilled. The run is explicitly
exploratory and cannot open audit, calibration or Seattle imagery.

Run serially with `scripts/run_conservative_adaptation.py`. On interruption,
inspect its status/logs and use `--resume` with unchanged inputs. It retains prior
results, pins source/configuration/input hashes, and writes local logs plus an
aggregate report after all three models finish. It does not create QA approvals.
