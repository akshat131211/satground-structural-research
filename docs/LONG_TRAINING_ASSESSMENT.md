# Assess longer training from saved checkpoints

The user offered a longer laptop training budget on 17 September 2026, conditional
on it being useful. The completed 21-reviewed-target evaluation does not establish
an A3/A1 boundary benefit. Training loss alone cannot answer whether extending
the schedule would improve held-out images.

Before making that decision, evaluate every saved conservative-run checkpoint:
A1, A2 and A3 at steps 100, 200, 300, 400 and 500. Generate the same 24 development
views and score them with the same completed 21-target manual index, retaining
three remaining pseudo-label targets. Regenerated step-500 predictions must
match the previously saved images exactly. This uses one serial GPU evaluation
process and performs no training, checkpoint selection, learning-rate change,
additional seed, or changes to earlier reports.

Preserve and hash checkpoint files, existing approved masks, configuration,
source, and data inputs. Keep Seattle, audit and calibration sealed. Publish
only aggregate trajectories and local-test evidence; imagery, source IDs and
raw human responses stay local.

Examine all-view and reviewed boundary error, IoU, LPIPS, and target-defined
nonempty/empty strata. Report paired geographic intervals for late changes
(300 to 500 and 400 to 500), including A3 versus A1 at the same steps. Do not
convert post-hoc inspection of these development views into an audited claim or
retroactively choose a different primary checkpoint.

Decision guidance, recorded before intermediate evaluation: a very long run is
not justified by declining training loss. Consistent late improvement on reviewed
boundaries, without image-quality or false-positive deterioration, can motivate
a separately specified bounded extension. Flat, mixed or worsening development
results favor completing pairing/coverage and label QA before further training.
These are diagnostic decisions, not a new server gate. Do not launch training
from the diagnostic runner.

```powershell
& .\.venv\python.exe scripts/run_conservative_trajectory.py --output runs/conservative-trajectory-reviewed21
```
