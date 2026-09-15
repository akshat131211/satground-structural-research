# Laptop geometry probe

Preliminary, single-seed development evidence. No server-scale or publication claim is established. Human review is pending.

## Existing-adapter field interventions

Each row uses the same frozen-baseline ray samples. Lower boundary/LPIPS and higher IoU are better.

| Adapter | Field intervention | Boundary error | Building IoU | LPIPS |
|---|---|---:|---:|---:|
| A1 | base | 0.203957 | 0.531613 | 0.565909 |
| A1 | density | 0.202706 | 0.533198 | 0.566559 |
| A1 | appearance | 0.201994 | 0.535450 | 0.563079 |
| A1 | joint | 0.203275 | 0.524772 | 0.564395 |
| A3 | base | 0.203957 | 0.531613 | 0.565909 |
| A3 | density | 0.203952 | 0.528566 | 0.565973 |
| A3 | appearance | 0.203804 | 0.534826 | 0.565969 |
| A3 | joint | 0.202849 | 0.532284 | 0.566204 |

Unchanged opacity/radial maps in the appearance-only intervention are a software control. Changes in the density intervention do not establish more accurate buildings. Raw maps and every preselected view remain in the ignored local runs directory.

## Matched 100-step control

| Run | Boundary error | Building IoU | LPIPS | Loop seconds | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|
| GJ | 0.203285 | 0.532346 | 0.566210 | 721.8 | 668.1 |
| GD | 0.203049 | 0.535316 | 0.566611 | 682.4 | 668.1 |

GD versus GJ: boundary reduction **0.12%**; paired geographic-bootstrap 95% interval for absolute improvement [-0.000998, 0.001481]. IoU change 0.30 percentage points; LPIPS change 0.07%.

Both use the same A3 image-space losses, adapter capacity, samples, and optimizer budget. GJ adapts density and color features; GD uses baseline color features. All feature caches were already populated. Loop memory excludes initialization and separate desktop/driver allocations.

## Interpretation limits

- 24 repeatedly inspected validation views; 10 geographic groups; one seed.
- pseudo-mask scores; no independently measured geometric accuracy.
- frozen proposal can miss newly occupied space.
- exploratory bootstrap intervals; no multiple-comparison correction.

The next method stage needs reviewed instance correspondences and a direct semantic-rendering control. These interventions and GD alone are not the proposed building-identity contribution.

Protocol and commands: [geometry pilot](../../docs/GEOMETRY_PILOT.md). Full aggregate records: [summary.json](summary.json).
