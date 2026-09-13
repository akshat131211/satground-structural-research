# Research protocol and decision gates

## Question and claim boundary

Can a small shared adapter improve building structure in ground images predicted from one overhead RGB image, relative to ordinary adaptation with the same resources? The scene input at prediction is overhead RGB plus requested camera metadata and a single fixed training-derived illumination code. Ground photographs supervise training and evaluation but never provide test-time style, sky, masks, or depth conditioning.

Unseen facade appearance is underdetermined. The intended claim concerns measured visible structure, not exact recovery of signs, windows, hidden entrances, or metric height. See the ambiguity discussion in [ControlS2S](https://arxiv.org/abs/2502.03498).

## Hypotheses

1. H1: Building-region and boundary losses improve structure beyond matched RGB/LPIPS adaptation.
2. H2: Disagreement across independent adapters can rank and calibrate structural failures.
3. H3: Independent geometry improves results beyond information inferred from the same RGB.
4. H4: Improved structure translates to better connected views and measurable 3D geometry.

Only H1 infrastructure and initial H2 utilities are implemented now. H3/H4 depend on the earlier evidence gates. This is a candidate contribution, not an established first-ever novelty claim.

## Laptop allocation

Allow six to eight working weeks after data access, governed by evidence:

| Phase | Work | Gate |
|---|---|---|
| A: several days | Access, environment, synthetic camera checks, pretrained inference and short adaptation profiling | Finite losses, GPU fit, gradients, resume; then verify on real inputs |
| B: about one week | Baseline and roughly 200 manually reviewed development views | Enough aligned, building-containing pairs; retain no-building examples |
| C: two to three weeks | 100-view learning check; A1/A2/A3, three seeds, 5,000 steps | A3 must improve over A1; one justified extension to 10,000 allowed |
| D: one to two weeks | Freeze configuration, calibrate, audit once, stress tests | Quantitative and human review gates |

The 4,448-parameter adapter modifies the 64-channel scene tensor before the model splits it into a spatial plane and vertical feature line. Its zero-initialized restoration layer preserves A0 at initialization. Ray chunks are stitched before super-resolution because the latter includes image-wide normalization.

## Evaluation

Primary: building IoU and symmetric boundary distance divided by image diagonal. Empty-empty masks receive IoU 1 and distance 0; one empty boundary receives distance 1 so missing/extra buildings cannot disappear from the metric. Report scene composition and no-building examples alongside the mean.

Training supervision uses frozen SegFormer B0 Cityscapes pseudo-labels with confidence and dynamic-object masking. Evaluation uses separately checkpointed B1 and manually reviewed boundaries. These models share a family, so manual review remains necessary; a second-family evaluator is a useful later robustness check.

LPIPS is the appearance constraint; PSNR and SSIM are descriptive. KID is optional and FID is reserved for adequate matched samples. Bootstrap geographic groups, not individual nearby views. Do not perform best-of-many target-based sample selection. Inspect missing, extra, merged, misplaced, and obscured buildings; separate pose/date mismatch from facade ambiguity.

## Server gate

The implementation conservatively requires each of three seeds to meet all numerical thresholds:

- A3 boundary error at least 10% lower than A1.
- Paired geographic 95% interval supports improvement.
- Building IoU decreases by no more than 0.01 absolute.
- LPIPS increases by no more than 5% relative.
- Human review confirms visible structural gains without frequent new building hallucinations.
- Frozen audit and complete reproducibility records.

Reliability has a separate gate. If H1 fails, permit one diagnosis-backed revision, then preserve the negative result. Larger servers are not a substitute for an inconclusive pilot.

## Conditional expansion

After H1 passes, profile one 48 GB GPU before scaling eligible training data; retain three seeds, exclusions, full-resolution ablations, and then evaluate sealed Seattle. Compare compatible released baselines and a second geography only with authorized data. Measure throughput before estimating GPU hours or costs.

Geometry comparison: G0 RGB; G1 RGB plus geometry predicted from that RGB; G2 independent footprints/heights; G3 multiple-satellite-view reconstruction. Use measured LiDAR/DSM/CAD for metric geometry. Predicted monocular depth is not height ground truth.

Connected views progress from multiple headings at one position to short trajectories, a small building group, and mesh/Gaussian export. Evaluate against real targets as well as internal consistency. Generated views can agree while depicting the wrong scene. The later geometry and 3D systems are not implemented or claimed complete in this repository.
