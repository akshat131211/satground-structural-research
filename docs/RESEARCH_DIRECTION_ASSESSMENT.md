# Building identity and geometric fidelity in satellite-to-ground generation

Scope note, 16 September 2026: this document retains candidate mechanisms and
prior-work analysis supporting the original structural-fidelity question.
It does not authorize additional research tracks. The active study is the
matched Sat3DGen A1/A3 experiment in [the protocol](RESEARCH_PROTOCOL.md).
The subsequent discussion of shadow-based height estimation is outside scope.

The recommended next step is a small mechanism study on the existing VIGOR pipeline, followed by a narrowly defined geometry adaptation experiment if the diagnosis supports it. The current result establishes laptop feasibility. It does not yet establish a contribution suitable for a main-track computer-vision conference. This assessment covers primary literature available through 15 September 2026 and distinguishes existing methods, measured project results, and proposed research.

The candidate question is: **Can building identity inferred from a single overhead image constrain a shared 3D representation so that ground-view adaptation corrects building extent and separation, rather than satisfying structural image losses through appearance changes?** A second, optional question concerns identifying regions whose structural predictions remain unreliable. Neither question is a verified first-ever novelty claim.

## Current model and measured result

The project uses the released **Sat3DGen**, whose publication is listed in the ICLR 2026 proceedings.[^1] The exact installed source revision is `70763aff4d9fc5508e67685c906f84d3cebb34fe`; the model revision is `fa0ed75aff7f42dcd749764787c0184fcab0b12c`. The following description comes from the installed checkpoint configuration and executable code, rather than an assumed architecture name.

| Component | Actual implementation | Role |
|---|---|---|
| Satellite encoder | DINOv3 ViT-L/16, `dinov3-large-sat` | Extract overhead image features |
| Scene decoder | Pretrained decoder producing a 64-channel, 320 x 320 tensor | Construct the latent scene representation |
| Representation | Released `oneplane` configuration: a 32-channel spatial plane and a 32-channel feature line | Supply position-dependent features to the renderer |
| Rendering | Density/color MLP, volume rendering, 2x image super-resolution | Produce the requested 256 x 256 ground image |
| Current adaptation | Shared residual convolutional adapter, width 16 | Update 4,448 parameters while the 381,022,223 base parameters remain frozen |
| Training labels | Frozen SegFormer B0, Cityscapes | Produce building pseudo-labels and valid-pixel masks |
| Evaluation labels | Separately checkpointed SegFormer B1, Cityscapes | Evaluate generated and real images using a different checkpoint |
| Appearance metric/loss | LPIPS with AlexNet | Measure perceptual similarity |

The inference path is overhead RGB plus camera metadata, through the frozen encoder/decoder, adapter, and renderer. A fixed illumination code is computed from training panoramas. No target photograph supplies style, geometry, or labels at test time. The public paper describes a triplane formulation; the installed checkpoint uses `oneplane`. This release difference should be stated in any reproduction section, and its implications should not be guessed.[^2]

The completed learning check used 100 training views, one seed, 100 optimizer steps, and 24 validation views in ten geographic groups. The training subset contains **99 satellite tiles and 100 distinct panoramas**: 98 tiles have only one selected training view. The validation subset contains 23 tiles and 23 panoramas. These counts limit what this check can reveal about learning a shared scene.

| Method | Building IoU | Normalized boundary error | LPIPS |
|---|---:|---:|---:|
| A0, frozen | 0.531613 | 0.203957 | 0.565909 |
| A1, RGB/perceptual adapter | 0.540302 | 0.200953 | 0.561346 |
| A3, additional region/boundary losses | 0.531960 | 0.200765 | 0.565367 |

A3 reduces boundary error by 0.09% relative to A1. The paired geographic-bootstrap 95% interval for the absolute error reduction is [-0.003246, 0.004538]. IoU decreases by 0.83 percentage points and LPIPS increases by 0.72%. The result is inconclusive for the structural hypothesis. It is neither evidence of success nor enough training to conclude that structural supervision cannot work. The complete measurements and hashes are in `reports/learning100/`.

Three validation views receive boundary error 1 in every variant. One contains tiny uncertain mask fragments. That is a metric diagnostic, not permission to delete examples or alter the score after seeing results. Human masks and a separately specified supplementary boundary analysis are needed before interpreting small score changes. Both semantic evaluators belong to the SegFormer family; switching B0 to B1 does not eliminate correlated evaluation bias.

## Prior work that limits novelty claims

The following is a targeted overlap matrix, not an exhaustive claim that every relevant paper has been found. Preprints are identified separately from verified conference publications. A missing implementation is an evaluation limitation, not evidence that a contribution is novel.

| Work | Established overlap | Consequence for this project |
|---|---|---|
| Geometry-Aware Satellite-to-Ground Image Synthesis, CVPR 2020 | Predicted height and geometric projection connect overhead and ground views.[^3] | Geometric projection or height prediction alone is established. |
| ControlS2S, ICLR 2025 | Pose alignment and controllable satellite-to-street synthesis.[^4] | Pose correction and environmental prompts are not an unoccupied contribution. |
| Satellite to GroundScape, CVPR 2025 | Multi-satellite conditions and temporal denoising for neighboring ground views.[^5] | Multi-view consistency overlaps; also distinguish its input regime from single-image prediction. |
| SatDreamer360, ICLR 2026 | Single-image panoramic sequences, triplane features, ray attention, and epipolar attention; introduces VIGOR++.[^6] | Multiple views, ray attention, and consistency losses alone are insufficient differentiation. |
| Sat3DGen, ICLR 2026 | Geometry-oriented training with gravity regularization, satellite depth priors, spatial tokens, and perspective supervision.[^7] | An additional depth, skyline, or generic geometric loss needs a more specific contribution. |
| MagicCity, ICCV 2025 | Geometry-aware multi-view generation and subsequent 3D reconstruction.[^8] | Generating views and fitting 3D is already studied. |
| Sat2City v2, September 2026 revision of a preprint | Adapts a native 3D prior with real satellite–mesh pairs; generated geometry anchors appearance.[^9] | Geometry/appearance separation and real single-image mesh generation are already active research directions. |
| Sat2RealCity, preprint | Building entities, OSM structural priors, and controlled appearance.[^10] | Instance-level generation and external footprint/height guidance are not independently new. |
| Sat-Skylines, 3DV 2026 | Satellite images plus coarse geometry guide detailed building generation.[^11] | Adding coarse building geometry to a pretrained 3D generator overlaps substantially. |
| Semantic-NeRF; Panoptic NeRF; Semantic Ray | Semantic rendering, instance fields, semantic geometry refinement, and generalizable semantic fields.[^12][^13][^14] | Moving semantic supervision into a renderer must be compared with these mechanisms. |
| NeRF++ | Discusses shape–radiance ambiguity.[^15] | The possibility of appearance fitting despite wrong geometry is established in general neural rendering. |
| Conditional-Flow NeRF; OracleGS | Uncertainty in radiance fields and evidence-based validation of generated views.[^16][^17] | An uncertainty head or uncertainty weighting alone is not a sufficient novelty claim. |
| Satellite-to-Street post-disaster synthesis, 2026 preprint | Reports a realism–structural-fidelity trade-off and proposes structure-aware evaluation.[^18] | The broad observation that realistic images can be structurally wrong is already documented in this application area. |

Two omissions from the earlier project matrix materially change the positioning: Sat2City v2 and Sat2RealCity. A paper framed simply as “better geometry and appearance from a satellite image” would face substantial overlap. The useful remaining question is more precise: whether **cross-view building identity and explicitly constrained geometry updates** improve generalization under single-image inference and sparse real supervision.

## Recommended method hypothesis

### Diagnose whether the present adaptation changes structure

The current adapter changes features used by both the density and color heads. Its structural losses are computed from a segmenter applied to the final RGB image. Therefore, the computational graph permits several ways to change the score, including geometry, color features, and how the fixed renderer translates those features into RGB. The observed near-zero gain does not reveal which mechanism occurred.

The first new experiment should expose density, opacity, depth, and renderer features for the existing A0/A1/A3 checkpoints. On a fixed development subset, compare the normal renders with two controlled interventions: adapted density with baseline color features, and baseline density with adapted color features. Query both fields at the same 3D locations, keep illumination and image super-resolution fixed, and record the ray-sampling convention. These hybrids are diagnostics rather than deployable models; density changes also affect transmittance and importance sampling, so a shared sampling schedule is needed for the controlled comparison.

If the apparent gains persist mainly through color changes while density-derived structure barely changes, this supports investigating how gradients are directed into geometry. If neither field changes enough, adapter capacity or optimization is a competing explanation. If improvements appear in a tiny training set but vanish on held-out locations, generalization becomes the issue. These outcomes require different responses; increasing loss weights before identifying them is poorly motivated.

### Constrain structural updates through building identity

The proposed extension would infer soft building-instance regions from the overhead RGB image and link only unambiguous training instances to their ground-view annotations. Distinguish an observed roof region from a true ground footprint: off-nadir imaging, overhangs, vegetation, and temporal changes can make them disagree. Ambiguous regions must retain an unknown label rather than an invented correspondence.

A small geometry branch would predict residual density adjustments from scene features and 3D position. During the first geometry experiment, the color pathway remains frozen and reads the baseline features. A later appearance branch can be optimized with density fixed. Update after this assessment: the [GJ/GD laptop control](../reports/geometry-pilot/report.md) now routes the existing adapter to joint versus density-only queries using shared A0 proposals. Its 100-step comparison did not establish a meaningful gain. A dedicated position-conditioned residual branch and rendered building-identity supervision remain proposed.

The crucial supervision would compare **rendered building identity or occupancy**, not just buildings recognized in a completed RGB image. For a ray r, a standard semantic volume-rendering construction is:

`P_j(r) = sum_k T_k(sigma) * alpha_k(sigma) * pi_j(x_k)`

Here `pi_j` is the probability that a sampled point belongs to building j, and `T_k * alpha_k` is its rendering weight. Fixing a validated instance-assignment field during a geometry update means a reduction in this semantic loss must act through density. This equation and semantic rendering are established tools; they are not the proposed novelty.[^12][^13]

The research challenge is to obtain useful instance constraints from **one overhead image and sparse paired training views**, transfer them to unseen locations, and prevent incorrect correspondences from forcing false geometry. Infinitely extruding every roof mask into a building column would introduce a strong and often wrong height prior. Start with carefully reviewed instances and soft spatial support; test a simple extrusion baseline explicitly. Reject a design that needs unavailable ground photographs or measured heights at inference to work.

### Handle limited observation support without hiding errors

A ground camera can see buildings outside the overhead crop, vegetation can hide boundaries, and some overhead regions cannot identify a particular facade. Use camera geometry and validated correspondence confidence to distinguish supported structural supervision from ambiguous supervision. Crop membership alone does not establish that a facade was observed from above.

For the first experiment, support weights should be derived from fixed metadata or reviewed labels, and should not be trainable gates that can suppress every difficult pixel. Report the retained fraction of supervision. Evaluation must retain all preselected views and show both overall and support-stratified performance. A model must not improve its published score merely by declaring difficult buildings unknown.

The intended contribution would be the demonstrated advantage of a particular cross-view constraint and geometry update mechanism, with evidence explaining when it works. Combining standard modules under a new name would be weak positioning. A comparison against a direct adaptation of semantic/panoptic rendering is essential, because that may explain the entire gain.

## Research directions ranked for this project

| Direction | Laptop feasibility | Main novelty risk | Recommendation |
|---|---|---|---|
| Building identity constraints with explicit density adaptation | Small cached-feature experiments look plausible; profile the new backward path | Existing semantic/panoptic NeRF mechanisms | Primary hypothesis after mechanism diagnosis |
| Structural reliability under missing evidence and geographic shift | Evaluation and small heads are feasible | Generic calibration and uncertainty are established | Add only if labels and a clear failure event support it |
| Benchmark of per-building structural failures across generators | Local inspection feasible; annotation and baseline coverage cost dominate | A small collection of pseudo-label metrics is insufficient | Supporting contribution; standalone only if broad, reusable, and revealing |
| External map heights, footprints, or multi-satellite geometry | Depends on data and alignment | Sat-Skylines/Sat2RealCity/GroundScape overlap | Keep as a separate additional-information setting |
| Replace the backbone, train a larger diffusion model, or optimize a full city | Poor fit for the immediate laptop study | Cost rises before the hypothesis is established | Defer |

Reliability should answer a defined question, such as whether a building's visible outline exceeds a prespecified error tolerance. Compare against segmentation entropy, distance to the image boundary, constant confidence, and a small ensemble. Use held-out calibration locations and risk–coverage curves. No calibration guarantee should be asserted for an unseen city without the assumptions supporting it; geographic shift can invalidate ordinary exchangeability assumptions. Agreement among generators is not proof of correctness.

## Concrete next experiments

The next working allocation is approximately two weeks after the needed annotations are available. This is a planning allocation, not a promise of completion or performance.

| Stage | Concrete task | Decision evidence |
|---|---|---|
| 1. Data and scoring | Review the prepared 200-view packet; resolve camera/footprint and duplicate issues; label visible buildings and uncertainty | Actual reviewed labels; no fabricated verification flags |
| 2. Mechanism probe | Export A0/A1/A3 geometric diagnostics; run density/color interventions on a fixed subset | Identify appearance effects, geometric effects, insufficient updates, or alignment errors |
| 3. Tiny fit check | Use roughly 20–32 reviewed training tiles with multiple eligible views; test whether geometry can fit known visible boundaries | A small optimization/capacity check, explicitly not generalization evidence |
| 4. Matched comparison | Compare ordinary adaptation, current image-space structural losses, direct semantic-rendering adaptation, and proposed identity/support mechanism | Same samples, ray budgets, training steps, supervision, and trainable-parameter controls |
| 5. Generalization | Evaluate untouched development locations, then the existing three-seed pilot after configuration selection | Consistent structural benefit without unacceptable appearance loss |
| 6. Final evidence | Freeze the protocol, evaluate the laptop audit, and later sealed Seattle and compatible independent geometry | Evidence sufficient for the claimed image/3D/reliability contribution |

Rotating crops of a single panorama provides useful angular coverage but no new translational parallax. Use separate panoramas from the same overhead tile where legitimate pairs exist, and report those cases separately. VIGOR supports sparse paired-view checks; it does not become a dense trajectory ground-truth dataset through synthetic cropping. For a temporal claim, obtain a compatible sequence dataset through an authorized release.[^6][^7]

The present A0–A3 records and manifests must remain reproducible. Introduce new experiment names and write the revised objective, correspondence assumptions, capacity controls, and metrics before examining a new audit. Do not relabel the current A3 or lower the existing server thresholds after seeing its result. The earlier 10% boundary reduction is an internal resource-allocation rule, not a conference acceptance criterion.

A renderer-level prototype may fit the 6 GB GPU because the expensive backbone can remain frozen and cached. This is a hypothesis to profile, not a demonstrated memory result. Start with batch size one and sequential views; multi-view consistency losses may require additional graphs and more memory. The observed A3 peak of 668 MiB used cached features and excludes initialization/driver allocations, so it is not a blanket budget for a redesigned model.

## What would make the paper convincing

A main-track submission needs a clear knowledge advance and sound evidence. Official CVPR guidance asks reviewers to weigh novelty and potential impact alongside performance, while ECCV's algorithm guidance emphasizes meaningful differentiation, justified design choices, fair experiments, reproducibility, and generality.[^19][^20] Neither prescribes a universal percentage improvement or requires the largest model.

For this project, a convincing paper would ideally demonstrate:

1. **A specific failure mechanism:** identify how image-based adaptation can leave building geometry wrong, and how this varies with observation support. The broad shape–radiance problem is prior art; the useful contribution is the satellite-specific diagnosis and remedy.
2. **A method that survives close controls:** improvement beyond ordinary adaptation, additional parameter count, multiple-view sampling, generic semantic rendering, and simple confidence weighting. A2 remains useful for isolating the original region versus boundary terms.
3. **Geographic generalization:** matched evaluation on unseen locations, a sealed unseen city, and a compatible second setting if the claim is broad. The current training-city holdout may have been seen during base-model pretraining.
4. **Independent structural evidence:** manually reviewed outlines/instances plus a second-family semantic evaluator. Mask2Former is one established option, but its checkpoint taxonomy and memory requirements need verification.[^21] A category called building is not necessarily an individual-building instance label.
5. **Geometry evidence for a geometry claim:** independent DSM/LiDAR/CAD for height or surface accuracy. Depth predicted by another monocular model is a prior, not metric ground truth. Connected-view agreement alone can reward consistently wrong structures.
6. **A reusable and honest artifact:** code, pinned assets, data manifests or permitted reconstruction instructions, aggregate results, uncertainty intervals, and representative failure cases. A dataset contribution requires a credible release arrangement; the existing private VIGOR image folder is not a new public benchmark.[^19]

Suggested working title: **“Preserving Building Identity in Satellite-to-Ground Scene Generation.”** It is a hypothesis label, not a claim that this exact method is implemented or novel. A reliability phrase should be added only if calibration is independently demonstrated.

If the proposed mechanism does not outperform a direct semantic-rendering baseline, report that outcome and revise the scientific claim. If real geometric errors are primarily caused by inaccessible height or badly aligned pairs, a better architecture may not solve the limiting factor. A rigorous benchmark or analysis contribution is a possible alternative, but the current 24-view single-seed result is far too narrow to support it alone.

The immediate recommendation is therefore **retain VIGOR and the pinned Sat3DGen baseline, complete the review, then test geometry-versus-appearance interventions before investing in a new branch or a server run**. This preserves the work already completed while replacing speculation about why the adapter failed with measurable evidence.

## Sources

[^1]: Qian et al. *Sat3DGen: Comprehensive Street-Level 3D Scene Generation from Single Satellite Image*. ICLR 2026. [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7edc4f114647cea088db4e27ff330d03-Abstract-Conference.html).
[^2]: Qian et al. *Sat3DGen official implementation*, pinned source revision used by this project. [Source code](https://github.com/qianmingduowan/Sat3DGen/blob/70763aff4d9fc5508e67685c906f84d3cebb34fe/source/generator.py). Checkpoint `artifacts/Sat3DGen/config.json` and project `src/satground/backend.py`, `adapter.py`, `semantics.py`, and `losses.py` were the local implementation evidence.
[^3]: Lu et al. *Geometry-Aware Satellite-to-Ground Image Synthesis for Urban Areas*. CVPR 2020. [Official paper record](https://openaccess.thecvf.com/content_CVPR_2020/html/Lu_Geometry-Aware_Satellite-to-Ground_Image_Synthesis_for_Urban_Areas_CVPR_2020_paper.html).
[^4]: *Controllable Satellite-to-Street-View Synthesis with Precise Pose Alignment and Zero-Shot Environmental Control* (ControlS2S). ICLR 2025. [Paper](https://arxiv.org/html/2502.03498).
[^5]: Xu et al. *Satellite to GroundScape—Large-scale Consistent Ground View Generation from Satellite Views*. CVPR 2025. [Official PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Satellite_to_GroundScape_-_Large-scale_Consistent_Ground_View_Generation_from_CVPR_2025_paper.pdf).
[^6]: Ze et al. *SatDreamer360: Multiview-Consistent Generation of Ground-Level Scenes from Satellite Imagery*. ICLR 2026. [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ce33ac9baea433f1b2e5d1333023b947-Abstract-Conference.html).
[^7]: Qian et al. *Sat3DGen*, method and limitations, arXiv version dated 14 May 2026. [Full paper](https://arxiv.org/html/2605.14984).
[^8]: Yao et al. *MagicCity: Geometry-Aware 3D City Generation from Satellite Imagery with Multi-View Consistency*. ICCV 2025. [Official paper record](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_MagicCity_Geometry-Aware_3D_City_Generation_from_Satellite_Imagery_with_Multi-View_ICCV_2025_paper.html).
[^9]: Hua et al. *Sat2City v2: Native 3D City Asset Generation from a Single Satellite Image*. Preprint, revision dated 9 September 2026; publication acceptance of this extension is not established here. [Paper](https://arxiv.org/html/2606.24138v2).
[^10]: Kang et al. *Sat2RealCity: Geometry-Aware and Appearance-Controllable 3D Urban Generation from Satellite Imagery*. Preprint, initially November 2025; latest accessible text inspected. [Paper](https://arxiv.org/html/2511.11470).
[^11]: Jin and Feng. *SAT-SKYLINES: 3D Building Generation from Satellite Imagery and Coarse Geometric Priors*. 3DV 2026 according to the official project. [Project and paper](https://jinzhangyu.github.io/projects/SatSkylines/).
[^12]: Zhi et al. *In-Place Scene Labelling and Understanding with Implicit Scene Representation*. ICCV 2021. [Official PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhi_In-Place_Scene_Labelling_and_Understanding_With_Implicit_Scene_Representation_ICCV_2021_paper.pdf).
[^13]: Fu et al. *Panoptic NeRF: 3D-to-2D Label Transfer for Panoptic Urban Scene Segmentation*. 2022. [Paper](https://arxiv.org/abs/2203.15224).
[^14]: Liu et al. *Semantic Ray: Learning a Generalizable Semantic Field With Cross-Reprojection Attention*. CVPR 2023. [Official paper record](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Semantic_Ray_Learning_a_Generalizable_Semantic_Field_With_Cross-Reprojection_Attention_CVPR_2023_paper.html).
[^15]: Zhang et al. *NeRF++: Analyzing and Improving Neural Radiance Fields*. 2020 technical report. [Paper](https://arxiv.org/abs/2010.07492).
[^16]: *Conditional-Flow NeRF: Accurate 3D Modelling with Reliable Uncertainty Quantification*. 2022. [Paper](https://arxiv.org/abs/2203.10192).
[^17]: Topaloglu et al. *OracleGS: Grounding Generative Priors for Sparse-View Gaussian Splatting*. WACV 2026. [Institutional publication record](https://portal.fis.tum.de/en/publications/41b866b6-1762-4b3e-a697-246a8731cd57).
[^18]: Yang, Zou, and Jepson. *Satellite-to-Street: Synthesizing Post-Disaster Views from Satellite Imagery via Generative Vision Models*. March 2026 preprint. [Paper](https://arxiv.org/abs/2603.20697).
[^19]: CVPR 2026 program chairs. *Reviewer Guidelines*, including novelty/performance and dataset-contribution guidance. [Official guidance](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines).
[^20]: ECCV 2026. *Guidance to Reviewers on Contribution Types*, Algorithms/General. [Official guidance](https://eccv.ecva.net/Conferences/2026/ReviewerContributionTypes).
[^21]: Cheng et al. *Masked-Attention Mask Transformer for Universal Image Segmentation*. CVPR 2022. [Official paper record](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html).
