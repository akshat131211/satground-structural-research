# Related work and overlap matrix

This focused matrix carries forward the accepted research plan and official reference links. It is not an exhaustive literature search or a verified novelty claim. Recheck current releases and exact input settings before writing a paper or a direct leaderboard.

The [15 September research-direction assessment](RESEARCH_DIRECTION_ASSESSMENT.md) adds a targeted novelty audit against newer city generators and semantic neural rendering. It recommends a geometry/appearance diagnostic before testing a proposed building-identity constraint. These proposed extensions are not implemented or validated results.

| Work | Relevance and overlap | Source |
|---|---|---|
| BEV-to-street-view survey (2024) | Terminology and broader cross-view taxonomy | [arXiv](https://arxiv.org/abs/2405.08961) |
| Sat2Density (ICCV 2023) | Density learning from paired satellite and ground views | [Project](https://sat2density.github.io/) |
| Geospecific View Generation (ECCV 2024) | Geometry and texture guidance; geometry alone is not novel | [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06376.pdf) |
| Sat2Scene (CVPR 2024) | Satellite-conditioned diffusion for urban 3D scenes | [Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/04/2401.10786.pdf) |
| Geometry-guided Cross-view Diffusion | Geometric conditioning and multiple plausible ground views | [Paper](https://arxiv.org/abs/2412.03315) |
| ControlS2S (ICLR 2025) | Pose alignment, environmental controls and input noise | [Paper](https://arxiv.org/abs/2502.03498) |
| Satellite to GroundScape (CVPR 2025) | Multi-satellite geometry and neighboring ground-view consistency; different input setting from our first pilot | [User-supplied paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Satellite_to_GroundScape_-_Large-scale_Consistent_Ground_View_Generation_from_CVPR_2025_paper.pdf), [project](https://gdaosu.github.io/sat2groundscape/) |
| MagicCity (ICCV 2025) | Geometry-aware multi-view generation and Gaussian reconstruction | [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Yao_MagicCity_Geometry-Aware_3D_City_Generation_from_Satellite_Imagery_with_Multi-View_ICCV_2025_paper.pdf) |
| Sat2City (ICCV 2025) | Single-image city geometry and appearance generation already studied | [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Hua_Sat2City_3D_City_Generation_from_A_Single_Satellite_Image_with_ICCV_2025_paper.pdf) |
| Sat2Density++ (TPAMI 2026) | Potential released comparator; match sky/style and pose settings | [Official repository](https://github.com/qianmingduowan/Sat2Densitypp) |
| SatDreamer360 (ICLR 2026) | Multi-view panoramic generation with shared geometry | [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ce33ac9baea433f1b2e5d1333023b947-Abstract-Conference.html) |
| Sat3DGen (ICLR 2026) | Starting checkpoint; shared scene representation and geometry-informed training already present | [Official code](https://github.com/qianmingduowan/Sat3DGen), [method](https://arxiv.org/abs/2605.14984) |
| Skyfall-GS | Multi-view satellite reconstruction and diffusion refinement; later geometry track | [Paper](https://arxiv.org/abs/2510.15869) |
| GS-Voxel (2026 preprint) | Structured large aerial Gaussian generation | [Paper](https://arxiv.org/abs/2608.17988) |
| Cross-View Splatter | Uses ground photographs alongside overhead inputs; separate input regime | [Paper](https://arxiv.org/abs/2605.19656) |
| Conditional-Flow NeRF | General uncertainty in neural rendering predates this project | [Paper](https://arxiv.org/abs/2203.10192) |
| Sat2City v2 (September 2026 preprint revision) | Real satellite-mesh adaptation and geometry-anchored appearance; broad geometry/appearance separation overlaps | [Paper](https://arxiv.org/html/2606.24138v2) |
| Sat2RealCity (preprint) | Individual building generation with OSM structural priors and controlled appearance | [Paper](https://arxiv.org/html/2511.11470) |
| Sat-Skylines (3DV 2026, official project listing) | Satellite plus coarse geometric priors for detailed buildings | [Project](https://jinzhangyu.github.io/projects/SatSkylines/) |
| Semantic-NeRF (ICCV 2021) | Joint semantic, geometry, and appearance fields; semantic rendering alone is established | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhi_In-Place_Scene_Labelling_and_Understanding_With_Implicit_Scene_Representation_ICCV_2021_paper.pdf) |
| Panoptic NeRF (2022) | Semantic constraints improve geometry with consistent instance labels | [Paper](https://arxiv.org/abs/2203.15224) |
| Semantic Ray (CVPR 2023) | Generalizable semantic fields and cross-reprojection attention | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Semantic_Ray_Learning_a_Generalizable_Semantic_Field_With_Cross-Reprojection_Attention_CVPR_2023_paper.html) |
| NeRF++ (2020 technical report) | Shape-radiance ambiguity is established general neural-rendering prior art | [Paper](https://arxiv.org/abs/2010.07492) |
| OracleGS (WACV 2026) | Validates generative priors using multi-view evidence and uncertainty | [Institutional record](https://portal.fis.tum.de/en/publications/41b866b6-1762-4b3e-a697-246a8731cd57) |

The candidate gap is an adequately controlled building-specific structural accuracy and calibrated reliability study under a single-overhead-image constraint. An adapter or extra loss alone is not sufficient novelty. A3 must beat matched A1, explain its failure cases, and withstand independent segmentation and manual review.
