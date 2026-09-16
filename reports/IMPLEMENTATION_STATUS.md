# Implementation status — updated 16 September 2026

The laptop research infrastructure now runs on real VIGOR inputs. The selected pilot has been imported and decoded. A1 and A3 each completed 100 optimizer steps, and all three A0/A1/A3 variants were evaluated on the same 24 validation views. The preliminary comparison does not show a meaningful structural gain. No controlled three-seed improvement, calibrated reliability result, or server gate has been established.

## Guided annotation and camera audit

- Added a local drawing editor with paint/erase brushes, polygon inclusion/exclusion, zoom/pan, independent building and validity masks, undo/redo, resumable automatic drafts, immutable saved versions, and ZIP export of binary PNGs. A one-click Windows launcher reuses the active editor for the same draft. Its source targets/proposals remain unchanged; saving creates pending decisions rather than approval.
- Browser smoke testing used a separate ignored output directory: one-pixel painting, undo/redo, saving, and reload recovery were exercised. Saved PNGs were checked for exact 256 x 256 size, L mode and 0/255 values; the original target and separate validity mask remained unchanged. These automated test drawings are not user annotations. Storage/API tests and JavaScript drawing-operation tests passed; at that editor milestone, the Python suite reported **50 passed, two optional CUDA tests skipped**.

- Saved three actual human ground-view pairing observations, preserving a suspected mismatch and explicit uncertainty. These are not certified roof/facade correspondences.
- Checked all six source records against the pinned training metadata. Their perspective crops and rotation matrices match the release exactly, including the flagged pair. This verifies implementation consistency; independent heading, capture date, scale, and building identity remain unresolved.
- Prepared three target-only building/validity mask proposals. The user rejected view 1's initial building mask. A larger segmentation input and explicit pixel corrections were saved as separate proposal versions; no earlier labels or experiment outputs were overwritten.
- **Two complete reviewed mask pairs have now been exported and evaluated.** For view 1, following the earlier assisted corrections, the user painted 413 additional building pixels and explicitly requested that those additions be scored. A separate accepted packet extends validity by exactly 411 pixels; the other two were already valid. The user's submitted building mask, target image, and original saved version remain unchanged. The accepted scoring region covers 27,095 of 65,536 pixels (41.3%). Raw responses, exact masks, and verification records remain local.
- The initial one-view diagnostic re-evaluated the same saved GJ and GD predictions with one reviewed target and the other 23 original pseudo-label targets. All other views' metrics and all image-quality metrics were verified unchanged. On the reviewed view, GJ/GD building IoU is 0.806923/0.827901 and boundary error is 0.082974/0.082898. Changed scores versus the original evaluation reflect changed labels and scoring regions, not changed models. This single previously inspected view does not establish an accuracy gain or pass full data QA. See the [separate diagnostic](reviewed-label-probe/report.md).
- The editor now identifies the review view, provides general image-region presets, and reports how many newly painted building pixels the validity mask actually scores. This feedback does not alter either mask. View 2 is now complete: the user preserved the 1,231-pixel building correction and manually revised validity (233 pixels added, 735 removed). Both submitted masks and cards are preserved exactly; the assistant truck-exclusion proposal was not applied. View 3 is open for manual review.
- The [two-view diagnostic](reviewed-labels-2/report.md) re-scores all 24 saved predictions per model, with two reviewed targets and 22 pseudo-labelled targets. View 2's scored fraction is 90.3%; GJ/GD boundary error is 0.061869/0.062389 and IoU is 0.585436/0.588302. Compared with the earlier human1 evaluations, every metric for view 1 and all other 22 views is unchanged. These descriptive results do not establish a model improvement. The original and one-view reports are preserved.
- Added a reusable cohort report that rejects altered predictions, inconsistent view sets, mismatched reviewed labels, and unrelated metric changes. It records per-view coverage and excludes private sample identifiers from the published report. Matching evaluator source versions were used for the final paired reruns.
- Added a region-preview command that verifies source hashes and identical targets, enlarges exact scoring pixels, and highlights additions/removals. It does not modify masks or write acceptance. Inspected the original panorama crop as an annotation aid; the 256-pixel scoring target remains unchanged. Unresolvable mixed foliage pixels remain outside the proposed scoring region.
- Added exact target/mask/card checksums, separate component decisions, versioned pixel edits, and a reviewed-index exporter. Changes invalidate the affected approval; unchanged component acceptance can be retained only after exact artifact equality checks. Evaluation rejects altered masks, target mismatches, unrelated sample indices, and mismatched reviewed labels between comparisons.
- Annotation records distinguish machine proposals and assistant corrections from human decisions. These development views were previously shown with generated results, so the guided review is not blind. Original A/G experiment scores remain unchanged.
- Latest checks: **19 focused review-reporting/annotation tests passed**. The complete Python suite passed **62 tests with two optional CUDA tests skipped**. The renderer is unchanged by this reporting update; earlier JavaScript editor checks passed.

See [the mask review guide](../docs/MASK_REVIEW_GUIDE.md) and [aggregate review progress](REVIEW_PROGRESS.json). More GPU training is not the next dependency; the current labels and correspondence assumptions need review first.

## Completed geometry control on the laptop

- Implemented density/appearance interventions with frozen-A0 importance samples and a separate density-only training mode. The original A0/A1/A3 configuration paths retain their renderer.
- Diagnosed both completed A1/A3 adapters on all 24 fixed validation views. Shared baselines matched the original chunked baseline exactly. Appearance-only interventions preserved opacity/radial maps exactly; density-only and joint interventions shared identical geometric maps.
- Native versus shared-proposal adapted renders differed, so hybrid comparisons explicitly use one shared sampling convention. Radial maps are model-coordinate diagnostics, not measured geometric accuracy.
- New GJ and GD runs each completed 100 steps on the same 100 training views, with 4,448 parameters, seed 17, accumulation eight, 256-pixel RGB, and identical A3 image-space losses. GD completed a 10-step stop/resume check. Initial losses were identical; all logged losses/gradients and final adapter/optimizer tensors were finite.
- GJ: boundary error **0.203285**, IoU **0.532346**, LPIPS **0.566210**. GD: boundary **0.203049**, IoU **0.535316**, LPIPS **0.566611**.
- GD versus GJ: **0.116%** relative boundary reduction, paired geographic-bootstrap absolute-improvement CI **[-0.000998, 0.001481]**, IoU **+0.297 percentage points**, LPIPS **+0.071% relative**. The interval spans zero. **Density routing alone has not demonstrated a meaningful gain in this short probe.**
- GD loop time: **682.4 seconds**; GJ: **721.8 seconds**. Both peaked at **668.1 MiB** allocated CUDA memory with existing feature caches. These exclude initialization and separate desktop/driver allocations; no statistical speed claim is made.
- All three maximum-boundary-penalty cases remain in both evaluations. Their contribution is about 61% of the A0 group-weighted boundary error; no view was excluded to improve a score.
- Prepared a training-only candidate identity-review packet: **24 tiles, 48 distinct panorama positions, 192 perspective views, 24 training groups**, eight tiles per city. Every exported image's dimensions/hash and every preserved manifest row were verified. Empty identity/annotation records are not training-eligible. Guided human pairing observations are kept locally and separately from certified masks and full data QA.
- Validation: **38 CPU/protocol tests passed, two opt-in CUDA tests skipped in that run**; both CUDA integration tests were separately run successfully on this laptop. Dependency consistency passed. The baseline/intervention CUDA check retains 48 importance samples and verifies checkpointed versus direct gradients.

The [full geometry report](geometry-pilot/report.md), [protocol](../docs/GEOMETRY_PILOT.md), and [codebase explanation](../docs/CODEBASE_GUIDE.md) distinguish this completed control from the unimplemented building-identity method. The next research step needs reviewed correspondences and direct semantic-rendering supervision/control; larger GPUs are not justified by these results.

## Completed preliminary adaptation check

- A1 and A3 each trained a 4,448-parameter adapter for 100 steps, with gradient accumulation of eight, 100 fixed training views, seed 17, and 256 x 256 outputs.
- A1 was stopped at step 10 and resumed through step 100. Both final checkpoints and every logged optimizer step have finite values; the first-step RGB/LPIPS losses match exactly before any update.
- A1: boundary error 0.200953, building IoU 0.540302, LPIPS 0.561346. A3: boundary error 0.200765, building IoU 0.531960, LPIPS 0.565367.
- A3 versus A1 boundary reduction: 0.09%; the paired geographic-bootstrap 95% interval for the absolute improvement is [-0.003246, 0.004538]. The interval spans zero. IoU fell 0.83 percentage points and LPIPS worsened 0.72% relative.
- Three views receive the maximum boundary penalty in every variant. One involves tiny uncertain mask fragments; these views remain in all primary scores and need human-mask review.
- A1 training-loop time was 719.2 seconds with 1,971.8 MiB peak allocated VRAM. A3 took 605.2 seconds with 668.1 MiB after reusing the feature cache. These are not controlled speed or memory comparisons; initialization and driver allocations are excluded.
- A 200-view review packet (100 train, 100 validation; 82 geographic groups) is prepared locally. All target crop dimensions and hashes were verified. Twenty contact sheets, a pairing checklist, and a pending annotation index were prepared. At that milestone no human masks were certified; the later guided review above has completed two pairs.
- Verification: 32 tests passed, one opt-in GPU test skipped; dependency consistency passed. The GPU equality test was previously completed on 13 September and was not rerun for these reporting-only additions.

See [the full preliminary comparison](learning100/report.md), [machine-readable measurements](learning100/summary.json), and [review instructions](../docs/REVIEW_GUIDE.md). The local first-six gallery preserves manifest order and is not selected by metric performance.

## Real-data milestone on 15 September 2026

- Six training-city RGB archives passed full gzip EOF/CRC checks and received SHA-256 records.
- All 3,000 satellite tiles and 5,622 panoramas required by the existing pilot were found and decoded: zero missing, invalid, or exact cross-split duplicates.
- Two dHash near-duplicate candidate pairs are pending review; these are not confirmed duplicates.
- All 3,747 training panorama sky masks were retrieved from the pinned supplement. Original downloads remain in `dataset/`; the selected files are in `data/vigor/`. Neither directory is tracked by Git.
- Seattle archive member names were inspected to identify the city; its imagery was not extracted, decoded, or used.
- A fixed preliminary subset has 100 training views and 24 validation views from 10 validation geographic groups. Selection preceded predictions and metrics.
- The 24-view frozen baseline finished at 256 x 256: 1,949.8 MiB peak PyTorch allocation and 36.2 seconds for the generation loop, including cache misses but excluding model startup. These timings are specific to this small run.
- Baseline group-weighted pseudo-label metrics: building IoU 0.5316, normalized boundary error 0.2040, LPIPS 0.5659, SSIM 0.1566. The independent B1 evaluation segmenter differs from B0 training supervision; that original baseline evaluation used no human-verified masks. This small adaptation holdout may have been seen by the pretrained model.
- Illumination preprocessing now matches the pinned 512 x 128 RGB/mask resize before averaging training histograms; the reference equivalence test passes.
- Protocol, archive integrity/path handling, and illumination checks: 28 tests passed.

See [machine-readable ingest and baseline evidence](DATA_INGEST_STATUS.json). Full manifests, archive/member hashes, predictions, and per-view results remain local under ignored data/run directories.

## Verified locally on 13 September 2026

- Pinned Sat3DGen source and safetensors checkpoint downloaded and loaded on native Windows.
- NVIDIA RTX 4050 Laptop GPU; PyTorch 2.6.0+cu124 reports CUDA available.
- Zero-initialized adapter reproduces the frozen baseline exactly (maximum RGB difference 0).
- Full RGB/LPIPS/building/boundary objective passed synthetic 128- and 256-pixel tests.
- 256-pixel test: 10 optimizer steps with accumulation 8, then a resumed step; finite losses and gradients, frozen base parameters.
- Peak allocated VRAM: 1904.0 MiB; peak PyTorch reserved memory: 2220.0 MiB. These exclude the desktop/driver's separate allocations.
- Trainable adapter parameters: 4,448; frozen model parameters: 381,022,223.
- Median warm synthetic optimizer-step time: 3.78 seconds. This is not a real-dataset or server throughput estimate; frozen features were already cached in memory.
- Protocol and CUDA reference-renderer checks: `24 passed in 9.95s`.
- Dependency consistency check passed. Real-data training rejects missing original images before creating a run.
- Public metadata split preparation produced 22,488 fixed-view records from 3,000 selected overhead tiles: 2,000 train, 250 validation, 250 calibration, 500 audit. RGB validation reports 8,622 missing files. Seattle imagery was not opened.

The controlled native-renderer equality check disables importance resampling to avoid differences in random-number request ordering; normal profiles retain the checkpoint's 48 coarse and 48 importance samples. Full-image super-resolution is applied after ray chunks are stitched.

## What remains

1. Review the prepared development packet, real image scale/footprints, the two near-duplicate candidates, camera alignment, and any available date evidence; annotate development building boundaries.
2. Run matched main A1/A2/A3 experiments with seeds 17/29/43 and validation-only selection after the required review. The 100-step result alone neither validates nor conclusively rejects the hypothesis.
3. Freeze the audit, complete independent/manual evaluation, and assess structural and reliability gates.
4. Start server, external-geometry, and 3D phases only if the preceding gates pass.

Current server decision: **insufficient evidence**. The 13 September GPU profiles are synthetic diagnostics. The new 100-step comparison uses real images, but the main controlled adaptation study and human review remain incomplete. Optional KID has not been run on a real dataset. Human annotations have not been fabricated.

## Implementation corrections during verification

The panorama pitch sign was aligned with the pinned release and checked against reference crops. Temporary test files were moved into the workspace to avoid a pre-existing Windows shared-temp permission problem. NVIDIA's pinned SegFormer release uses legacy weight files, loaded with PyTorch's restricted weights-only loader; no safetensors file was assumed. The unmodified upstream code and checksum-pinned assets remain separate from this project's source.

See [the runbook](../docs/RUNBOOK.md), [data access](../docs/DATA_ACCESS.md), [verification records](verification.json), and [the main resource profile](profile256-accum8.json).
