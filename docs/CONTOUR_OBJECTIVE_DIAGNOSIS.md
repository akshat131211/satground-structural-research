# Contour objective diagnosis after the eight-view fitting test

The user authorized the next diagnosis and a focused correction supported by its
evidence. The completed 200-step fitting result remains unchanged and negative
under its predeclared criterion. Continue the original single-image VIGOR and
Sat3DGen structural-fidelity question; no height, dataset or server change.

Baseline commit: `748d271ca4d6c353e15f2010156b509a976e4ad1`.
The new trial branch preserves all previous scripts, protocols, masks and results.
The frozen diagnostic specification is
`configs/contour-objective-diagnosis/protocol.json`.

## Questions to distinguish

- Does the existing boundary penalty prefer displaced outlines or missing edges
  over correctly located outlines in controlled cases?
- How much of its actual gradient comes from positive target contours and from
  locations whose target contour is zero?
- Does that gradient reinforce or oppose the combined RGB, perceptual and region
  objective at the same scene parameters?

Sparse support and crop-perimeter effects motivate these questions. They do not
by themselves establish gradient suppression, harmful conflicts or a successful
alternative loss. The earlier 100-view pseudo-label probe concerns different
labels and checkpoints; it cannot answer the reviewed eight-view question.

## CPU controlled cases

Fix synthetic probability maps, building targets and validity masks before
executing the cases. Include increasing outline displacement, missing and extra
buildings, erosion/dilation, confidence changes without moving geometry, crop
perimeters, sparse support and ignored pixels. Record raw region and boundary
losses and their weighted contributions separately. Use finite differences for
hard translations; only differentiate continuous transformations of soft maps.

These cases test the mathematics of the loss, not generated-image accuracy or
real sensor errors. Document undesirable behavior as a property to investigate,
not as a failed implementation test. Preserve failed diagnostic attempts.

## Twenty-four zero-update gradient checks

Use exactly the frozen eight training views in `data/review/reviewed-fitting8-v1`,
at zero-output A0 initialization and the final A2/A3 adapters from
`runs/reviewed-fitting200-seed17`. Keep the original training-derived illumination,
resolution, camera transforms, renderer and accepted masks. The per-view seed is
101 plus its zero-based frozen manifest index, shared across all three states.
This is a diagnostic seed policy; it does not replay historical training draws.

At every state measure the common objective with weights RGB 1, LPIPS 0.1,
region 0.5 and boundary 0.5. The label "A2 objective" means RGB+LPIPS+region at
that state's same parameters, not a different checkpoint. In particular boundary
gradients at final A2 are measured with the common diagnostic weight, rather than
being zeroed by A2's historical training configuration.

Split the current boundary error into positive-target-contour and zero-target-
contour locations on its actual eroded validity support. Both contributions use
the original full-support denominator, so their values and gradients must sum to
the original boundary term. This decomposition changes neither training nor labels.
Measure rendered-RGB and all 4,448 adapter-parameter gradients, component norms,
cosines and support counts. Keep zero norms explicit and undefined ratios
unavailable. Verify component sums against the unchanged Objective implementation.

Use float32 with TF32 disabled for this diagnostic only. Image-gradient relative
error must not exceed 5e-5 and adapter-gradient error 2e-4. Do not silently relax
these tolerances to obtain a pass. Check frozen reference modules, unchanged
adapter tensors, exact checkpoint and input hashes and zero optimizer updates.
Raw gradients precede AdamW, clipping and moment scaling; they do not reconstruct
the historical optimizer's effective update.

The first of the fixed 24 checks may run alone as a startup check, then an exact
compatible resume continues the remaining 23. Hold one GPU lock, verify no other
runner or GPU child is live, and preserve partial outputs on error. Recovery may
skip only exact completed records under an unchanged identity. This is not
training and must instantiate no optimizer.

## Decision after diagnosis

If controlled cases identify a concrete defect, define one correction that fixes
the defect without rewarding missing/extra buildings or treating ignored pixels
as known background. Test numerical gradients, support, empty cases and validity
handling before any adaptation. Gradient magnitude alone cannot select a weight
or prove the correction will help.

Only then freeze a separate matched, fresh 200-step A2/candidate pair, one seed,
with the same reviewed cohort, final-step scoring and frozen A0 reference. State
its metric conventions, quality limits and fitting criterion before launch.
No sweep, extra seed or longer budget is implied by this diagnosis. Any completed
result remains training-example fitting evidence with incomplete independent QA.

Seattle, audit, calibration and the 21 development image/mask pairs remain outside
these checks. User-approved masks are preserved, including unresolved QA concerns.
Publish only code, documentation and aggregate evidence; raw imagery, labels,
sample identifiers, coordinates, reviewer information and responses stay local.
