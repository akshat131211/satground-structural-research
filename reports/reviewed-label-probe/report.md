# One-view reviewed-label diagnostic

Differences from earlier scores come from the reviewed target mask and validity region, not a changed model. One previously inspected view with partial visibility is insufficient for an accuracy claim. The same reviewed masks are applied to both models; all other views retain their original pseudo-label scoring.

Reviewed scoring covers **41.3%** of this image. The remaining pixels are outside the accepted validity mask. The annotation combines a machine proposal, assistant corrections, the user's own drawing, and an explicitly requested extension of the scoring region. This was not a blind annotation study.

| Saved model | Reviewed-view boundary error | Reviewed-view building IoU | Same-view LPIPS |
|---|---:|---:|---:|
| GJ | 0.082974 | 0.806923 | 0.557526 |
| GD | 0.082898 | 0.827901 | 0.558591 |

The 24-view reruns contain one reviewed target and 23 pseudo-labelled targets. They remain descriptive mixed-label outputs, not a human-verified benchmark. No main data-QA flag or server gate was passed.

See [mask review instructions](../../docs/MASK_REVIEW_GUIDE.md) and [aggregate evidence](summary.json).
