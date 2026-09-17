# Additional label review: separate from the fixed experiment

17 September 2026: the first three views in the 18-view batch are complete.
Their user responses, saved masks, decisions and versioned approved indices
remain local. Later progress is recorded in [the review status](../ADDITIONAL_REVIEW_STATUS.json).

- View 1's starting proposal incorrectly classified the overhead bridge as a
  building. The user explicitly accepted a separate building-only correction
  removing 26,197 pixels and preserving the submitted validity mask. Clear bridge
  pixels remain assessable non-building pixels.
- View 2 contains user edits to both masks; all 453 newly added building pixels
  are scored. The submitted version was preserved without assistant changes.
- View 3 adds 994 building pixels, all scored. Its previous pseudo-label target
  had **zero scored building pixels**; the accepted target has **994**. Thus one
  of the two empty-target cases in the checkpoint diagnosis was not actually a
  building-free scene. The original strata described the scoring labels, and
  must not be interpreted as certified scene categories.

The original evaluations and running A1/A2/A3 label index are unchanged. A future
expanded-label evaluation must apply the same new masks to every compared model
and report the label-version change explicitly. This annotation correction is
not a model improvement, new training result, or completed full-data QA.
