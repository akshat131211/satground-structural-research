# Draw your own building and validity masks

## Review the full additional batch in one page

Double-click **Open Additional Review.cmd**. It now opens all 18 images together,
starting at the first unfinished image. The image selector shows every view and
its status. Existing drafts are restored; accepted images display their exact
approved masks read-only, including separately accepted corrections.

1. Review **Building mask** and **Validity mask** for the current image.
2. Click **Save & Next** to save an immutable version of both masks, mark the
   image finished for this batch, and open the next unfinished image.
3. Use **Previous image**, **Next image**, or the **Image** selector to move
   around. These save any changed draft before switching, but do not mark that
   image finished. **Save version** also saves without marking it finished.
4. On the last image, **Save & Finish** saves both masks. If earlier images are
   unfinished, the editor returns to one of them. When every image is done,
   the page shows **All images are finished**. Tell Codex once: **all my masks
   are ready**. No chat message is needed between images.

An edit after finishing an image changes its status back to a draft; use
**Save & Next** again when satisfied. An unfinished polygon must be filled or
cancelled before switching. An entirely empty building mask is allowed; validity
must retain some confidently assessable pixels. New building additions keep the
previously requested scoring rule, and validity can still be edited afterward.

Batch completion markers identify the exact saved masks and remain pending final
human acceptance/verification. They do not change an approved evaluation index,
complete dataset QA, or alter past experiment labels. Previously accepted images
count toward the progress total and need not be redone. Other saved versions,
including submissions made before this update, remain preserved.

One server holds the writer locks for this batch's editable drafts. Reopening
the launcher reuses it. Saved progress survives server restarts. A backup now
also identifies its image, preventing cross-image restoration inside a packet.

## Single-image editor

Double-click **Open Mask Editor.cmd** in the project folder. Keep its terminal
open while editing; closing it stops the local save service. Reopening the tool
restores the latest draft. If the editor is already running, the launcher reuses
it rather than opening a second writer for the same draft.

1. Choose **Building mask** or **Validity mask**.
2. On the right image, use **Paint** to include pixels or **Erase** to exclude
   them. The left image remains an unmodified reference.
3. For larger regions, choose a polygon tool, click its corners, and use **Fill
   polygon** or Enter. Escape cancels an unfinished polygon.
4. Use **Area**, **Zoom**, and **Pan** to reach small details. Brush sizes
   always refer to original image pixels. Hold O to see the original; B/E/H
   select paint/erase/pan when the drawing canvas is focused.
5. **Undo** and **Redo** restore both mask states. Drafts save after completed
   edits. Wait for the saved status before closing the tab.
6. Click **Save version** when ready, then tell Codex that the masks are ready.
   **Download PNGs** optionally exports the saved version as a ZIP. If you edit
   again, save a new version before downloading.

Building white pixels mean visible building surfaces; foliage, cars, people and
road are excluded. Validity white pixels mean confidently assessable pixels;
uncertain, hidden and dynamic regions are excluded. The page states the current scoring rule. By default the masks are independent.
With the optional building-additions rule enabled, each newly painted or polygon-filled
building pixel is also scored. Erasing a building label does not erase validity: a
clear static non-building object can still be scored. You can edit validity afterward
to exclude a pixel explicitly. Undo/redo includes the automatic scoring change. The counter below the images reports how many of your newly added building
pixels are scored and how many validity excludes. This is feedback only; it does
not change either mask. An uncertain gap may remain unscored.

The default launcher starts from proposal 7 for the first guided view. A review
session opened for another sample shows its view number in the page heading.
It records changes relative to that source. All stored masks are exact-size, single-channel PNGs
containing only 0 and 255; brush and polygon operations do not introduce blended
labels. Zoom affects the display, not the mask dimensions.

## Saving and review

Drafts and versions stay under `data/review/manual-edits/`, grouped by sample and
source-packet hash. Saving a version creates exact target/mask files, comparison
cards, provenance and **pending** review decisions. Original proposals and earlier
saved versions remain unchanged. Saving alone does not approve either mask or
complete the main dataset QA. The existing [mask review workflow](MASK_REVIEW_GUIDE.md)
handles final acceptance and evaluation.

If saving fails, **Backup and recovery** can download an editable JSON backup.
Restore it only into the same source-image session. A conflicting draft from
another window is rejected rather than silently overwriting that window's work.

The service listens only on the laptop's loopback interface, serves this one
sample, and checks the source hashes, request origin, session token and draft
revision before accepting a save. There is no image upload to an external service.

## Other packets and verification

```powershell
& .\.venv\python.exe scripts/edit_masks.py --packet data/review/mask-first3-v7 --open-browser
# With a multi-sample packet, also pass --sample-id SAMPLE_ID.
& .\.venv\python.exe -m pytest tests/test_editor.py -q
node tests/editor_core_test.cjs
```

The tests cover exact PNG export, resumability, immutable versions, wrong-source
and stale-draft rejection, local request restrictions, binary brush/polygon
operations, and undo/redo across both masks. UI smoke-test drawings are kept in a
separate ignored runs directory and never counted as user annotations.

## Automatically score new building pixels

The user requested this preference on 16 September 2026. New sessions read
`score_building_additions` from the local, untracked
`data/review/editor-policy.json`. The heading explains whether it is enabled.
Existing sessions retain their startup policy; finish/save and restart their
server to change it. The launcher warns if a reused session has a different policy.

Use `--score-building-additions` or `--no-score-building-additions` for an explicit
session override. The rule applies to new brush/polygon additions, preserves
existing drafts and backups, and does not infer human acceptance. Later manual
validity exclusions remain possible; painting elsewhere does not re-score a
previously excluded building pixel unless it is erased and newly painted again.
