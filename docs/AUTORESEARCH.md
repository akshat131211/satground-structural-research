# Applying autoresearch to the satellite study

The user requested Karpathy's repository and Eric's guide on 19 September 2026.
The original repository is imported under ignored `external/autoresearch`,
detached at `228791fb499afffb54b46200aca536f79142f117`. Its six principal source
and dependency files are pinned by Git-blob SHA256 in
`configs/autoresearch/upstream.json`. No upstream dependencies or training data
were installed. No upstream Python training code was executed.

## What the references contribute

[Karpathy's autoresearch](https://github.com/karpathy/autoresearch/tree/228791fb499afffb54b46200aca536f79142f117)
is an agent-driven language-model experiment: the agent edits `train.py`, keeps
data preparation/evaluation fixed, trains for a fixed five-minute training-time
budget, measures validation bits per byte, records results, and retains useful
changes. `program.md` supplies agent instructions; it is not an autonomous service
or a model that performs research by itself. The reference is tested on H100.

[Eric's guide](https://ericslab.notion.site/Autoresearch-Guide-3dfb2014b38380ab8efdf6170c2022ac)
recommends a fresh branch per trial from the current baseline, retaining
unsuccessful branches and merging improvements. It also recommends reading and
steering the reports. We adopt preserved trial branches instead of the upstream
example's reset-based discard loop. The guide was read in the browser after
the web text fetch could not retrieve it.

## Adaptation to this project

| Reference workflow | SatGround implementation or requirement |
|---|---|
| Agent instructions | Root `program.md`, scoped to the existing question |
| Imported source | Pinned, unchanged local reference; separate environment |
| One editable training script | A narrow, frozen allowlist per future trial; core changes require a diagnosed reason |
| Fixed evaluator | Existing cameras, masks, segmenter, metrics, style and manifests stay pinned |
| Validation bits per byte | Building boundary error with IoU, LPIPS, empty-target errors and geographic intervals |
| Five-minute trials | Fixed matched optimizer budgets for the causal comparison; runtime recorded separately |
| Keep/discard | Preserve every branch/result; classify supported, inconclusive, regressed or failed |
| Repeated automatic trials | One bounded reviewed-data fitting pair next; no indefinite search |
| Results log | Verified local JSON plus TSV, with aggregate publication only |

Equal wall-clock time and equal updates answer different questions. A fixed
time budget can compare efficiency on one device, but changing a loss changes
the number of updates completed within that time. Our current hypothesis needs
matched updates. Recorded 500-step training times were 3,441.5 seconds for A2
and 3,923.4 seconds for A3; the five-minute default is not a validated budget for
this task. These historical timings are not a throughput guarantee for new data.

The upstream lock selects Torch 2.9.1/CUDA 12.8 and NumPy >=2.2.6; this project
uses its existing resolved environment and requires NumPy <2. Its attention
kernels and language-model data path also differ. We imported the workflow,
not those packages or a new dataset.

## Executable commands

From the project root, use the existing environment:

```powershell
.venv/python.exe scripts/autoresearch.py status
.venv/python.exe scripts/autoresearch.py import-contour --output runs/autoresearch/contour-baseline-v1
.venv/python.exe scripts/autoresearch.py verify-ledger --session runs/autoresearch/contour-baseline-v1
```

`status` checks the clean pinned reference and actual local review submissions.
Completed forms still require verification of their exact evidence; this command
does not assign training eligibility. It does not inspect sealed images.

`import-contour` is deliberately specific to the existing completed comparison.
It reruns its CPU completion audit, verifies agreement with preserved evidence,
checks the aggregate arithmetic and identities, and creates `record.json`,
`results.tsv`, `session.json` and the fresh audit. It refuses existing output.
It runs zero inference or optimizer updates. Failed/partial outputs are retained.
`verify-ledger` checks the artifact hashes, original source hashes, audit equality,
record arithmetic and TSV reproduction. These integrity checks are not proof
of ground-truth label quality.

The ledger classification summarizes the existing outcome. It is not a new
preregistered statistical test or a revision to the completed protocol. A
geographic interval spanning zero remains inconclusive even when the mean falls.
IoU loss over 0.01, relative LPIPS deterioration over 5%, or increased empty-target
false-positive fraction is flagged as a quality regression. An exploratory signal
never automatically promotes a model, merges code or passes the server gate.

To restore the reference on another checkout:

```powershell
git clone https://github.com/karpathy/autoresearch.git external/autoresearch
git -C external/autoresearch checkout --detach 228791fb499afffb54b46200aca536f79142f117
```

## Current stopping point and next trial

The separate 12-view training review is pending. The existing 21 human-reviewed
development masks remain evaluation-only. Review all new views through
`Open Training Review.cmd`, then verify the exact saved masks, scene choices,
target hashes and accepted subset. Uncertain cases remain recorded and ineligible
unless their uncertainties are resolved. No replacement scenes are selected.

After that verification, specify and implement the previously proposed fresh
200-step A2/A3 fitting pair on eligible training examples, seed 17 and learning
rate 0.00003. Fix its success criterion before launch; it tests learnability on
those examples, not held-out accuracy. The fitting runner/configurations are
not yet implemented. This integration is the agent protocol and evidence layer,
not a claim that an autonomous GPU search is running.

No new seed, learning-rate sweep, longer budget, dataset, height track or server
allocation follows from importing this reference. Repeatedly optimizing the
already inspected development set would increase selection bias. Seattle, audit
and calibration remain sealed. Human review and a later independent evaluation
cannot be replaced by a keep/discard loop.
