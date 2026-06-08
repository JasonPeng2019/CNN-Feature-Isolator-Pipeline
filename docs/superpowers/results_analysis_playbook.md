# Results Analysis Playbook — ML Experiment Repo → Report PDF

**Audience:** an autonomous coding agent with filesystem access, a shell, and the ability to run code and compile LaTeX.
**Goal:** read an ML experiments repository, reconcile what the code/results *actually* did against the guide docs, and produce a thorough, sectioned, graph-rich report compiled to PDF (`.tex` → `.pdf`).
**Operating principle:** Trust the code and the produced artifacts over the prose. Guides describe intent; results describe reality. When they disagree, that disagreement *is the finding* — flag it, don't paper over it.

---

## 0. Ground Rules (read before doing anything)

- **Do not fabricate.** Every number, metric, and claim in the report must trace back to a file in the repo or a script you ran. If a value isn't recoverable, write "not recoverable from repo" rather than inventing one.
- **Reproduce, don't transcribe.** Where results are embedded in code (hardcoded, in comments, in notebook outputs), re-derive them from the underlying data/logs when possible. If you can only read the embedded value, label it `[embedded, unverified]`.
- **Preserve provenance.** For each result, record the source file + line/cell so the report is auditable.
- **Idempotent + non-destructive.** Never modify source experiment files. Write all new artifacts to a dedicated `analysis/` output directory.
- **Stop-and-flag, don't guess.** When intent is ambiguous, record the ambiguity in a "Open Questions" log rather than resolving it silently.

---

## 1. Repository Reconnaissance

**Objective:** build a complete map of the repo before analyzing anything.

1. Walk the entire tree. Produce an inventory of every file classified into:
   - **Guide docs** — pre-registration / design docs describing what each experiment was *meant* to test.
   - **Implementation guides** — how experiments were *meant* to be built/run.
   - **Experiment code** — training/eval/data scripts, configs, pipelines.
   - **Results artifacts** — logs, metrics files (`.json`/`.csv`/`.parquet`), checkpoints, notebook outputs, embedded constants, plots.
   - **Configs** — hyperparameters, seeds, splits, env files.
   - **Orchestration** — anything describing stages, parallel runs, sweeps (Makefiles, DAGs, `*.yaml`, schedulers, run scripts).
2. Build an **experiment registry** (`analysis/registry.json`): one entry per distinct experiment, with `id`, `purpose` (from guide), `code_paths`, `config_paths`, `result_paths`, `stage`, and `parallel_group` (which runs are permutations of one another).
3. Identify the **stage/parallelism structure**: which experiments run in which stage, what runs in parallel, and what the permutation axes are (seed, hyperparameter, dataset variant, ablation, etc.). This structure drives the graphs later.

> Output of this phase: `analysis/registry.json` + `analysis/inventory.md` (human-readable map).

---

## 2. Intent Extraction (from the guides)

**Objective:** capture the *designed* experiment before looking at what happened, so the comparison is honest.

For each experiment in the registry, extract from the guide + implementation docs:
- **Hypothesis / purpose** — what question it answers.
- **Designed method** — model, data, splits, metrics, success criteria, expected ranges.
- **Designed procedure** — steps, stages, parallel structure, intended hyperparameters/seeds.
- **Declared success criteria** — what "it worked" was *supposed* to mean.

Record into `analysis/intent.json`. Keep this **separate** from anything derived from code, so the two can be diffed in Section 3. Do not let the guide's claims contaminate your reading of the results.

---

## 3. Reality Extraction & Reconciliation (the core)

**Objective:** determine what actually ran and what it produced, then diff against intent.

### 3a. What actually ran
- Read the experiment code and configs as the source of truth. Extract the *as-implemented* method, hyperparameters, seeds, data splits, and procedure.
- Recover results: parse metrics files/logs; re-derive embedded results from data where feasible; capture notebook output cells. Tag each with provenance and `[verified]` / `[embedded, unverified]`.

### 3b. Implementation-vs-guide diff
For each experiment, compare as-implemented (3a) vs as-designed (Section 2). Produce a structured diff with severity:
- **Match** — implementation follows the guide.
- **Benign deviation** — differs but doesn't threaten validity (e.g., extra logging).
- **Material deviation** — changes what the experiment measures (different metric, split, objective, missing ablation). **Flag prominently.**
Record into `analysis/deviations.json`.

### 3c. Data-leak audit
Actively check for leakage and validity threats; do not assume cleanliness. Inspect for:
- Train/test (or train/val) overlap — shared IDs, duplicated rows, overlapping time windows.
- Target leakage — features derived from the label, or post-outcome information.
- Preprocessing fit on the full dataset before splitting (scalers, encoders, imputers, vocab, feature selection).
- Tuning/early-stopping/model selection on the test set.
- Temporal leakage — future information in past predictions.
- Group leakage — same subject/entity spanning splits.
- Duplicate or near-duplicate samples across splits.
For each experiment, emit a leak-audit verdict (`clean` / `suspected` / `confirmed`) with evidence and the exact file/line. Record into `analysis/leak_audit.json`. A confirmed leak invalidates downstream results — say so explicitly.

### 3d. Outcome classification
Classify each experiment/run against its declared success criteria (Section 2):
- **Success (as hoped)** — met criteria, results desirable.
- **Success (unexpected)** — worked but not as predicted; note the surprise.
- **Undesirable result** — ran cleanly but results are negative/weak.
- **Failed** — errored, didn't complete, or is invalid (e.g., due to a confirmed leak).
- **Inconclusive** — insufficient signal/data.
Record into `analysis/outcomes.json`, each with a one-line justification tied to evidence.

---

## 4. Data Preparation for Graphing

**Objective:** assemble clean, tidy data so plotting is deterministic and covers *every* parallel permutation.

1. Consolidate all recovered metrics into a single tidy table `analysis/results_tidy.csv` with columns at minimum: `experiment_id`, `stage`, `parallel_group`, `permutation_key` (the axis values, e.g. seed/hparam/variant), `metric_name`, `metric_value`, `step_or_epoch` (if time-series), `provenance`, `verification_status`.
2. **Cover all permutations.** Every parallel run must appear — no silent dropping. If a permutation is missing, record it in `analysis/missing_runs.md`.
3. Validate the tidy table (no NaNs where values are expected, consistent metric units, expected row counts per group).

---

## 5. Generate Plotting Scripts (described in English, written as code)

**Objective:** for each needed figure, write a standalone, re-runnable script that reads `results_tidy.csv` and emits a vector figure (`.pdf`/`.pgf`) into `analysis/figures/`.

For each figure, the agent should follow this English specification and then emit the script:

- **Stage progression** — for each experiment, plot the chosen metric across stages/epochs/steps, one line per parallel permutation, so the reader sees how each run evolved. Shade or band across permutations to show spread.
- **Permutation comparison** — bar/box/violin of final metric across the parallel axis (e.g., across seeds or hyperparameter settings) per experiment, so variance and outliers are visible.
- **Cross-experiment summary** — a single comparison figure ranking experiments by their headline metric, color-coded by outcome class from Section 3d.
- **Ablation / sweep surfaces** — where a permutation axis is a swept hyperparameter, plot metric-vs-hyperparameter (line or heatmap for 2-axis sweeps).
- **Leak/validity callout** — where leakage is suspected/confirmed, annotate the affected figures so a reader can't mistake an invalid result for a real one.

Script requirements: deterministic, no network, save vector output, readable axis labels/titles/legends, colorblind-safe palette, and a caption string written to a sidecar so the `.tex` can reuse it. One script may emit several figures; keep them in `analysis/scripts/` and have them run from a single `make_figures` entrypoint.

> Deliverable of this phase: `analysis/scripts/*` + populated `analysis/figures/*`.

---

## 6. Author the Report (`.tex`)

**Objective:** write a thorough, report-style LaTeX document, organized and sectioned, embedding the figures and findings.

Use this section structure:

1. **Title / metadata** — repo name, commit hash, date, agent + run info.
2. **Executive summary** — what was run, headline outcomes, the most important flags (material deviations, confirmed leaks, failures) up top.
3. **Experiment catalog** — table of every experiment: purpose, stage, parallel structure, status.
4. **Per-experiment sections** — for each experiment:
   - Purpose & hypothesis (from intent).
   - As-designed vs as-implemented, with the deviation diff and severity (Section 3b).
   - Data-leak/validity verdict with evidence (Section 3c).
   - Results: figures (Section 5) + tables, with provenance and verification status.
   - Interpretation: what the numbers mean and what they imply.
   - Outcome classification with justification (Section 3d).
5. **Cross-experiment analysis** — comparison figures, patterns across stages and permutations, what the parallel sweeps revealed.
6. **Flags & risks** — consolidated list: material deviations, data leaks, failed/undesirable results, missing runs.
7. **What worked / what we hoped for** — successes against the original success criteria.
8. **What still should be run** — gaps that prevent this from being a fully fleshed-out experiment.
9. **Next steps** — prioritized, concrete follow-ups.
10. **Appendix** — full provenance table, configs, the tidy results table, open questions, environment.

Authoring rules: cite provenance for every nontrivial claim; embed figures as vector graphics with captions from Section 5; never present an embedded-unverified or leak-affected number without its label; keep prose precise and report-toned, not promotional.

---

## 7. Compile to PDF

1. Write the document to `analysis/report.tex`.
2. Compile (e.g., `latexmk -pdf report.tex`, or `pdflatex` run multiple times for cross-refs/TOC, plus a bibliography pass if used).
3. On compile error: read the log, fix the offending LaTeX, recompile. Do not silently drop content to make it compile — fix the cause.
4. Verify the PDF: TOC resolves, all figures render, no `??` references, no missing-file warnings.

> Deliverables: `analysis/report.tex` + `analysis/report.pdf`.

---

## 8. Self-Verification Before Declaring Done

Run this checklist and record pass/fail in `analysis/verification.md`:

- [ ] Every experiment in the registry has a section in the report.
- [ ] Every parallel permutation appears in at least one figure (or is logged as missing).
- [ ] Every reported number has provenance + verification status.
- [ ] Implementation-vs-guide deviations are all surfaced; material ones are prominent.
- [ ] Data-leak audit covered all listed leak types; verdicts have evidence.
- [ ] Failures, undesirable results, and successes are all classified and justified.
- [ ] "What still should be run" and "Next steps" are present and concrete.
- [ ] PDF compiles cleanly with all figures and resolved references.
- [ ] No fabricated values; all ambiguities live in Open Questions.

---

## Output Manifest (everything the agent should produce)

```
analysis/
├── inventory.md            # human-readable repo map
├── registry.json           # experiment registry
├── intent.json             # as-designed (from guides)
├── deviations.json         # implementation-vs-guide diff
├── leak_audit.json         # data-leak verdicts + evidence
├── outcomes.json           # success/fail classification
├── results_tidy.csv        # consolidated tidy results
├── missing_runs.md         # permutations not recovered
├── scripts/                # English-spec'd plotting scripts
├── figures/                # vector figures + caption sidecars
├── report.tex
├── report.pdf
├── verification.md         # final checklist results
└── open_questions.md       # logged ambiguities
```
