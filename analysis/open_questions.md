# Open questions

- The repo contains historical report text and stale-report material that embeds some chain-variant values (for example, the Family I re-mask chain). Those values were not re-derived from a dedicated standalone artifact in this pass and should be treated as embedded, not independently verified.
- The amount by which repeated test-set monitoring may overstate broader held-out generalization cannot be quantified from the surviving repo state alone.
- The pareto launcher metadata contains one inconsistent rc value even though the corresponding result artifact exists.
- The current analysis focuses on surviving artifacts and code paths. It does not reconstruct any missing scheduler context outside the JSON launcher summaries under `LISP-Setup/LISP-3-Setup/logs/`.
