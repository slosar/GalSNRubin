# Session handoff

## Completed

Implemented the Galactic neutrino-trigger follow-up experiment described by the
workspace task. GalSNRubin calls the independent `galdust` and `ccsnlc` packages.
The baseline contains 1,000 shared candidates for each of RSG, BSG and WR, with
2027 trigger dates. Source, configuration, tests, saved results, exposure audits,
figure PDFs/PNGs, LaTeX source and the compiled report are included.

The latest additions are Galactic Mollweide success maps, failure maps for no
night visibility / too faint / search miss, and an extinction histogram counting
the 1,000 shared candidates once. All are included in the report. Plot regeneration
uses existing results and does not resimulate events.

## Baseline results

| Progenitor | Detected / 1,000 | Median delay among detections | Immediately saturated / 1,000 |
|---|---:|---:|---:|
| RSG | 794 | 28.52 hours | 520 |
| BSG | 802 | 9.74 hours | 486 |
| WR | 777 | 7.16 hours | 446 |

Median extinction is Av = 11.33 mag. The outcome maps agree with the event-table
counts, and the histogram contains exactly 1,000 candidates.

## Validation and reproduction

Read README.md for installation and full commands; AGENTS.md gives agent guidance.
From the package root, use `.venv/bin/python` in the existing workspace.

```bash
export MPLCONFIGDIR=/tmp/galsnrubin-mpl
export XDG_CACHE_HOME=/tmp/galsnrubin-cache
.venv/bin/python -m pytest tests ../CCLightCurves/tests -q
.venv/bin/python scripts/validate_run.py outputs/baseline
.venv/bin/python -m galsnrubin.products outputs/baseline
```

All 20 tests passed. Full-run accounting, paired candidates, timing and direct
Astropy visibility checks passed; see `outputs/baseline/validation.json`.
The 10-page report compiled and its rendered pages were visually checked.
Compile `report/report.tex` from the `report/` directory twice after plot changes.

## Inputs and limitations

The original TRILEGAL pickle remains an external workspace input, outside this
repository. Its name and SHA256 are in run metadata. The sampled candidate table
is included. Neither sibling package was modified. The virtual environment,
caches, build metadata, smoke outputs and LaTeX auxiliary files are ignored.

The forecast assumes independent area-ratio searches, a radius interpretation
of localization uncertainty, ideal weather and dark-sky/zenith sensitivity,
and localization of saturated detections. Source visibility is evaluated at the
true position, not over a trigger probability map. Light curves are not extended
beyond the existing model's temperature cutoff. Future IERS polar motion may
fall back to a long-term mean and emits an expected Astropy warning. These
limitations are explained in README.md and the report.

There is no outstanding implementation request at handoff. The next session can
start from the saved results without repeating the Monte Carlo run.
