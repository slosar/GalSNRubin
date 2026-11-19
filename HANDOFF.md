# Session handoff

## Completed

Implemented the Galactic neutrino-trigger follow-up experiment described by the
workspace task. GalSNRubin calls the independent `galdust` and `ccsnlc` packages.
The baseline contains 1,000 shared candidates for each of RSG, BSG and WR, with
2027 trigger dates. Source, configuration, tests, saved results, exposure audits,
figure PDFs/PNGs, LaTeX source and the compiled report are included.

The saturation correction fixes accounting: complete the first pointing in
which all exposures saturate before stopping, and classify any event with at
least one usable detection as `detected_unsaturated`, even if it later saturates.
Long-exposure saturation alone no longer stops subsequent short exposures.
The cadence remains [1, 1, 15, 15] s with no coaddition. `saturation_s` now marks
the end of the first fully saturated pointing; `immediate_saturation` means
this is the discovery pointing. Individual audit saturation flags are retained.

The preceding change added a configurable topocentric Moon exclusion, defaulting to
10 degrees (`moon_avoidance_deg`; zero disables it). Figure 1 labels the existing
nighttime category as "Not visible during night" and adds "Moon avoidance."
The latter means complete pointings could fit without the lunar veto, but none
fit after it. Partial lunar losses retain the final photometric/search outcome.
The report narrative and generated parameter table describe these definitions.
The updated run is in `outputs/baseline_moon/`; the original no-Moon run remains
in `outputs/baseline/`, with its configuration explicitly pinned to zero radius
and the historical first-saturated-exposure rule. The Moon baseline has been
fully regenerated with the corrected saturation rule.
Galactic success/failure maps and the extinction histogram remain included.
Plot regeneration uses existing results and does not resimulate events.

## Baseline results

| Progenitor | Detected / 1,000 | Median delay among detections | Saturated only / 1,000 | Usable / 1,000 |
|---|---:|---:|---:|---:|
| RSG | 794 | 28.55 hours | 521 | 273 |
| BSG | 802 | 9.77 hours | 485 | 317 |
| WR | 775 | 7.20 hours | 423 | 352 |

Previously the terminal saturated counts were 702, 708, 674, respectively,
including events with earlier usable exposures. Beyond that classification
correction, continuing short exposures recovers usable data for 2 BSG and 22 WR
events. The detection flags, discovery times and discovery magnitudes are
unchanged for all 3,000 events.

Median extinction is Av = 11.33 mag. The outcome maps agree with the event-table
counts, and the histogram contains exactly 1,000 candidates.
The nighttime category contains 148 candidates per type. The Moon reduces
available pointings for 15 candidates, but completely excludes none over three
days, so its exclusive category has zero counts. Two WR detections are lost
to delayed observations outside model validity and count as no valid model
exposure. Pre/post-veto scheduling counts and discovery lunar separations are
saved in the event table.

## Validation and reproduction

Read README.md for installation and full commands; AGENTS.md gives agent guidance.
From the package root, use `.venv/bin/python` in the existing workspace.

```bash
export MPLCONFIGDIR=/tmp/galsnrubin-mpl
export XDG_CACHE_HOME=/tmp/galsnrubin-cache
.venv/bin/python -m pytest tests ../CCLightCurves/tests -q
.venv/bin/python scripts/validate_run.py outputs/baseline_moon
.venv/bin/python -m galsnrubin.products outputs/baseline_moon
```

All 32 tests passed. Full-run accounting, paired candidates, timing and direct
Astropy visibility checks passed; see `outputs/baseline_moon/validation.json`.
Maximum discovery lunar-separation error was 0.00034 degrees (0.02 tolerance).
The report was compiled and its rendered pages were visually checked.
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

Outcome-trend plots now cover distance, Galactic latitude/longitude, Av, month,
localization radius and maximum altitude. They use all triggers in each bin,
show shared candidate counts, and export `figures/outcome_trends.csv`. All bin
counts were checked against the saved events; all 32 tests pass. The report
includes all seven plots. Simulation outputs were reused without changing physics.

## Exit checkpoint (2026-11-19)

All requested outcome plots are implemented in `galsnrubin/products.py` and
included in Section 7 of the rebuilt 17-page report. PDF/PNG plots and the
binned CSV are saved under `outputs/baseline_moon/figures/`. Empty bins stay
blank, and the latitude/altitude axes cover the sampled ranges. These are
marginal distributions, not causal effects; sparse bins have no error bands.
The report was compiled twice and visually inspected, with no overfull boxes.
All 32 tests passed, and all seven binned tables reproduce the event counts.
No simulation rerun is needed to resume. No outstanding requested work remains.
