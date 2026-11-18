# GalSNRubin

A reproducible, idealized forecast of Rubin follow-up of Galactic supernova
neutrino triggers. Calls `galdust.av` and `ccsnlc.LightCurveModel` directly.
The baseline samples **1,000 candidates per progenitor** (RSG, BSG, WR; 3,000
simulated events) using shared positions, distances, dates and search uniforms.

The [compiled report](report/report.pdf) includes the baseline results, observing
assumptions, parameter table, Galactic sky maps, and extinction histogram.
The [session handoff](HANDOFF.md) records the validated state and known limitations.

## Set up a fresh workspace

Clone the three independent packages side by side before following the install
commands below:

```bash
mkdir GalacticSN
cd GalacticSN
git clone https://github.com/slosar/GalSNRubin.git
git clone https://github.com/slosar/CCLightCurves.git
git clone https://github.com/slosar/galdust.git DustModel/galdust-repo
```

Place the input `TRILEGAL_noLMCSMC_10plusZAMS.pkl` in this workspace directory.
The original catalogue is an external input and is not bundled in this repository;
its SHA256 is recorded in `outputs/baseline_moon/metadata.json`. The saved baseline
candidate/event tables are bundled, so regenerating figures and inspecting the
results do not require the original catalogue. Building the report needs a LaTeX
installation providing `pdflatex` and the packages listed in `report/report.tex`.

## Install and run

Python >=3.10, NumPy >=2 (required by ccsnlc and the supplied pickle). From the
GalacticSN workspace root:

```bash
python -m venv GalSNRubin/.venv
GalSNRubin/.venv/bin/python -m pip install -e './CCLightCurves[test]' -e ./DustModel/galdust-repo -e './GalSNRubin[test]'
cd GalSNRubin
export MPLCONFIGDIR=/tmp/galsnrubin-mpl
export XDG_CACHE_HOME=/tmp/galsnrubin-cache
.venv/bin/python -m pytest tests ../CCLightCurves/tests -q
.venv/bin/python -m galsnrubin.cli --catalog ../TRILEGAL_noLMCSMC_10plusZAMS.pkl --output outputs/baseline_moon
.venv/bin/python scripts/validate_run.py outputs/baseline_moon
cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

`galsnrubin` is an equivalent CLI after installation. The module command also
works from the project root. A small smoke run uses `--n 6 --output outputs/smoke`.
The full baseline typically takes a few minutes; Astropy's calendar is computed
once. All astronomy is offline. For 2027, bundled IERS polar motion can run out;
Astropy then uses mean polar motion (arcsecond-scale uncertainty), which is small
compared with the 60-second visibility sampling. No downloads occur during runs.

All settings are listed in `config/default.json` and in the report. Override any
subset with `--config my-settings.json`; `--n` overrides the file's candidate count.
For example `{"band":"i", "min_alt_deg":25, "followup_days":5,
"moon_avoidance_deg":20}`. The Moon exclusion radius defaults to 10 degrees;
set `moon_avoidance_deg` to 0 to disable it. Use a separate
output directory for each experiment; rerunning into an existing directory
replaces that run's products. The report's `\runpath` points to the baseline;
change it to compile an alternate run.

## Assumptions that define the forecast

* Uniform sampling with replacement from the trusted local pandas pickle, with
  `distance_kpc = 10**((mu0 - 10)/5)`. Coordinates are ICRS degrees. This is a
  catalogue-weighted experiment, not a calibrated Galactic supernova rate model.
  Pickles execute Python during loading: use only a trusted catalogue.
* The supplied catalogue's `av` and `m_ini` are not used. GNILC-rescaled `galdust`
  supplies Av; ccsnlc uses its canonical RSG/BSG/WR models with deterministic delay,
  blackbody spectrum, light-travel smearing, CCM extinction and a 1 eV cutoff.
* The localization angle is a **radius**, not diameter or Gaussian sigma:
  `theta = min(180 deg, 30 deg * (d/25 kpc)**2)`; spherical-cap area is used.
  Pointings are independent draws **with replacement** at `min(1, FoV/area)`.
  No actual cap centre, tiling, exclusion of searched fields, probability-map
  contour level or confidence-tail loss is simulated.
* Source visibility gates the search using the true source coordinates. This
  implements the requested idealized algorithm; it is not a scheduler which
  discovers which part of a neutrino probability map is observable.
* Night starts 600 s after evening Sun=-18 deg and ends at morning Sun=-18 deg.
  Source altitude is geometric, with no refraction. Source CIRS coordinates are
  frozen at the trigger, with interpolated Earth rotation and solar altitude.
  The topocentric source--Moon separation must be at least the configured
  `moon_avoidance_deg` (10 degrees by default), independent of lunar phase.
  Astropy's builtin lunar ephemeris is computed offline on the same 300 s grid.
  Visibility edges round inward by <=60 s; sub-step windows or lunar crossings
  may be unresolved. The cut gates the true source, like the altitude cut;
  individual random field centres are not generated.
  Full pointing blocks must fit, including final dead time. Initial filter setup
  costs 90 s after the first eligible instant; it is charged once per event.
* Each pointing has ordered exposures `[1,1,15,15]` s, each followed by 15 s dead
  time, hence a 92 s pointing block. This dead time includes repositioning.
* Exposures integrate flux with five midpoint samples. The cutoff is unknown
  rather than zero flux; exposure intervals crossing it are invalid. Pre-breakout
  light is zero. A detection is one exposure at or above 5 sigma, no coadding or
  multi-exposure confirmation. Time to discovery is the exposure end, with no
  processing/alert latency. After detection, subsequent pointings track the source.
* Complete the first pointing in which **all exposures saturate**, then stop.
  A saturated long exposure does not prevent subsequent short exposures. Any
  usable detection, before or after a saturated exposure, makes the event an
  unsaturated success. Only detections with no usable photometry count as
  saturated-only. The two 15 s exposures are separate, not coadded into 30 s;
  the same rule supports configurations with individual 30 s exposures.
  Saturated sources count as localized
  even if no unsaturated exposure exists: this assumes recognisable saturated
  images and **does not establish real pipeline recovery performance**.
* Sensitivity uses the SMTN-002 count/noise prescription, with fixed dark sky and
  zenith seeing, read noise and dark current. No airmass penalty, moonlight sky
  brightening, weather, crowding, blending, templates, normal survey scheduling
  or telescope contention.
  The 1 s mode and saturation thresholds are assumptions, not verified operating
  capabilities. Saturation values are historical Science Book fiducials at 15 s.
  These omissions are especially consequential near 15-degree altitude.

## Products and outcome definitions

The figures include Galactic Mollweide HEALPix count maps (`sky_success` and
`sky_failures`) comparing the three progenitors. Failure maps separate no night
visibility, too faint, and search miss, with a shared scale across all panels.
`av_distribution` counts the 1,000 shared candidates once each in 2 mag bins.
Maps use nside=32, Galactic centre at the centre, longitude increasing left,
and grey for empty pixels. They show event counts, not detection efficiencies.
Underlying map and histogram counts are saved in `figures/sky_extinction_counts.npz`.
To regenerate plots and tables from saved events without rerunning the simulation:

```bash
.venv/bin/python -m galsnrubin.products outputs/baseline_moon
```

`outputs/baseline_moon/` contains `candidates.csv`, `events.csv`, `summary.json`,
`config.json`, `metadata.json` (catalogue SHA256 and dependency versions), `validation.json` (full-run checks against direct Astropy),
`exposure_audit.csv.gz` (first three candidates for every progenitor), figure PDFs
and PNGs, and generated LaTeX parameter/results tables. NaN means unavailable;
pre-breakout exposure magnitudes are +infinity. Times ending in `_s` are seconds
since the neutrino trigger; `trigger_unix` is UTC Unix seconds; magnitudes are AB.
Discovery brightness is the underlying flux-averaged model magnitude, even for
saturated exposures; it is not a measurable unsaturated PSF magnitude.

Miss outcomes are mutually exclusive and assigned in this order:

1. `never_above_horizon`: maximum geometric altitude <=0 degrees.
2. `never_above_min_alt`: above horizon but never above the configured minimum.
3. `no_night_visibility` (Figure 1: "Not visible during night"): the source
   exceeds the minimum altitude at transit, but no simultaneous night/source
   window exists before the deadline, evaluated before the Moon veto.
4. `moon_avoidance`: complete pointings fit those night/altitude windows without
   the Moon veto, but none fit after applying it (including setup and dead time).
5. `no_complete_pointing`: windows exist but intrinsically cannot fit setup plus
   a full block, even without the Moon veto.
6. `no_valid_model_exposure`: all exposures before breakout or beyond model cutoff.
7. `too_faint`: some valid model exposures, but none meet depth even if on target.
8. `search_miss`: detectable opportunities existed, but random search missed them.

Partial lunar losses can delay discovery or reduce the available search, but
events with remaining pointings retain their final photometric/search outcome.
`n_night_windows`, `n_pointings_without_moon`, and `n_scheduled_pointings` retain
the pre-veto windows and before/after scheduling counts; `n_windows` counts
Moon-safe windows. `discovery_moon_sep_deg` records the discovery separation.

Detections are `detected_unsaturated` if **any** detected exposure is usable,
even when longer or later exposures saturate. Otherwise they are
`detected_saturated` (saturated-only). A source jumping from undetectable to
saturated in all exposure modes without an observed usable interval belongs to
the latter category. If short exposures are too faint but the following long
exposures saturate, subsequent pointings still attempt short exposures.
`model_expired_by_deadline` flags expiry for
modelled events and is false for events rejected before modelling. It is not an
additional exclusive miss reason. No prediction of later recombination or
radioactive emission is made. `saturation_s` is the end of the first **fully
saturated pointing**; `immediate_saturation` means this is the discovery pointing.
These are stopping diagnostics, separate from the event classification.
`usable_photometry` means at least one detected unsaturated exposure;
it can be true even when a later pointing fully saturates. `n_hits` counts successful
random search draws within used pointing blocks, not tracked-field observations.
Repeated exposure rows share a pointing's FOV result. Audits keep individual
exposure saturation flags and stop after a fully saturated pointing, if any.

The original run without Moon avoidance remains in `outputs/baseline/`; its
configuration explicitly records `moon_avoidance_deg=0`. The updated report
and reproduction commands use `outputs/baseline_moon/`, regenerated with the
usable-photometry classification and fully saturated pointing stop rule. The
original no-Moon run is historical and uses the old first-saturated-exposure rule.

## Scientific inputs

* Local `CCLightCurves/README.md` and `DustModel/galdust-repo/README.md` document
  the light-curve physics and the GNILC-rescaled smooth dust model.
* Rubin site coordinates: https://rubinobservatory.org/for-scientists/rubin-101
* Camera field: https://rubinobservatory.org/for-scientists/rubin-101/instruments
* SNR, zero points, dark sky, noise and seeing: https://smtn-002.lsst.io/
* Historical saturation fiducials: https://www.lsst.org/sites/default/files/docs/sciencebook/SB_Whole.pdf

## Layout

`config.py`: validated settings. `visibility.py`: geometry/calendar/cadence.
`photometry.py`: exposure integration and instrument limits. `simulation.py`:
catalogue, search state and results. `products.py`: figures and tables.
`tests/`: numerical and behavioural checks. `report/`: LaTeX source and PDF.
