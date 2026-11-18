# Agent instructions

Read README.md before changing the model. This is a separate package that calls
ccsnlc and galdust; do not modify either sibling merely to add integration logic.
Use .venv/bin/python from this directory. All parameter values belong in Config,
with units and descriptions, and must propagate to config/default.json and the
report table. Keep outputs from distinct configurations in separate directories.

Run `.venv/bin/python -m pytest tests ../CCLightCurves/tests -q`. After physics or
scheduler changes, run the full baseline command in README, regenerate figures,
and compile report/report.tex twice. Inspect report layout. Do not extrapolate
ccsnlc beyond its temperature cutoff or silently substitute catalogue extinction.
Retain the radius interpretation, paired candidates, independent pointing draws,
and saturated-localization caveat unless explicitly changing those assumptions.
Never commit .venv, caches or LaTeX auxiliary files. Keep scientific baseline data,
figures, report source and report PDF available for review. No network is needed
for simulation; installation needs package access.
