"""Check saved run accounting and discovery geometry against direct Astropy.

Run from the project root: .venv/bin/python scripts/validate_run.py outputs/baseline_moon
Writes validation.json into that run directory. Raises on failed invariants.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_body
from astropy.time import Time
from galsnrubin import Config

output = Path(sys.argv[1])
cfg = Config.load(output/'config.json')
x = pd.read_csv(output/'events.csv')
assert len(x) == cfg.n_candidates * len(cfg.progenitors)
assert x.groupby('progenitor').size().eq(cfg.n_candidates).all()
assert set(x.progenitor) == set(cfg.progenitors)
assert x.groupby('candidate_id')[['ra','dec','distance_kpc','Av','trigger_unix']].nunique().eq(1).all().all()
d = x[x.detected]
assert d.first_detection_s.ge(d.first_observation_s).all()
assert d.first_detection_s.le(cfg.followup_days*86400).all()
assert x.immediate_saturation.sum() == ((x.first_detection_s == x.saturation_s) & x.detected).sum()
assert (x.n_usable_exposures > 0).equals(x.usable_photometry)
assert x.outcome.notna().all()
assert (x.n_detected_exposures > 0).equals(x.detected)
night_misses = x[x.outcome == 'no_night_visibility']
assert night_misses.max_alt_deg.gt(cfg.min_alt_deg).all()
assert night_misses.n_night_windows.eq(0).all()
moon_misses = x[x.outcome == 'moon_avoidance']
assert moon_misses.n_night_windows.gt(0).all()
assert moon_misses.n_pointings_without_moon.gt(0).all()
assert moon_misses.n_scheduled_pointings.eq(0).all()
assert x[x.n_scheduled_pointings.gt(0)].n_night_windows.gt(0).all()
location = EarthLocation.from_geodetic(cfg.longitude_deg*u.deg,cfg.latitude_deg*u.deg,cfg.elevation_m*u.m)
t = Time(d.trigger_unix.to_numpy()+d.first_detection_s.to_numpy(),format='unix')
frame = AltAz(obstime=t,location=location,pressure=0*u.hPa)
coord = SkyCoord(ra=d.ra.to_numpy()*u.deg,dec=d.dec.to_numpy()*u.deg)
source = coord.transform_to(frame)
source_alt = source.alt.deg
sun_alt = get_sun(t).transform_to(frame).alt.deg
moon = get_body('moon',t,location,ephemeris='builtin').transform_to(frame)
moon_sep = source.separation(moon).deg
assert np.all(source_alt >= cfg.min_alt_deg - .02)
assert np.all(sun_alt <= cfg.twilight_deg + .02)
assert np.all(moon_sep >= cfg.moon_avoidance_deg - .02)
np.testing.assert_allclose(moon_sep,d.discovery_moon_sep_deg,atol=.02,rtol=0)
check = dict(event_count=len(x),detected_count=len(d),accounting='passed',
    paired_candidates='passed',timing='passed',direct_astropy_visibility='passed',
    astrometry_tolerance_deg=.02,
    moon_avoidance='passed',night_failure_accounting='passed',
    max_moon_separation_error_deg=float(np.max(np.abs(moon_sep-d.discovery_moon_sep_deg))) if len(d) else None,
    max_source_alt_error_deg=float(np.max(np.abs(source_alt-d.discovery_alt_deg))) if len(d) else None,
    max_solar_alt_error_deg=float(np.max(np.abs(sun_alt-d.discovery_sun_alt_deg))) if len(d) else None)
(output/'validation.json').write_text(json.dumps(check,indent=2))
print(json.dumps(check,indent=2))
