"""Check saved run accounting and discovery geometry against direct Astropy.

Run from the project root: .venv/bin/python scripts/validate_run.py outputs/baseline
Writes validation.json into that run directory. Raises on failed invariants.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
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
location = EarthLocation.from_geodetic(cfg.longitude_deg*u.deg,cfg.latitude_deg*u.deg,cfg.elevation_m*u.m)
t = Time(d.trigger_unix.to_numpy()+d.first_detection_s.to_numpy(),format='unix')
frame = AltAz(obstime=t,location=location,pressure=0*u.hPa)
coord = SkyCoord(ra=d.ra.to_numpy()*u.deg,dec=d.dec.to_numpy()*u.deg)
source_alt = coord.transform_to(frame).alt.deg
sun_alt = get_sun(t).transform_to(frame).alt.deg
assert np.all(source_alt >= cfg.min_alt_deg - .02)
assert np.all(sun_alt <= cfg.twilight_deg + .02)
check = dict(event_count=len(x),detected_count=len(d),accounting='passed',
    paired_candidates='passed',timing='passed',direct_astropy_visibility='passed',
    astrometry_tolerance_deg=.02,
    max_source_alt_error_deg=float(np.max(np.abs(source_alt-d.discovery_alt_deg))) if len(d) else None,
    max_solar_alt_error_deg=float(np.max(np.abs(sun_alt-d.discovery_sun_alt_deg))) if len(d) else None)
(output/'validation.json').write_text(json.dumps(check,indent=2))
print(json.dumps(check,indent=2))
