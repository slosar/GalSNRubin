"""Paired-candidate Monte Carlo and explicit per-exposure outcome accounting."""
import hashlib
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from ccsnlc import LightCurveModel
import galdust
from .visibility import Calendar, pointing_starts, altitude
from .photometry import limits, exposure_magnitudes


def localization(distance_kpc, config):
    """Spherical cap radius, area [deg^2], Bernoulli hit probability.

    Radius saturates at 180 degrees. Independent pointings sample with replacement;
    this is the requested area-ratio model, not a non-overlapping tiling algorithm.
    """
    theta = np.minimum(180.,config.theta25_deg*(np.asarray(distance_kpc)/config.reference_distance_kpc)**2)
    area = 4*np.pi*np.sin(np.deg2rad(theta)/2)**2*(180/np.pi)**2
    return theta, area, np.minimum(1., config.fov_deg2/np.maximum(area,np.finfo(float).tiny))


def sample_catalog(path, config):
    """Load trusted local pandas pickle; sample rows with replacement uniformly.

    ra/dec are ICRS degrees; mu0 is true distance modulus. Input row position
    is retained because catalogue index labels are not unique. Catalogue av and
    initial masses do not set the dust or canonical progenitor models.
    """
    catalog = pd.read_pickle(path)
    required = ['ra','dec','mu0']
    if len(catalog) == 0 or not set(required).issubset(catalog):
        raise ValueError('catalogue must contain nonempty ra, dec, mu0 columns')
    values = catalog[required].to_numpy(float)
    if not np.all(np.isfinite(values)) or np.any(np.abs(values[:,1]) > 90) or np.any((values[:,0]<0)|(values[:,0]>=360)):
        raise ValueError('catalogue has invalid coordinates or distance moduli')
    rng = np.random.default_rng(config.seed)
    ids = rng.integers(len(catalog),size=config.n_candidates)
    frame = catalog.iloc[ids][required].reset_index(drop=True).copy()
    frame['catalog_row'] = ids
    frame['distance_kpc'] = 10**((frame.mu0.to_numpy()-10)/5)
    if not np.all(np.isfinite(frame.distance_kpc)) or np.any(frame.distance_kpc <= 0):
        raise ValueError('invalid catalogue distances')
    start,end = [Time(f'{year}-01-01T00:00:00').unix for year in [config.year,config.year+1]]
    frame['trigger_unix'] = rng.uniform(start,end,config.n_candidates)
    frame['Av'] = galdust.av(frame.ra.to_numpy(),frame.dec.to_numpy(),frame.distance_kpc.to_numpy())
    if not np.all(np.isfinite(frame.Av)) or np.any(frame.Av < 0):
        raise ValueError('dust model returned invalid extinction')
    frame['theta_deg'],frame['area_deg2'],frame['p_hit'] = localization(frame.distance_kpc.to_numpy(),config)
    return frame


def observe(model, starts, probability, rng, config):
    """Search until first single-exposure detection, then track until saturation.

    A saturated exposure counts as a localization with unusable ordinary PSF
    photometry. This optimistic assumption is explicit in the report. Every
    exposure in a pointing shares one FOV draw. No model extrapolation occurs.
    """
    mags = exposure_magnitudes(model,starts,config)
    depth,satlimit = limits(config)
    detectable = np.isfinite(mags) & (mags <= depth)
    saturated = np.isfinite(mags) & (mags <= satlimit)
    hits = rng.random(len(starts)) < probability
    found = np.argwhere(detectable & hits[:,None])
    first = tuple(found[0]) if len(found) else None
    in_field = np.broadcast_to(hits[:,None],starts.shape).copy()
    if first is not None:
        in_field[first[0]:] = True
    detections = detectable & in_field
    saturation = saturated & in_field
    sat_indices = np.argwhere(saturation)
    stop = tuple(sat_indices[0]) if first is not None and len(sat_indices) else None
    used = np.ones(starts.shape,dtype=bool)
    if stop is not None:
        used.ravel()[np.ravel_multi_index(stop,starts.shape)+1:] = False
    usable = detections & ~saturated & used
    valid = np.isfinite(mags)
    reason = ('detected_saturated' if stop is not None else 'detected_unsaturated') if first is not None else (
        'no_valid_model_exposure' if not valid.any() else
        'too_faint' if not detectable.any() else 'search_miss')
    duration = np.asarray(config.exposures_s)
    summary = dict(outcome=reason, detected=first is not None,
        first_detection_s=float(starts[first]+duration[first[1]]) if first is not None else np.nan,
        discovery_mag=float(mags[first]) if first is not None else np.nan,
        discovery_exposure_s=float(duration[first[1]]) if first is not None else np.nan,
        immediate_saturation=bool(saturated[first]) if first is not None else False,
        saturation_s=float(starts[stop]+duration[stop[1]]) if stop is not None else np.nan,
        usable_photometry=bool(usable.any()),
        n_pointings=int(np.any(used,axis=1).sum()),
        n_hits=int((hits & np.any(used,axis=1)).sum()),
        n_exposures=int(used.sum()), n_valid_exposures=int((valid & used).sum()),
        n_detectable_exposures=int((detectable & used).sum()),
        n_detected_exposures=int((detections & used).sum()),
        n_usable_exposures=int(usable.sum()),
        model_expired_by_deadline=bool(model.delay+model.t_end < config.followup_days*86400))
    audit = dict(start_s=starts[used], exposure_s=np.broadcast_to(duration,starts.shape)[used],
                 magnitude=mags[used],in_field=in_field[used],detected=detections[used],
                 saturated=saturation[used],pointing=np.broadcast_to(np.arange(len(starts))[:,None],starts.shape)[used])
    return summary,audit


def run(catalog_path, output_dir, config):
    """Run all progenitors, write reproducible event records and exposure audits."""
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    config.save(output/'config.json')
    candidates = sample_catalog(catalog_path,config)
    candidates.to_csv(output/'candidates.csv',index=False)
    print('Building offline solar/visibility calendar...',flush=True)
    calendar = Calendar(config)
    coords = SkyCoord(candidates.ra.to_numpy()*u.deg,candidates.dec.to_numpy()*u.deg)
    depth,satlimit = limits(config)
    rows,audits = [],[]
    for event, candidate in candidates.iterrows():
        windows,maxalt,angles = calendar.windows(coords[event],candidate.trigger_unix)
        starts = pointing_starts(windows,candidate.trigger_unix,config)
        for prog in config.progenitors:
            row = dict(candidate_id=event,progenitor=prog,**candidate.to_dict(),
                       trigger_utc=Time(candidate.trigger_unix,format='unix').isot,
                       max_alt_deg=maxalt, n_windows=len(windows),
                       first_observation_s=float(starts[0,0]) if len(starts) else np.nan,
                       detected=False,first_detection_s=np.nan,discovery_mag=np.nan,
                       discovery_exposure_s=np.nan,immediate_saturation=False,
                       saturation_s=np.nan,usable_photometry=False,n_pointings=0,n_hits=0,
                       n_exposures=0,n_valid_exposures=0,n_detectable_exposures=0,
                       n_detected_exposures=0,n_usable_exposures=0,model_expired_by_deadline=False,
                       discovery_alt_deg=np.nan,discovery_sun_alt_deg=np.nan)
            if maxalt <= 0:
                row['outcome'] = 'never_above_horizon'
            elif maxalt <= config.min_alt_deg:
                row['outcome'] = 'never_above_min_alt'
            elif not windows:
                row['outcome'] = 'no_night_visibility'
            elif not len(starts):
                row['outcome'] = 'no_complete_pointing'
            else:
                model = LightCurveModel(prog,candidate.distance_kpc,candidate.Av,
                         bands=(config.band,),Rv=config.Rv,T_min_eV=config.T_min_eV,
                         ngrid=config.lightcurve_ngrid)
                # Same event seed across types couples search uniforms, reducing comparison noise.
                rng = np.random.default_rng(np.random.SeedSequence([config.seed,int(event),1]))
                summary,audit = observe(model,starts,candidate.p_hit,rng,config)
                row.update(summary)
                if summary['detected']:
                    t = candidate.trigger_unix+summary['first_detection_s']
                    row['discovery_alt_deg'] = float(altitude(*angles,np.interp(t,calendar.grid,calendar.era),config.latitude_deg))
                    row['discovery_sun_alt_deg'] = float(np.interp(t,calendar.grid,calendar.sun))
                if event < config.audit_candidates:
                    a = pd.DataFrame(audit)
                    a['candidate_id'],a['progenitor'] = event,prog
                    audits.append(a)
            rows.append(row)
        if (event+1)%100 == 0:
            print(f'{event+1}/{len(candidates)} paired candidates complete',flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(output/'events.csv',index=False)
    if audits:
        pd.concat(audits).to_csv(output/'exposure_audit.csv.gz',index=False)
    summary = {}
    for prog, group in results.groupby('progenitor',sort=False):
        detected = group[group.detected]
        summary[prog] = dict(n=len(group),detected=int(group.detected.sum()),
             immediate_saturation=int(group.immediate_saturation.sum()),
             usable_photometry=int(group.usable_photometry.sum()),
             median_detection_hours=float(detected.first_detection_s.median()/3600) if len(detected) else None,
             median_discovery_mag=float(detected.discovery_mag.median()) if len(detected) else None,
             outcomes={str(k):int(v) for k,v in group.outcome.value_counts().items()})
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    metadata = dict(catalog=Path(catalog_path).name,
         catalog_sha256=hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest(),
         versions={p:version(p) for p in ['numpy','pandas','astropy','ccsnlc','galdust','matplotlib']},
         magnitude_limits=depth.tolist(),saturation_limits=satlimit.tolist(),
         calendar='Bundled offline IERS; geometric altitude; frozen CIRS per trigger')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return results
