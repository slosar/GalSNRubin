"""All simulation settings; units appear in names or field metadata."""
from dataclasses import dataclass, field, asdict
import json
import numpy as np


def setting(value, description):
    return field(default=value, metadata={"description": description})


@dataclass(frozen=True)
class Config:
    n_candidates: int = setting(1000, "Candidates per progenitor, shared draws")
    seed: int = setting(2027, "Random seed")
    year: int = setting(2027, "Uniform UTC trigger year")
    progenitors: tuple = setting(("RSG", "BSG", "WR"), "Canonical ccsnlc progenitors")
    band: str = setting("z", "LSST band")
    latitude_deg: float = setting(-30.244633, "Site latitude [deg]")
    longitude_deg: float = setting(-70.749417, "East-positive site longitude [deg]")
    elevation_m: float = setting(2647., "Site elevation [m]")
    min_alt_deg: float = setting(15., "Minimum geometric source altitude [deg]")
    twilight_deg: float = setting(-18., "Solar altitude at astronomical twilight [deg]")
    calibration_s: float = setting(600., "Delay after evening twilight [s]")
    filter_change_s: float = setting(90., "Initial setup after first eligible instant [s]")
    followup_days: float = setting(3., "Follow-up deadline after trigger [days]")
    visibility_step_s: float = setting(60., "Conservative visibility sampling interval [s]")
    ephemeris_step_s: float = setting(300., "Solar altitude and Earth angle grid [s]")
    theta25_deg: float = setting(30., "Search cap radius at reference distance [deg]")
    reference_distance_kpc: float = setting(25., "Localization reference distance [kpc]")
    fov_deg2: float = setting(9.6, "Effective field of view [square deg]")
    exposures_s: tuple = setting((1., 1., 15., 15.), "Ordered exposure durations per pointing [s]")
    dead_time_s: float = setting(15., "Dead time after every exposure, includes slewing [s]")
    integration_samples: int = setting(5, "Midpoint flux quadrature samples per exposure")
    snr_threshold: float = setting(5., "Single-exposure detection SNR")
    zeropoints: tuple = setting((26.52, 28.51, 28.36, 28.17, 27.78, 26.82), "One electron/s AB zero points, ugrizy")
    sky_mags: tuple = setting((23.05, 22.25, 21.2, 20.46, 19.61, 18.6), "Dark sky AB mag per square arcsec, ugrizy")
    seeing_arcsec: tuple = setting((.92, .87, .83, .80, .78, .76), "Effective zenith PSF FWHM [arcsec], ugrizy")
    saturation15_mags: tuple = setting((14.7, 15.7, 15.8, 15.8, 15.3, 13.9), "Fiducial 15 s saturation AB magnitudes, ugrizy")
    pixel_scale_arcsec: float = setting(.2, "Pixel scale [arcsec]")
    read_noise_e: float = setting(8.8, "Read noise per pixel per exposure [electrons]")
    dark_current_e_s: float = setting(.2, "Dark current per pixel [electrons/s]")
    Rv: float = setting(3.1, "CCM extinction parameter")
    T_min_eV: float = setting(1., "ccsnlc lower temperature validity cutoff [eV]")
    lightcurve_ngrid: int = setting(1200, "ccsnlc internal time samples")
    audit_candidates: int = setting(3, "Save full exposure audits for first N candidates per type")

    def __post_init__(self):
        positive = ['n_candidates', 'followup_days', 'visibility_step_s', 'ephemeris_step_s',
                    'theta25_deg', 'reference_distance_kpc', 'fov_deg2', 'integration_samples',
                    'snr_threshold', 'pixel_scale_arcsec', 'Rv', 'T_min_eV', 'lightcurve_ngrid']
        for key in positive:
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ['calibration_s', 'filter_change_s', 'dead_time_s', 'read_noise_e', 'dark_current_e_s', 'audit_candidates']:
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        for key in ['n_candidates', 'seed', 'year', 'integration_samples', 'lightcurve_ngrid', 'audit_candidates']:
            if not isinstance(getattr(self, key), int):
                raise ValueError(f'{key} must be an integer')
        if self.seed < 0 or not 1900 <= self.year <= 2100:
            raise ValueError('seed must be nonnegative and year in 1900..2100')
        if self.band not in 'ugrizy' or len(self.band) != 1:
            raise ValueError('band must be one of ugrizy')
        if not self.progenitors or len(set(self.progenitors)) != len(self.progenitors) or set(self.progenitors) - {'RSG','BSG','WR'}:
            raise ValueError('unique canonical progenitors required')
        if not 0 <= self.min_alt_deg < 90 or not -90 < self.twilight_deg < 0:
            raise ValueError('invalid altitude thresholds')
        if not -90 <= self.latitude_deg <= 90 or not -180 <= self.longitude_deg <= 180:
            raise ValueError('invalid observatory coordinates')
        if not np.isfinite(self.elevation_m):
            raise ValueError('nonfinite elevation')
        if not self.exposures_s or np.any(np.asarray(self.exposures_s) <= 0) or not np.all(np.isfinite(self.exposures_s)):
            raise ValueError('exposures must be finite and positive')
        for key in ['zeropoints','sky_mags','seeing_arcsec','saturation15_mags']:
            if len(getattr(self,key)) != 6 or not np.all(np.isfinite(getattr(self,key))):
                raise ValueError(f'{key} must have six finite values')
        if min(self.seeing_arcsec) <= 0:
            raise ValueError('seeing must be positive')

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))
