"""Offline astrometry and sampled night/altitude/Moon observing windows."""
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, get_sun, get_body, CIRS
from astropy.time import Time
from astropy.utils import iers

iers.conf.auto_download = False
iers.conf.auto_max_age = None


def maximum_altitude(dec_deg, latitude_deg):
    return 90. - np.abs(np.asarray(dec_deg)-latitude_deg)


def altitude(ra_rad, dec_rad, era_rad, latitude_deg):
    """Geometric altitude using CIRS RA and local Earth rotation angle."""
    lat = np.deg2rad(latitude_deg)
    return np.rad2deg(np.arcsin(np.clip(np.sin(lat)*np.sin(dec_rad) +
           np.cos(lat)*np.cos(dec_rad)*np.cos(era_rad-ra_rad), -1, 1)))


class Calendar:
    """Interpolated Sun, topocentric Moon and Earth rotation over the year.

    CIRS source coordinates are frozen at the trigger over the few-day window.
    Atmospheric refraction is disabled. Future UT1 uses bundled IERS predictions.
    """
    def __init__(self, config):
        self.config = config
        self.location = EarthLocation.from_geodetic(config.longitude_deg*u.deg,
                            config.latitude_deg*u.deg, config.elevation_m*u.m)
        start = Time(f'{config.year}-01-01T00:00:00').unix
        self.year_end = Time(f'{config.year+1}-01-01T00:00:00').unix
        self.year_start = start
        self.grid = np.arange(start-86400, self.year_end+(config.followup_days+1)*86400,
                              config.ephemeris_step_s)
        times = Time(self.grid, format='unix')
        self.sun = get_sun(times).transform_to(AltAz(obstime=times, location=self.location,
                                                   pressure=0*u.hPa)).alt.deg
        self.era = np.unwrap(times.earth_rotation_angle(longitude=self.location.lon).rad)
        # Explicit builtin ephemeris keeps lunar positions offline. Use the same
        # observer-centred CIRS frame as the source, including lunar parallax.
        moon = get_body('moon', times, self.location, ephemeris='builtin').transform_to(
            CIRS(obstime=times, location=self.location))
        self.moon_ra = np.unwrap(moon.ra.rad)
        self.moon_dec = moon.dec.rad
        # Linear twilight crossings; calibrated nights may span UTC midnight.
        dark = self.sun < config.twilight_deg
        changes = np.flatnonzero(dark[1:] != dark[:-1])
        crossings = self.grid[changes] + (config.twilight_deg-self.sun[changes]) / (
                    self.sun[changes+1]-self.sun[changes])*config.ephemeris_step_s
        dusk = list(crossings[~dark[changes]])
        dawn = list(crossings[dark[changes]])
        if dark[0]:
            dusk.insert(0, self.grid[0])
        if dark[-1]:
            dawn.append(self.grid[-1])
        self.nights = [(a+config.calibration_s,b) for a,b in zip(dusk,dawn)
                       if a+config.calibration_s < b]

    def moon_separation(self, ra_rad, dec_rad, times):
        """Source--Moon separation [deg] at Unix times, using CIRS angles [rad]."""
        moon_ra = np.interp(times, self.grid, self.moon_ra)
        moon_dec = np.interp(times, self.grid, self.moon_dec)
        # Haversine form remains accurate for near-coincident directions.
        hav = (np.sin((dec_rad-moon_dec)/2)**2 +
               np.cos(dec_rad)*np.cos(moon_dec)*np.sin((ra_rad-moon_ra)/2)**2)
        return np.rad2deg(2*np.arcsin(np.sqrt(np.clip(hav, 0, 1))))

    def windows(self, coord, trigger, *, avoid_moon=True):
        """Return conservative intervals [unix s], max altitude and CIRS angles.

        Boundaries are rounded inward by at most visibility_step_s. Windows
        shorter than the sampling step can be missed (documented resolution).
        Set avoid_moon=False to retain night/altitude windows for failure
        accounting. The angular Moon veto is independent of lunar phase.
        """
        cfg = self.config
        cirs = coord.transform_to(CIRS(obstime=Time(trigger,format='unix'),location=self.location))
        ra, dec = cirs.ra.rad, cirs.dec.rad
        maxalt = float(maximum_altitude(cirs.dec.deg, cfg.latitude_deg))
        windows = []
        for a,b in self.nights:
            a,b = max(a,trigger), min(b,trigger+cfg.followup_days*86400)
            if a >= b:
                continue
            t = np.unique(np.r_[np.arange(a,b,cfg.visibility_step_s), b])
            alt = altitude(ra, dec, np.interp(t,self.grid,self.era), cfg.latitude_deg)
            good = alt >= cfg.min_alt_deg
            if avoid_moon and cfg.moon_avoidance_deg > 0:
                good &= self.moon_separation(ra, dec, t) >= cfg.moon_avoidance_deg
            edges = np.diff(np.r_[False,good,False].astype(int))
            for lo,hi in zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1)-1):
                if t[hi] > t[lo]:
                    windows.append((t[lo],t[hi]))
        return windows, maxalt, (ra,dec)


def pointing_starts(windows, trigger, config):
    """Fit complete pointing blocks in windows, including all exposure dead time.

    Initial setup starts at first eligible instant and can finish during a gap.
    No further setup is charged after the initial filter change.
    """
    if not windows:
        return np.empty((0,len(config.exposures_s)))
    durations = np.asarray(config.exposures_s)+config.dead_time_s
    offsets = np.r_[0.,np.cumsum(durations)[:-1]]
    ready = windows[0][0]+config.filter_change_s
    starts = []
    for a,b in windows:
        a = max(a,ready)
        n = max(0,int(np.floor((b-a)/durations.sum())))
        starts.extend(a+np.arange(n)*durations.sum()-trigger)
    return np.asarray(starts)[:,None]+offsets[None,:]
