from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz, CIRS
from astropy.time import Time
from ccsnlc import LightCurveModel
from galsnrubin import Config
from galsnrubin.simulation import localization, observe, sample_catalog
from galsnrubin.photometry import limits, exposure_magnitudes
from galsnrubin.visibility import maximum_altitude, altitude, pointing_starts, Calendar


def test_localization_geometry():
    cfg=Config()
    theta,area,p=localization(np.array([.1,25,100]),cfg)
    assert theta[1]==30
    assert area[1]==pytest.approx(2*np.pi*(1-np.cos(np.pi/6))*(180/np.pi)**2)
    assert area[2]==pytest.approx(4*np.pi*(180/np.pi)**2)
    assert p[0]==1 and np.all((p>0)&(p<=1))


def test_max_altitude_and_cadence():
    assert maximum_altitude(80,-30)==-20
    assert maximum_altitude(50,-30)==10
    cfg=Config()
    s=pointing_starts([(100,500),(1000,1100)],0,cfg)
    np.testing.assert_allclose(s[0],[190,206,222,252])
    assert s.shape==(4,4)
    assert np.all(s[-1]+np.array(cfg.exposures_s)+cfg.dead_time_s<=1100)
    assert len(pointing_starts([(0,100)],0,cfg))==0


def test_noise_and_saturation_scaling():
    cfg=Config()
    depth,sat=limits(cfg,[1,15,30])
    assert np.all(np.diff(depth)>0)
    assert sat[1]-sat[0]==pytest.approx(2.5*np.log10(15))
    zp=cfg.zeropoints[4]; t=1
    counts=t*10**(.4*(zp-depth[0]))
    var=2.266*(cfg.seeing_arcsec[4]/cfg.pixel_scale_arcsec)**2*(
        t*10**(.4*(zp-cfg.sky_mags[4]))*cfg.pixel_scale_arcsec**2+cfg.read_noise_e**2+cfg.dark_current_e_s*t)
    assert counts/np.sqrt(counts+var)==pytest.approx(5)


class ToyModel:
    delay=0
    t_end=1e6
    def __init__(self, mag): self.mag=mag
    def magnitude(self,t,band): return np.broadcast_to(self.mag,np.shape(t))


def test_detection_saturation_and_misses():
    cfg=Config()
    starts=pointing_starts([(0,1000)],0,cfg)
    result,_=observe(ToyModel(5),starts,1,np.random.default_rng(1),cfg)
    assert result['immediate_saturation'] and result['detected']
    assert result['n_exposures']==1 and not result['usable_photometry']
    result,_=observe(ToyModel(18),starts,1,np.random.default_rng(1),cfg)
    assert result['outcome']=='detected_unsaturated' and result['usable_photometry']
    result,_=observe(ToyModel(18),starts,0,np.random.default_rng(1),cfg)
    assert result['outcome']=='search_miss'
    result,_=observe(ToyModel(35),starts,1,np.random.default_rng(1),cfg)
    assert result['outcome']=='too_faint'
    result,_=observe(ToyModel(np.nan),starts,1,np.random.default_rng(1),cfg)
    assert result['outcome']=='no_valid_model_exposure'


def test_tracking_after_discovery():
    cfg=replace(Config(),exposures_s=(1.,),filter_change_s=0,dead_time_s=0)
    class Evolving(ToyModel):
        def magnitude(self,t,band): return np.where(t<2,18.,5.)
    class Draws:
        def random(self,n): return np.array([0.,1.,1.,1.])
    result,audit=observe(Evolving(0),np.arange(4.)[:,None],.1,Draws(),cfg)
    assert result['saturation_s']==3
    assert result['usable_photometry'] and not result['immediate_saturation']
    assert np.all(audit['in_field'])


def test_lightcurve_integration_and_validity():
    cfg=Config()
    m=LightCurveModel('WR',10,5,bands=('z',))
    before=np.zeros((1,4))
    assert np.all(np.isinf(exposure_magnitudes(m,before,cfg)))
    after=np.full((1,4),m.delay+m.t_end+1)
    assert np.all(np.isnan(exposure_magnitudes(m,after,cfg)))
    starts=np.full((1,4),m.delay+5000)
    np.testing.assert_allclose(exposure_magnitudes(m,starts,cfg),m.magnitude(starts+np.array(cfg.exposures_s)/2,'z'),atol=.002)


def test_geometric_distribution():
    p=.1
    # Distribution of the actual independent Bernoulli search implementation.
    draws=np.random.default_rng(2).random((100000,10))<p
    assert abs(draws.any(axis=1).mean()-(1-(1-p)**10))<.005


def test_catalog_distance_and_reproducibility(tmp_path):
    path=tmp_path/'c.pkl'
    pd.DataFrame({'ra':[10.,20.],'dec':[-20.,0.],'mu0':[10.,15.]}).to_pickle(path)
    cfg=replace(Config(),n_candidates=20)
    a=sample_catalog(path,cfg); b=sample_catalog(path,cfg)
    pd.testing.assert_frame_equal(a,b)
    assert set(a.distance_kpc)=={1.,10.}
    assert np.all(a.trigger_unix>=Time('2027-01-01').unix)
    assert np.all(a.trigger_unix<Time('2028-01-01').unix)


def test_astrometry_and_windows():
    cfg=replace(Config(),ephemeris_step_s=1800.)
    cal=Calendar(cfg)
    coord=SkyCoord(ra=270*u.deg,dec=-30*u.deg)
    trigger=Time('2027-07-01').unix
    windows,maxalt,angles=cal.windows(coord,trigger)
    assert windows and maxalt>89
    starts=pointing_starts(windows,trigger,cfg)
    t=Time(trigger+starts.ravel(),format='unix')
    exact=coord.transform_to(AltAz(obstime=t,location=cal.location,pressure=0*u.hPa)).alt.deg
    approx=altitude(*angles,np.interp(t.unix,cal.grid,cal.era),cfg.latitude_deg)
    np.testing.assert_allclose(approx,exact,atol=.02)
    assert np.all(exact>=cfg.min_alt_deg-.02)
    for a,b in windows:
        assert any(a>=dusk and b<=dawn for dusk,dawn in cal.nights)


@pytest.mark.parametrize('kwargs',[{'band':'zz'},{'exposures_s':[]},{'min_alt_deg':90},
    {'followup_days':0},{'n_candidates':1.2},{'read_noise_e':-1},{'seeing_arcsec':(1,)}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError): Config(**kwargs)
