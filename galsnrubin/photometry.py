"""Single-exposure ideal dark-sky sensitivity, including read noise."""
import numpy as np


def limits(config, durations=None):
    """Return (5-sigma limit, saturation limit) in AB mag for exposures [s].

    No coaddition. Saturation scales at fixed charge capacity. Seeing and
    sky are held at zenith values; these are forecasts, not an operations ETC.
    """
    t = np.asarray(config.exposures_s if durations is None else durations, float)
    i = 'ugrizy'.index(config.band)
    zp = config.zeropoints[i]
    background = t * 10**(.4*(zp-config.sky_mags[i])) * config.pixel_scale_arcsec**2
    neff = 2.266*(config.seeing_arcsec[i]/config.pixel_scale_arcsec)**2
    variance = neff*(background + config.read_noise_e**2 + config.dark_current_e_s*t)
    q = config.snr_threshold**2
    counts = (q + np.sqrt(q*q+4*q*variance))/2
    return zp-2.5*np.log10(counts/t), config.saturation15_mags[i]+2.5*np.log10(t/15.)


def exposure_magnitudes(model, starts, config):
    """Flux-average each exposure; pre-breakout flux is zero.

    Exposures extending past the model cutoff are unknown (NaN), never dark.
    starts has shape (pointings, exposures); output has the same shape.
    """
    duration = np.asarray(config.exposures_s)
    phase = (np.arange(config.integration_samples)+.5)/config.integration_samples
    times = starts[..., None] + duration[None, :, None]*phase
    m = model.magnitude(times.ravel(), config.band).reshape(times.shape)
    flux = np.where(np.isfinite(m), 10**(-.4*m), 0.)
    with np.errstate(divide='ignore'):
        result = -2.5*np.log10(flux.mean(axis=-1))
    result[starts+duration > model.delay+model.t_end] = np.nan
    return result
