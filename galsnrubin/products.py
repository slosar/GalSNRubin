"""Statistical and QA figures, plus LaTeX tables tied to the saved run."""
from pathlib import Path
from dataclasses import fields
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ccsnlc import LightCurveModel
from .photometry import limits


def make_sky_products(results, figures, config):
    """Galactic HEALPix count maps and extinction of the shared candidate sample.

    Maps use RING nside=32 (3.36 square degrees per pixel), Galactic centre
    at the centre and longitude increasing to the left. Counts are not rates
    or efficiencies: repeated sampled catalogue rows count as distinct events.
    """
    import healpy as hp
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MaxNLocator

    figures = Path(figures)
    nside = 32
    coords = SkyCoord(ra=results.ra.to_numpy()*u.deg,
                      dec=results.dec.to_numpy()*u.deg).galactic
    pixels = hp.ang2pix(nside, coords.l.deg, coords.b.deg, lonlat=True)

    def count_map(mask):
        return np.bincount(pixels[np.asarray(mask)], minlength=hp.nside2npix(nside))

    def save(fig, name):
        # Healpy positions its own axes; tight_layout does not support these axes.
        for ext in ['png', 'pdf']:
            fig.savefig(figures/f'{name}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)

    def maps_figure(maps, titles, rows, cols, name, heading):
        vmax = max(1, max(int(m.max()) for m in maps))
        fig = plt.figure(figsize=(5*cols, 2.9*rows+1.0))
        for i, (counts, title) in enumerate(zip(maps, titles)):
            visible = np.where(counts > 0, counts.astype(float), hp.UNSEEN)
            hp.mollview(visible, fig=fig.number, sub=(rows, cols, i+1),
                        coord='G', flip='astro', rot=(0, 0, 0),
                        min=0, max=vmax, cmap='viridis', cbar=False,
                        badcolor='#f0f2f4', title=f'{title} (N={counts.sum()})',
                        notext=True, margins=(.015, .20/rows, .015, .08/rows))
            hp.graticule(dpar=30, dmer=60, color='#adb5bd', alpha=.5, local=True)
        fig.suptitle(heading, y=1.05, fontsize=14)
        cax = fig.add_axes([.32, .07 if rows == 1 else .03, .36, .025 if rows == 1 else .012])
        bar = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap='viridis'),
                           cax=cax, orientation='horizontal')
        bar.set_label(f'Events per HEALPix pixel (nside={nside}; {hp.nside2pixarea(nside, degrees=True):.2f} deg²)')
        bar.locator = MaxNLocator(nbins=6, integer=True)
        bar.update_ticks()
        save(fig, name)

    successes = [count_map((results.progenitor == p) & results.detected)
                 for p in config.progenitors]
    maps_figure(successes, [f'{p}: detected' for p in config.progenitors],
                1, len(config.progenitors), 'sky_success',
                'Successful detections in Galactic coordinates')
    failures, titles = [], []
    reasons = [('no_night_visibility', 'No night visibility'),
               ('too_faint', 'Too faint'), ('search_miss', 'Search miss')]
    for p in config.progenitors:
        for reason, label in reasons:
            failures.append(count_map((results.progenitor == p) & (results.outcome == reason)))
            titles.append(f'{p}: {label}')
    maps_figure(failures, titles, len(config.progenitors), 3, 'sky_failures',
                'Missed supernovae in Galactic coordinates, by failure reason')

    # Drop progenitor copies, not repeated catalogue rows in distinct candidates.
    candidates = results.drop_duplicates('candidate_id')
    av = candidates.Av.to_numpy()
    bins = np.arange(0, max(2., 2*np.ceil(av.max()/2))+2, 2.)
    counts, edges = np.histogram(av, bins=bins)
    fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
    ax.hist(av, bins=edges, color='#4477aa', edgecolor='white', linewidth=.6)
    ax.axvline(np.median(av), color='#aa3377', ls='--',
               label=fr'Median $A_V = {np.median(av):.2f}$ mag')
    ax.set(xlabel=r'V-band extinction $A_V$ [mag]', ylabel='Number of candidates',
           title=f'Extinction of the {len(av):,} shared candidates', xlim=(0, edges[-1]))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    save(fig, 'av_distribution')
    np.savez(figures/'sky_extinction_counts.npz', nside=nside,
             progenitors=np.array(config.progenitors),
             failure_reasons=np.array([reason for reason, _ in reasons]),
             success_counts=np.array(successes),
             failure_counts=np.array(failures).reshape(len(config.progenitors), 3, -1),
             av_counts=counts, av_edges=edges)


def escape(value):
    return str(value).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def make_products(results, output_dir, config):
    output = Path(output_dir)
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    make_sky_products(results, figures, config)
    colors = dict(RSG='tab:red',BSG='tab:blue',WR='tab:green')
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:
            fig.savefig(figures/f'{name}.{ext}',dpi=160)
        plt.close(fig)
    def ecdf(ax,values,label,color,denominator=None):
        x = np.sort(np.asarray(values)[np.isfinite(values)])
        if len(x):
            ax.step(x,np.arange(1,len(x)+1)/(len(x) if denominator is None else denominator),where='post',label=label,color=color)
        else:
            ax.plot([],[],label=label+' (none)',color=color)
    groups = [(p,results[results.progenitor==p]) for p in config.progenitors]
    outcomes = ['never_above_horizon','never_above_min_alt','no_night_visibility','no_complete_pointing',
                'no_valid_model_exposure','too_faint','search_miss','detected_unsaturated','detected_saturated']
    fig,ax = plt.subplots(figsize=(10,4.5))
    for p,g in groups:
        counts = g.outcome.value_counts()
        rates = np.array([counts.get(o,0)/len(g) for o in outcomes])
        ax.errorbar(np.arange(len(outcomes)),rates,yerr=np.sqrt(rates*(1-rates)/len(g)),marker='o',label=p,color=colors[p],capsize=3)
    ax.set_xticks(np.arange(len(outcomes)),[o.replace('_','\n') for o in outcomes],fontsize=8)
    ax.set_ylabel('Fraction of all triggers (1 sigma MC error)');ax.legend();save(fig,'outcomes')
    fig,axes = plt.subplots(1,3,figsize=(12,3.8))
    for p,g in groups:
        d = g[g.detected]
        ecdf(axes[0],d.first_detection_s/3600,p,colors[p],len(g))
        ecdf(axes[1],d.discovery_mag,p,colors[p])
        ecdf(axes[2],g[g.saturation_s.notna()].saturation_s/3600,p,colors[p],len(g))
    for ax,xlabel,ylabel in zip(axes,['Hours from trigger to first detection','Discovery AB magnitude','Hours from trigger to first saturation'],['Fraction of all triggers detected','Fraction of detections','Fraction of all triggers saturated']):
        ax.set(xlabel=xlabel,ylabel=ylabel,ylim=(0,1));ax.legend()
    save(fig,'discovery')
    fig,axes = plt.subplots(1,2,figsize=(10,3.8))
    for p,g in groups:
        values = [g.detected.mean(),g.immediate_saturation.mean(),g.usable_photometry.mean()]
        axes[0].plot(range(3),values,'o-',label=p,color=colors[p])
        d=g[g.detected]
        axes[1].scatter(d.first_detection_s/3600,d.discovery_mag,s=9,alpha=.35,label=p,color=colors[p])
    axes[0].set_xticks(range(3),['Detected','Saturated at\nfirst detection','Any usable\nphotometry'])
    axes[0].set(ylabel='Fraction of all triggers',ylim=(0,1));axes[0].legend()
    axes[1].set(xlabel='Hours from trigger',ylabel='Discovery AB magnitude');axes[1].invert_yaxis();axes[1].legend()
    save(fig,'saturation')
    c=groups[0][1]
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    axes[0,0].scatter(c.ra,c.dec,c=c.Av,s=6,cmap='magma',vmin=0,vmax=50)
    axes[0,0].set(xlabel='ICRS RA [deg]',ylabel='ICRS Dec [deg]',title='Sampled sky (colour: Av, 0 to 50 mag)')
    axes[0,1].scatter(c.distance_kpc,c.Av,s=6);axes[0,1].set(xlabel='Distance [kpc]',ylabel='Av [mag]')
    axes[0,2].scatter(c.distance_kpc,c.p_hit,s=6);axes[0,2].set(xlabel='Distance [kpc]',ylabel='Per-pointing hit probability',yscale='log')
    for p,g in groups:
        ecdf(axes[1,0],g.first_observation_s/3600,p,colors[p],len(g))
        d=g[g.detected]
        ecdf(axes[1,1],d.discovery_alt_deg,p,colors[p])
        ecdf(axes[1,2],d.discovery_sun_alt_deg,p,colors[p])
    axes[1,0].set(xlabel='First observation [hours]',ylabel='Fraction of all triggers')
    axes[1,1].set(xlabel='Source altitude at discovery [deg]',ylabel='Fraction of detections');axes[1,1].axvline(config.min_alt_deg,c='k',ls=':')
    axes[1,2].set(xlabel='Solar altitude at discovery [deg]',ylabel='Fraction of detections');axes[1,2].axvline(config.twilight_deg,c='k',ls=':')
    for ax in axes[1]: ax.legend()
    save(fig,'qa_population')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    depth,sat=limits(config)
    times=np.geomspace(30,config.followup_days*86400,2000)
    for p in config.progenitors:
        m=LightCurveModel(p,10.,10.,bands=(config.band,),Rv=config.Rv,T_min_eV=config.T_min_eV,ngrid=config.lightcurve_ngrid)
        axes[0].plot(times/3600,m.magnitude(times,config.band),label=p,color=colors[p])
        axes[0].axvline(m.delay/3600,color=colors[p],ls=':',alpha=.5)
    for t,d,s in zip(config.exposures_s,depth,sat):
        axes[0].axhline(d,color='grey',ls='--',alpha=.4);axes[0].axhline(s,color='grey',ls=':',alpha=.4)
    axes[0].set(xlabel='Hours from trigger',ylabel=f'{config.band}-band AB magnitude',xscale='log',title='10 kpc, Av=10; dotted vertical: breakout');axes[0].invert_yaxis();axes[0].legend()
    from .simulation import localization
    distance=np.linspace(.1,40,400)
    theta,area,prob=localization(distance,config)
    axes[1].plot(distance,theta,label='Cap radius [deg]');axes[1].plot(distance,area,label='Cap area [deg2]')
    axes[1].set(xlabel='Distance [kpc]',ylabel='Radius / area (log scale)',yscale='log');axes[1].legend()
    save(fig,'qa_models')
    # Same random-draw algorithm: geometric hit frequency regression figure.
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    rng=np.random.default_rng(config.seed)
    for p in [.01,.1,.5]:
        draws=rng.geometric(p,10000); x=np.arange(1,int(np.quantile(draws,.99))+1)
        axes[0].plot(x,[(draws<=k).mean() for k in x],label=f'MC p={p}')
        axes[0].plot(x,1-(1-p)**x,'k--',alpha=.5)
    axes[0].set(xlabel='Independent pointings',ylabel='Probability of at least one hit',xscale='log');axes[0].legend()
    axes[1].hist((c.trigger_unix-c.trigger_unix.min())/86400,bins=12)
    axes[1].set(xlabel='Days since earliest sampled trigger',ylabel='Candidates',title='Uniform 2027 trigger dates')
    save(fig,'qa_sampling')
    table=[r'\begin{longtable}{p{.29\linewidth}p{.25\linewidth}p{.37\linewidth}}',r'\toprule Parameter & Value & Meaning \\ \midrule\endhead']
    for f in fields(config):
        table.append(f'{escape(f.name)} & {escape(getattr(config,f.name))} & {escape(f.metadata["description"])} '+r'\\')
    table += [r'\bottomrule\end{longtable}']
    (output/'parameters.tex').write_text('\n'.join(table))
    summary=json.loads((output/'summary.json').read_text())
    table=[r'\begin{tabular}{lrrrrr}\toprule',r'Type & Detected & Immediate sat. & Usable & Median hours & Median mag \\ \midrule']
    for p,s in summary.items():
        h='--' if s['median_detection_hours'] is None else f'{s["median_detection_hours"]:.2f}'
        m='--' if s['median_discovery_mag'] is None else f'{s["median_discovery_mag"]:.2f}'
        table.append(f'{p} & {s["detected"]}/{s["n"]} & {s["immediate_saturation"]} & {s["usable_photometry"]} & {h} & {m} '+r'\\')
    table.append(r'\bottomrule\end{tabular}')
    (output/'results.tex').write_text('\n'.join(table))


if __name__ == '__main__':
    import argparse
    import pandas as pd
    from .config import Config
    parser = argparse.ArgumentParser(description='Regenerate figures and tables from a saved run.')
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    make_products(pd.read_csv(args.output_dir/'events.csv'), args.output_dir,
                  Config.load(args.output_dir/'config.json'))
