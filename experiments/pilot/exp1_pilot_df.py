"""Pilot (separate seeds): reference distribution of DA-HAC and maximum variogram lag.

Run from the repository root: python experiments/pilot/exp1_pilot_df.py
"""
import sys; sys.path.insert(0, 'experiments')
import numpy as np, pandas as pd, itertools
from multiprocessing import Pool
import exp1_controlled as e
from dacomp import da_hac, blocked_folds, make_blocks
from dacomp.variogram import dependence_ranges
from dacomp.simulate import *
def run(job):
    level, rep = job
    a, phi = DEPENDENCE_LEVELS[level]; coords = grid_coords(12); chol = spatial_cholesky(coords, a)
    rng = np.random.default_rng([999001, list(DEPENDENCE_LEVELS).index(level), 0, rep])
    cs, t = panel_index(coords, 24)
    eps, ea, eb = st_fields(rng, chol, len(coords), 24, phi, 3)
    blocks = make_blocks(cs, t, 4, 6); folds = blocked_folds(blocks, 5, rng)
    common = rng.standard_normal(); eta = 0.05*(np.sqrt(.25)*common+np.sqrt(.75)*rng.standard_normal(5))
    d = (eps-ea)**2-(eps-eb)**2+eta[folds]
    out=[]
    for mls, mlt, tag in [(6,12,'half'),(11,23,'full')]:
        rs = da_hac(d, folds, cs, t, max_lag_space=mls, max_lag_time=mlt,
                    variants=[(f'neff_{tag}',True,'satterthwaite_neff'),(f'kdf_{tag}',True,'satterthwaite')])
        out += [dict(level=level, method=r.method, p=r.pvalue, df=r.df, nuk=r.info['nu_kernel'], rs=r.info['range_space']) for r in rs]
    return out
jobs=[(l,r) for l in DEPENDENCE_LEVELS for r in range(300)]
with Pool(4) as p: rows=list(itertools.chain.from_iterable(p.imap_unordered(run, jobs, chunksize=4)))
d=pd.DataFrame(rows); d['rej']=d.p<0.05
print((100*d.pivot_table(index='method',columns='level',values='rej')[list(DEPENDENCE_LEVELS)]).round(1))
print(d.groupby(['method','level'])[['df','nuk','rs']].median().round(1))
