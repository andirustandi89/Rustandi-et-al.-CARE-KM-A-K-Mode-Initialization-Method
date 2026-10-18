"""Recompute dataset-level ranks and benchmark statistical comparisons."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from run_lms_experiments import METHODS
ROOT=Path(__file__).resolve().parents[1]

def holm(p):
    p=np.asarray(p);order=np.argsort(p);q=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]);out=np.empty(len(p));out[order]=np.minimum(q,1);return out

def run(raw,out):
    d=pd.read_csv(raw)
    if d.duplicated(['dataset','rep','method']).any():raise ValueError('Duplicate experiment rows.')
    counts=d.groupby(['dataset','rep']).method.nunique()
    if not counts.eq(7).all():raise ValueError('Expected all seven methods in every resample.')
    if not d.groupby(['dataset','rep']).sample_hash.nunique().eq(1).all():raise ValueError('Samples are not matched.')
    means=d.groupby(['dataset','method'])[['initial_cost','final_cost','dbi','si','iterations']].mean().reset_index()
    ranks=[];tests=[];pairs=[]
    for metric in ['initial_cost','final_cost','dbi','si','iterations']:
        pivot=means.pivot(index='dataset',columns='method',values=metric)[METHODS]
        if pivot.isna().any().any():raise ValueError('Incomplete dataset means.')
        rank=pivot.rank(axis=1,ascending=metric!='si').mean()
        ranks.extend(dict(metric=metric,method=m,average_rank=rank[m]) for m in METHODS)
        stat,p=friedmanchisquare(*[pivot[m] for m in METHODS]);n,k=pivot.shape
        tests.append(dict(metric=metric,friedman_chi2=stat,p_value=p,kendall_w=stat/(n*(k-1))))
        family=[]
        for baseline in METHODS:
            if baseline=='CARE-KM':continue
            delta=(pivot['CARE-KM']-pivot[baseline]) if metric=='si' else (pivot[baseline]-pivot['CARE-KM'])
            nz=delta[np.abs(delta)>1e-12].to_numpy();r=rankdata(abs(nz))
            effect=((r[nz>0].sum()-r[nz<0].sum())/r.sum()) if len(nz) else 0
            p=wilcoxon(delta,alternative='two-sided',zero_method='wilcox',method='auto').pvalue if len(nz) else 1.
            family.append(dict(metric=metric,baseline=baseline,wins=int((delta>1e-12).sum()),ties=int((abs(delta)<=1e-12).sum()),losses=int((delta< -1e-12).sum()),rank_biserial=effect,p_value=p))
        for row,p in zip(family,holm([r['p_value'] for r in family])):row['holm_p']=p
        pairs.extend(family)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('benchmark_ranks',ranks),('benchmark_friedman',tests),('benchmark_pairwise',pairs)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    print(pd.DataFrame(tests).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=ROOT/'OUTPUT/experiments/10_cross_dataset_resample_raw.csv')
    p.add_argument('--out',type=Path,default=ROOT/'OUTPUT/recomputed/statistics')
    a=p.parse_args();run(a.raw,a.out)
