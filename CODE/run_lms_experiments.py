"""Run full-data, exchange-depth, or matched-resample LMS experiments."""
from pathlib import Path
import argparse
import hashlib
import numpy as np
import pandas as pd
import care_km as ck
from fslsm import MAIN_FEATURES, CARE_CONFIG
ROOT=Path(__file__).resolve().parents[1]
METHODS=['Random','Huang','Cao','Bai','K-Modes++','HR-2023','CARE-KM']
STOCHASTIC={'Random','Huang','K-Modes++','HR-2023'}

def run(mode,out):
    data=pd.read_csv(ROOT/'DATA/main_lms/behavior_categorical.csv')
    X=data[MAIN_FEATURES].astype(str).to_numpy(dtype=object)
    rows=[]
    if mode=='full':
        jobs=[(m,r,4161600+r,None,None) for m in METHODS for r in range(15 if m in STOCHASTIC else 1)]
    else:
        reps,seed=(20,5161600) if mode=='ablation' else (30,6161600)
        jobs=[]
        for r in range(reps):
            idx=np.sort(np.random.default_rng(seed+r).choice(len(X),size=int(.8*len(X)),replace=False))
            options=[('Cao',0),('CARE-KM',1),('CARE-KM',2),('CARE-KM',4)] if mode=='ablation' else [(m,None) for m in METHODS]
            jobs.extend((m,r,seed+r,idx,depth) for m,depth in options)
    out.mkdir(parents=True,exist_ok=True)
    for method,rep,seed,idx,depth in jobs:
        cfg=dict(CARE_CONFIG)
        if depth is not None:cfg['max_exchanges']=depth
        sample=X if idx is None else X[idx]
        row,initial,fit=ck.run_method(method,sample,None,16,seed=seed,care_config=cfg,si_sample_size=2000,si_seed=seed,max_iter=100)
        # Iteration counts describe convergence; wall-clock efficiency is not assessed.
        row={k:v for k,v in row.items() if not k.endswith('_sec')}
        row.update(rep=rep)
        if idx is not None:row['sample_hash']=hashlib.sha256(idx.tobytes()).hexdigest()
        if depth is not None:row['max_exchanges']=depth
        rows.append(row)
        pd.DataFrame(rows).to_csv(out/(mode+'_raw.csv'),index=False)
        if mode=='full' and method=='CARE-KM':
            pd.DataFrame(fit['modes'],columns=MAIN_FEATURES).to_csv(out/'full_modes.csv',index=False)
            data[['learner_id']].assign(cluster_id=fit['labels']).to_csv(out/'full_membership.csv',index=False)
            assert fit['cost']==23822
            assert round(row['dbi'],3)==1.541 and round(row['si'],3)==.120
        print(method,rep,'cost',row['final_cost'],flush=True)
    group=['method']+(['max_exchanges'] if mode=='ablation' else [])
    pd.DataFrame(rows).groupby(group)[['initial_cost','final_cost','dbi','si','iterations']].agg(['mean','std']).to_csv(out/(mode+'_summary.csv'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['full','ablation','robustness'],default='full')
    p.add_argument('--out',type=Path,default=ROOT/'OUTPUT/recomputed/lms')
    a=p.parse_args();run(a.mode,a.out)
