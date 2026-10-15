import argparse
import numpy as np
import pandas as pd
from experiment_common import ROOT, REPRODUCED, load_lms, sample_hash, METHODS, STOCHASTIC, CARE_CONFIG, K, MAX_ITER, SI_SAMPLE_SIZE
from care_km import run_method

def run_full(X):
    rows=[]
    for method in METHODS:
        reps=15 if method in STOCHASTIC else 1
        for rep in range(reps):
            seed=4161600+rep
            row,_,_=run_method(method,X,None,K,seed=seed,care_config=CARE_CONFIG,si_sample_size=SI_SAMPLE_SIZE,si_seed=seed,max_iter=MAX_ITER)
            row.update(method=method,rep=rep,seed=seed,stochastic_initializer=method in STOCHASTIC)
            rows.append(row)
    pd.DataFrame(rows).to_csv(REPRODUCED/'full_data_runs.csv',index=False)

def run_ablation(X):
    rows=[]
    for rep in range(20):
        seed=5161600+rep; rng=np.random.default_rng(seed); idx=np.sort(rng.choice(len(X),size=int(len(X)*.8),replace=False)); x=X[idx]; h=sample_hash(idx)
        for name,nex in [('A0_Cao',0),('A1_1_exchange',1),('A2_2_exchanges',2),('A3_CARE_4_exchanges',4)]:
            cfg=dict(CARE_CONFIG); cfg['max_exchanges']=nex
            method='Cao' if nex==0 else 'CARE-KM'
            row,_,_=run_method(method,x,None,K,seed=seed,care_config=cfg,si_sample_size=SI_SAMPLE_SIZE,si_seed=seed,max_iter=MAX_ITER)
            row.update(variant=name,max_exchanges=nex,rep=rep,seed=seed,sample_hash=h); rows.append(row)
    pd.DataFrame(rows).to_csv(REPRODUCED/'ablation_runs.csv',index=False)

def run_robustness(X):
    rows=[]
    for rep in range(30):
        seed=6161600+rep; rng=np.random.default_rng(seed); idx=np.sort(rng.choice(len(X),size=int(len(X)*.8),replace=False)); x=X[idx]; h=sample_hash(idx)
        for method in METHODS:
            row,_,_=run_method(method,x,None,K,seed=seed,care_config=CARE_CONFIG,si_sample_size=SI_SAMPLE_SIZE,si_seed=seed,max_iter=MAX_ITER)
            row.update(method=method,rep=rep,seed=seed,sample_hash=h,sample_n=len(idx)); rows.append(row)
    pd.DataFrame(rows).to_csv(REPRODUCED/'robustness_runs.csv',index=False)

parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['full','ablation','robustness','all'],default='full'); args=parser.parse_args()
_,X=load_lms()
if args.stage in ('full','all'): run_full(X)
if args.stage in ('ablation','all'): run_ablation(X)
if args.stage in ('robustness','all'): run_robustness(X)
