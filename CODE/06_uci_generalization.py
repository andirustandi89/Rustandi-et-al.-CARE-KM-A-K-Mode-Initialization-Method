import argparse
import numpy as np
import pandas as pd
from experiment_common import ROOT, REPRODUCED, METHODS, STOCHASTIC, CARE_CONFIG, MAX_ITER, SI_SAMPLE_SIZE, sample_hash
from care_km import run_method
from benchmark_data import DATASET_ORDER, DATASETS, load_dataset
SOURCE=ROOT/'DATA/BENCHMARK/SOURCE'

def run():
    full=[]; res=[]
    for key in DATASET_ORDER:
        Xdf,_=load_dataset(SOURCE,key); X=Xdf.astype(str).to_numpy(object); k=int(DATASETS[key]['reference_k'])
        for method in METHODS:
            reps=15 if method in STOCHASTIC else 1
            for rr in range(reps):
                seed=120000+rr; row,_,_=run_method(method,X,None,k,seed=seed,care_config=CARE_CONFIG,si_sample_size=SI_SAMPLE_SIZE,si_seed=120999,max_iter=MAX_ITER); row.update(dataset=key,method=method,run=rr,seed=seed,phase='full'); full.append(row)
        for rep in range(30):
            seed=130000+rep; rng=np.random.default_rng(seed); idx=np.sort(rng.choice(len(X),size=int(len(X)*.8),replace=False)); x=X[idx]; h=sample_hash(idx)
            for method in METHODS:
                row,_,_=run_method(method,x,None,k,seed=seed,care_config=CARE_CONFIG,si_sample_size=SI_SAMPLE_SIZE,si_seed=seed,max_iter=MAX_ITER); row.update(dataset=key,method=method,rep=rep,seed=seed,phase='resample',sample_n=len(idx),sample_hash=h); res.append(row)
    pd.DataFrame(full).to_csv(REPRODUCED/'generalization_full_runs.csv',index=False); pd.DataFrame(res).to_csv(REPRODUCED/'generalization_resample_runs.csv',index=False)
run()
