"""Run benchmark comparisons after placing the source archives in DATA/benchmarks."""
from pathlib import Path
import argparse
import hashlib
import numpy as np
import pandas as pd
import benchmark_data as bd
import care_km as ck
from fslsm import CARE_CONFIG
from run_lms_experiments import METHODS
ROOT=Path(__file__).resolve().parents[1]

def run(keys,out):
    rows=[];out.mkdir(parents=True,exist_ok=True)
    for key in keys:
        frame,y=bd.load_dataset(ROOT/'DATA/benchmarks',key)
        X=frame.to_numpy(dtype=object);k=int(y.nunique())
        for rep in range(30):
            seed=130000+rep
            idx=np.sort(np.random.default_rng(seed).choice(len(X),size=int(.8*len(X)),replace=False))
            for method in METHODS:
                row,_,_=ck.run_method(method,X[idx],None,k,seed=seed,care_config=CARE_CONFIG,si_sample_size=2000,si_seed=seed,max_iter=100)
                row={c:v for c,v in row.items() if not c.endswith('_sec')}
                row.update(dataset=key,rep=rep,sample_n=len(idx),sample_hash=hashlib.sha256(idx.tobytes()).hexdigest())
                rows.append(row)
            pd.DataFrame(rows).to_csv(out/'benchmark_resample_raw.csv',index=False)
            print(key,rep+1,flush=True)
    pd.DataFrame(rows).groupby(['dataset','method'])[['initial_cost','final_cost','dbi','si','iterations']].mean().to_csv(out/'benchmark_resample_means.csv')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=bd.DATASET_ORDER,default=bd.DATASET_ORDER)
    p.add_argument('--out',type=Path,default=ROOT/'OUTPUT/recomputed/benchmarks')
    a=p.parse_args();run(a.datasets,a.out)
