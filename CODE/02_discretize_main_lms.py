from pathlib import Path
import pandas as pd
from experiment_common import ROOT, FEATURES

def apply_thresholds(numeric, thresholds):
    out=pd.DataFrame({'record_id':numeric['record_id'].astype(str)})
    t=thresholds.set_index('feature')
    for f in FEATURES:
        lo=float(t.loc[f,'applied_low']); hi=float(t.loc[f,'applied_high'])
        x=pd.to_numeric(numeric[f],errors='raise')
        method=str(t.loc[f,'applied_method'])
        if 'ZERO_AWARE_POSITIVE_MEDIAN' in method:
            out[f]=1
            out.loc[x>0,f]=2
            out.loc[x>=hi,f]=3
        else:
            out[f]=1
            out.loc[x>=lo,f]=2
            out.loc[x>=hi,f]=3
    return out

numeric=pd.read_csv(ROOT/'DATA/MAIN_LMS/main_lms_numeric_6658x13.csv')
thresholds=pd.read_csv(ROOT/'DATA/MAIN_LMS/preprocessing_primary_thresholds.csv')
recreated=apply_thresholds(numeric,thresholds)
expected=pd.read_csv(ROOT/'DATA/MAIN_LMS/main_lms_categorical_6658x13.csv',dtype=str)
check=recreated.astype(str).equals(expected.astype(str))
path=ROOT/'REPRODUCED/main_lms_categorical_6658x13.csv'
recreated.to_csv(path,index=False)
print(f'Wrote {path}')
print('Exact match to archived categorical matrix:', check)
if not check: raise SystemExit(1)
