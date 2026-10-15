from pathlib import Path
import hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def c(path,rows=None,cols=None):
    p=ROOT/path; ok=p.is_file(); msg='missing' if not ok else 'ok'
    if ok and p.suffix.lower()=='.csv':
        d=pd.read_csv(p); ok &= rows is None or len(d)==rows; ok &= cols is None or d.shape[1]==cols; msg=f'{len(d)}x{d.shape[1]}'
    checks.append((path,ok,msg))
c('DATA/MAIN_LMS/main_lms_numeric_6658x13.csv',6658,14); c('DATA/MAIN_LMS/main_lms_categorical_6658x13.csv',6658,14); c('DATA/EXTERNAL_ILS/ieee_questionnaire_groundtruth_736_audit.csv',736,7); c('SPLITS/main_lms_70_30_split.csv',6658,3); c('SPLITS/external_ieee_70_30_split.csv',736,4); c('OUTPUT/CLUSTERING/robustness_runs.csv',210); c('OUTPUT/GENERALIZATION/generalization_resample_runs.csv',2100); c('OUTPUT/PREDICTION/main_lms_test_predictions.csv',7992); c('OUTPUT/PREDICTION/external_ieee_test_predictions.csv',884)
for p,ok,msg in checks: print(('PASS' if ok else 'FAIL'),p,msg)
if not all(x[1] for x in checks): raise SystemExit(1)
print('Package validation: PASS')
