"""Validate released data and recompute holdout metrics from saved predictions."""
from pathlib import Path
import ast
import hashlib
import argparse
import numpy as np
import pandas as pd
import fslsm as f
from run_questionnaire_holdout import metrics
ROOT=Path(__file__).resolve().parents[1]

def check(out):
    main=pd.read_csv(ROOT/'DATA/main_lms/behavior_categorical.csv')
    split=pd.read_csv(ROOT/'DATA/main_lms/split_membership.csv')
    assert len(main)==6658 and main.learner_id.is_unique
    assert main.columns.tolist()==['learner_id']+f.MAIN_FEATURES
    assert set(np.unique(main[f.MAIN_FEATURES]))<={1,2,3}
    assert split.split.value_counts().to_dict()=={'train':4660,'test':1998}
    assert set(main.learner_id)==set(split.learner_id)
    q=pd.read_csv(ROOT/'DATA/questionnaire_cohort/ils_labels_and_split.csv')
    assert len(q)==461 and q.learner_id.is_unique
    assert q.split.value_counts().to_dict()=={'train':322,'test':139}
    for part,n in [('train',322),('test',139)]:
        b=pd.read_csv(ROOT/f'DATA/questionnaire_cohort/behavior_{part}.csv')
        assert len(b)==n and b.learner_id.is_unique
        assert set(b.learner_id)==set(q.loc[q.split.eq(part),'learner_id'])
    h=ROOT/'OUTPUT/holdout'
    pred=pd.read_csv(h/'70_30_main_lms_prospective_predictions.csv')
    mapping=pd.read_csv(h/'70_30_main_lms_frozen_cluster_to_fslsm_mapping.csv').set_index('cluster_id')
    rows=[]
    for name in ['NB','DT','SVM']:
        for key,title in zip(f.DIM_KEYS,f.DIM_NAMES):
            truth=pred.reference_cluster_id.map(mapping[key]).map(f.pole_abbr_to_full)
            rows.append(dict(cohort='main_lms',method=name,dimension=key,**metrics(truth,pred[name+'_'+title])))
    p=pd.read_csv(h/'70_30_questionnaire_cohort_classifier_predictions.csv')
    direct=pd.read_csv(h/'70_30_questionnaire_cohort_test_carekm_vs_ils_full.csv').set_index('learner_id').loc[p.learner_id].reset_index()
    for name in ['NB','DT','SVM','CARE-KM']:
        for key in f.DIM_KEYS:
            truth=p[key+'_code'].map(f.ils_code_to_full)
            estimate=direct['carekm_'+key].map(f.pole_abbr_to_full) if name=='CARE-KM' else p[name+'_'+key]
            rows.append(dict(cohort='questionnaire_cohort',method=name,dimension=key,**metrics(truth,estimate)))
    results=pd.DataFrame(rows)
    summary=results.groupby(['cohort','method'])[['accuracy','balanced_accuracy','macro_f1']].mean()
    # All manuscript summaries, including imbalance-sensitive measures.
    expected={('main_lms','NB'):[81.31,79.44,78.01],('main_lms','DT'):[88.38,86.85,86.28],('main_lms','SVM'):[93.12,91.39,91.90],('questionnaire_cohort','NB'):[83.27,69.63,72.00],('questionnaire_cohort','DT'):[80.58,66.64,69.15],('questionnaire_cohort','SVM'):[82.19,68.70,70.91],('questionnaire_cohort','CARE-KM'):[79.32,66.75,69.50]}
    for pair,values in expected.items():
        assert np.allclose(np.round(100*summary.loc[pair].to_numpy(),2),values),pair
    for path in (ROOT/'CODE').glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    manifest=ROOT/'file_manifest.csv'
    if manifest.exists():
        for row in pd.read_csv(manifest).itertuples(index=False):
            assert hashlib.sha256((ROOT/row.path).read_bytes()).hexdigest()==row.sha256,row.path
    out.mkdir(parents=True,exist_ok=True)
    results.to_csv(out/'holdout_dimension_metrics.csv',index=False)
    summary.to_csv(out/'holdout_summary.csv')
    print((100*summary).round(2).to_string())
    print('Data, manuscript holdout summaries, Python syntax and available checksums passed.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'OUTPUT/recomputed/verification')
    check(p.parse_args().out)
