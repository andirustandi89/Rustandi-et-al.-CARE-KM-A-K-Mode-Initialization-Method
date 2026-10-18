"""Recompute the main LMS holdout when the original numeric matrix is supplied."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import care_km as ck
import fslsm as f
from run_questionnaire_holdout import metrics
ROOT=Path(__file__).resolve().parents[1]

def run(numeric,out):
    d=pd.read_csv(numeric)
    required=['learner_id']+f.MAIN_FEATURES
    if not set(required)<=set(d):raise ValueError('Expected learner_id and the 13 named numeric indicators.')
    if d.learner_id.duplicated().any():raise ValueError('Duplicate learner identifiers.')
    splits=pd.read_csv(ROOT/'DATA/main_lms/split_membership.csv')
    if set(d.learner_id)!=set(splits.learner_id):raise ValueError('Numeric records must match all released learner identifiers.')
    d=d.set_index('learner_id')
    tr=d.loc[splits.loc[splits.split.eq('train'),'learner_id']].reset_index().rename(columns={'learner_id':'user'})
    te=d.loc[splits.loc[splits.split.eq('test'),'learner_id']].reset_index().rename(columns={'learner_id':'user'})
    state,audit=f.fit_main_categorizer(tr)
    train=f.apply_main_categorizer(tr,state);test=f.apply_main_categorizer(te,state)
    X=train[f.MAIN_FEATURES].astype(str).to_numpy(dtype=object);T=test[f.MAIN_FEATURES].astype(str).to_numpy(dtype=object)
    _,fit=f.fit_care(X)
    mapping=f.semantic_mapping(X,fit['labels'],fit['modes'],f.MAIN_FEATURES,f.main_semantic_rules())
    labels,_=ck.assign_nearest(T,fit['modes'])
    ytr=f.mapping_to_targets(mapping,fit['labels']).apply(lambda c:c.map(f.pole_abbr_to_full))
    yte=f.mapping_to_targets(mapping,labels).apply(lambda c:c.map(f.pole_abbr_to_full))
    rows=[];predictions=test[['user']].rename(columns={'user':'learner_id'})
    for name in f.build_main_classifiers():
        for dim in f.DIM_NAMES:
            model=f.build_main_classifiers()[name]
            model.fit(train[f.MAIN_FEATURES].to_numpy(dtype=int),ytr[dim])
            pred=model.predict(test[f.MAIN_FEATURES].to_numpy(dtype=int))
            rows.append(dict(method=name,dimension=dim,**metrics(yte[dim],pred)))
            predictions[name+'_'+dim]=pred
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'main_lms_metrics.csv',index=False)
    predictions.to_csv(out/'main_lms_predictions.csv',index=False)
    audit.to_csv(out/'train_thresholds.csv',index=False)
    print(pd.DataFrame(rows).groupby('method')[['accuracy','balanced_accuracy','macro_f1']].mean())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--numeric',type=Path,required=True,help='Original numeric learner matrix; not contained in this release.')
    p.add_argument('--out',type=Path,default=ROOT/'OUTPUT/recomputed/main_holdout')
    a=p.parse_args();run(a.numeric,a.out)
