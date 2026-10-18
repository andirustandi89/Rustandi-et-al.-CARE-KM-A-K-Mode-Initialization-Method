"""Recompute the questionnaire holdout from released training-fitted indicators."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
import care_km as ck
import fslsm as f

ROOT = Path(__file__).resolve().parents[1]

def metrics(truth, pred):
    poles = sorted(set(truth))
    return dict(accuracy=accuracy_score(truth, pred),
                balanced_accuracy=recall_score(truth, pred, labels=poles, average='macro', zero_division=0),
                macro_f1=f1_score(truth, pred, labels=poles, average='macro', zero_division=0))

def run(out):
    data=ROOT/'DATA/questionnaire_cohort'
    labels=pd.read_csv(data/'ils_labels_and_split.csv')
    tr=pd.read_csv(data/'behavior_train.csv').set_index('learner_id')
    te=pd.read_csv(data/'behavior_test.csv').set_index('learner_id')
    # Retain saved split membership and within-split order.
    a=labels.loc[labels.split.eq('train')].reset_index(drop=True)
    b=labels.loc[labels.split.eq('test')].reset_index(drop=True)
    tr=tr.loc[a.learner_id].reset_index();te=te.loc[b.learner_id].reset_index()
    assert len(tr)==322 and len(te)==139 and not set(tr.learner_id)&set(te.learner_id)
    features=[c for c in tr if c!='learner_id' and tr[c].nunique()>1]
    X=tr[features].astype(str).to_numpy(dtype=object)
    T=te[features].astype(str).to_numpy(dtype=object)
    init,fit=f.fit_care(X)
    rules=pd.read_csv(data/'evidence_rules.csv')
    rules=rules[rules.feature.isin(features)]
    mapping=f.semantic_mapping(X,fit['labels'],fit['modes'],features,rules)
    mp=mapping.set_index('cluster_id')
    test_cluster,distance=ck.assign_nearest(T,fit['modes'])
    train_ls=np.array([mp.loc[c,'ls_class'] for c in fit['labels']])
    resolved=train_ls!='AMBIGUOUS'
    assert resolved.sum()==281, f'Resolved training records: {resolved.sum()}'
    out.mkdir(parents=True,exist_ok=True)
    mapping.to_csv(out/'questionnaire_mapping.csv',index=False)
    pd.DataFrame(fit['modes'],columns=features).to_csv(out/'questionnaire_modes.csv',index=False)
    predictions=te[['learner_id']].copy()
    predictions['reference_cluster_id']=test_cluster
    predictions['distance_to_frozen_mode']=distance
    rows=[]
    for name,model in f.build_external_classifiers().items():
        model.fit(tr.loc[resolved,features].to_numpy(dtype=int),train_ls[resolved])
        cls=model.predict(te[features].to_numpy(dtype=int))
        for j,key in enumerate(f.DIM_KEYS):
            pred=np.array([f.LS_TO_POLES[x][j] for x in cls])
            truth=b[key+'_code'].map(f.ils_code_to_full).to_numpy()
            predictions[name+'_'+key]=pred
            rows.append(dict(method=name,dimension=key,**metrics(truth,pred)))
    for j,key in enumerate(f.DIM_KEYS):
        truth=b[key+'_code'].map(f.ils_code_to_full).to_numpy()
        pred=np.array([f.pole_abbr_to_full(mp.loc[c,key]) for c in test_cluster])
        predictions['CARE-KM_'+key]=pred
        rows.append(dict(method='CARE-KM',dimension=key,**metrics(truth,pred)))
        majority=pd.Series([f.LS_TO_POLES[x][j] for x in train_ls[resolved]]).mode().iloc[0]
        rows.append(dict(method='Majority',dimension=key,**metrics(truth,np.repeat(majority,len(te)))))
    result=pd.DataFrame(rows)
    result.to_csv(out/'questionnaire_dimension_metrics.csv',index=False)
    predictions.to_csv(out/'questionnaire_predictions.csv',index=False)
    summary=result.groupby('method')[['accuracy','balanced_accuracy','macro_f1']].mean()
    summary.to_csv(out/'questionnaire_mean_metrics.csv')
    print((100*summary).round(2).to_string())
    # Agreement with manuscript Table 11 at reported precision.
    expected={'NB':[83.27,69.63,72.00],'DT':[80.58,66.64,69.15],'SVM':[82.19,68.70,70.91],'CARE-KM':[79.32,66.75,69.50],'Majority':[75.72,50.00,42.98]}
    for method,value in expected.items():
        assert np.allclose(np.round(100*summary.loc[method].to_numpy(),2),value),(method,summary.loc[method].to_dict())
    print('Questionnaire accuracy checks passed; 322 training, 281 resolved, 139 test learners.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'OUTPUT/recomputed/questionnaire')
    run(parser.parse_args().out)
