import argparse
import numpy as np
import pandas as pd
from experiment_common import ROOT, REPRODUCED, FEATURES, load_lms, sample_hash, CARE_CONFIG, K, MAX_ITER, SI_SAMPLE_SIZE
from care_km import care_init, kmodes_fit, evaluate_clustering
from fslsm import interpret

def primary_reference(df):
    a=pd.read_csv(ROOT/'OUTPUT/CLUSTERING/main_lms_full_assignments.csv',dtype={'record_id':str}); m=pd.read_csv(ROOT/'OUTPUT/INTERPRETATION/cluster_fslsm_profiles.csv')
    # direction columns are available in the full mapping rather than the compact paper table
    full=pd.read_csv(ROOT/'OUTPUT/PREDICTION/main_lms_train_cluster_to_fslsm_mapping.csv') if False else None
    modes=pd.read_csv(ROOT/'OUTPUT/CLUSTERING/main_lms_full_final_modes.csv',dtype=str)[FEATURES].to_numpy(object)
    labels=df[['record_id']].merge(a,on='record_id')['cluster_id'].to_numpy(int)
    ev=pd.read_csv(ROOT/'OUTPUT/INTERPRETATION/semantic_evidence_matrix.csv'); _,mp=interpret(df[FEATURES].to_numpy(object),modes,labels,FEATURES,ev)
    return labels,mp

def semantic_stability():
    df,X=load_lms(); full_labels,fullmap=primary_reference(df); ev=pd.read_csv(ROOT/'OUTPUT/INTERPRETATION/semantic_evidence_matrix.csv'); fullmap=fullmap.set_index('cluster_id')
    rows=[]
    for rep in range(30):
        seed=6161600+rep; rng=np.random.default_rng(seed); idx=np.sort(rng.choice(len(X),size=int(len(X)*.8),replace=False)); x=X[idx]
        init=care_init(x,K,**CARE_CONFIG); res=kmodes_fit(x,init,max_iter=MAX_ITER); met=evaluate_clustering(x,None,res,init,si_sample_size=SI_SAMPLE_SIZE,si_seed=seed)
        _,mp=interpret(x,res['modes'],res['labels'],FEATURES,ev); mp=mp.set_index('cluster_id')
        dims=['active_reflective','sensing_intuitive','visual_verbal','sequential_global']; agrees=[]
        for d in dims:
            ref=np.array([fullmap.loc[full_labels[i],d] for i in idx],object); cur=np.array([mp.loc[c,d] for c in res['labels']],object); agrees.append(float(np.mean(ref==cur)))
        rows.append({'rep':rep,'sample_hash':sample_hash(idx),'sample_n':len(idx),'final_cost':met['final_cost'],'dbi':met['dbi'],'si':met['si'],'iterations':met['iterations'],'active_reflective_agreement':agrees[0],'sensing_intuitive_agreement':agrees[1],'visual_verbal_agreement':agrees[2],'sequential_global_agreement':agrees[3],'mean_direction_agreement':float(np.mean(agrees))})
    pd.DataFrame(rows).to_csv(REPRODUCED/'semantic_stability_runs.csv',index=False)

def rule_sensitivity():
    df,X=load_lms(); modes=pd.read_csv(ROOT/'OUTPUT/CLUSTERING/main_lms_full_final_modes.csv',dtype=str)[FEATURES].to_numpy(object); a=pd.read_csv(ROOT/'OUTPUT/CLUSTERING/main_lms_full_assignments.csv'); labels=a.cluster_id.to_numpy(int); ev=pd.read_csv(ROOT/'OUTPUT/INTERPRETATION/semantic_evidence_matrix.csv')
    _,pri=interpret(X,modes,labels,FEATURES,ev,'PRIMARY_MODE_FIRST'); pri=pri.set_index('cluster_id'); rows=[]
    for rule in ['MEMBER_FIRST','MODE_ONLY','MEMBER_ONLY']:
        _,alt=interpret(X,modes,labels,FEATURES,ev,rule); alt=alt.set_index('cluster_id'); weights=pri.cluster_size.to_numpy(float)
        for d in ['active_reflective','sensing_intuitive','visual_verbal','sequential_global']:
            match=(pri[d].astype(str)==alt[d].astype(str)).to_numpy(); rows.append({'rule':rule,'dimension':d,'cluster_level_agreement':match.mean(),'learner_weighted_agreement':np.average(match.astype(float),weights=weights),'changed_clusters':int((~match).sum())})
    pd.DataFrame(rows).to_csv(REPRODUCED/'semantic_rule_sensitivity_dimension_summary.csv',index=False)

def preprocessing_sensitivity():
    numeric=pd.read_csv(ROOT/'DATA/MAIN_LMS/main_lms_numeric_6658x13.csv'); tables=pd.read_csv(ROOT/'OUTPUT/SENSITIVITY/preprocessing_sensitivity_thresholds.csv'); df0,_=load_lms(); full_labels,fullmap=primary_reference(df0); fullmap=fullmap.set_index('cluster_id'); ev=pd.read_csv(ROOT/'OUTPUT/INTERPRETATION/semantic_evidence_matrix.csv'); rows=[]
    for cfg,t in tables.groupby('configuration',sort=False):
        cat=pd.DataFrame({'record_id':numeric.record_id}); tt=t.set_index('feature')
        for f in FEATURES:
            x=pd.to_numeric(numeric[f]); lo=float(tt.loc[f,'applied_low']); hi=float(tt.loc[f,'applied_high']); method=str(tt.loc[f,'applied_method']); v=np.ones(len(x),int); v[x>0 if 'ZERO_AWARE_POSITIVE_MEDIAN' in method else x>=lo]=2; v[x>=hi]=3; cat[f]=v
        X=cat[FEATURES].astype(str).to_numpy(object); init=care_init(X,K,**CARE_CONFIG); res=kmodes_fit(X,init,max_iter=MAX_ITER); met=evaluate_clustering(X,None,res,init,si_sample_size=SI_SAMPLE_SIZE,si_seed=4161600); _,mp=interpret(X,res['modes'],res['labels'],FEATURES,ev); mp=mp.set_index('cluster_id')
        ag=[]
        for d in ['active_reflective','sensing_intuitive','visual_verbal','sequential_global']:
            ref=np.array([fullmap.loc[c,d] for c in full_labels],object); cur=np.array([mp.loc[c,d] for c in res['labels']],object); ag.append(float(np.mean(ref==cur)))
        rows.append({'configuration':cfg,'initial_cost':met['initial_cost'],'final_cost':met['final_cost'],'dbi':met['dbi'],'si':met['si'],'iterations':met['iterations'],'active_reflective_agreement':ag[0],'sensing_intuitive_agreement':ag[1],'visual_verbal_agreement':ag[2],'sequential_global_agreement':ag[3],'mean_dimensional_agreement':float(np.mean(ag))})
    pd.DataFrame(rows).to_csv(REPRODUCED/'preprocessing_sensitivity_summary.csv',index=False)

p=argparse.ArgumentParser(); p.add_argument('--analysis',choices=['semantic-stability','semantic-rules','preprocessing','all'],default='all'); a=p.parse_args()
if a.analysis in ('semantic-stability','all'): semantic_stability()
if a.analysis in ('semantic-rules','all'): rule_sensitivity()
if a.analysis in ('preprocessing','all'): preprocessing_sensitivity()
