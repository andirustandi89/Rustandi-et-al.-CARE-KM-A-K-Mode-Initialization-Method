import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from experiment_common import ROOT, REPRODUCED, FEATURES, CARE_CONFIG, K, MAX_ITER
from care_km import care_init, kmodes_fit, assign_nearest
from fslsm import interpret
TARGETS=['Processing','Perception','Input','Understanding']; SPLIT_SEED=161600
df=pd.read_csv(ROOT/'DATA/MAIN_LMS/main_lms_categorical_6658x13.csv',dtype=str); sp=pd.read_csv(ROOT/'SPLITS/main_lms_70_30_split.csv',dtype={'record_id':str}); ordered=sp[['record_id','split']].merge(df,on='record_id',how='left',validate='one_to_one'); tr=ordered[ordered.split.eq('train')].reset_index(drop=True); te=ordered[ordered.split.eq('test')].reset_index(drop=True)
Xtr=tr[FEATURES].to_numpy(object); init=care_init(Xtr,K,**CARE_CONFIG); res=kmodes_fit(Xtr,init,max_iter=MAX_ITER); ev=pd.read_csv(ROOT/'OUTPUT/INTERPRETATION/semantic_evidence_matrix.csv'); _,mp=interpret(Xtr,res['modes'],res['labels'],FEATURES,ev); mp=mp.set_index('cluster_id'); tcl,dist=assign_nearest(te[FEATURES].to_numpy(object),res['modes'])
def targets(labels):
    return pd.DataFrame({t:[mp.loc[int(c),d] for c in labels] for t,d in zip(TARGETS,['active_reflective','sensing_intuitive','visual_verbal','sequential_global'])}).replace({'Act.':'ACT','Ref.':'REF','Sens.':'SEN','Int.':'INT','Vis.':'VIS','Ver.':'VRB','Seq.':'SEQ','Gl.':'GLO'})
ytr=targets(res['labels']); yte=targets(tcl)
def ohe():
    try:return OneHotEncoder(handle_unknown='ignore',sparse_output=True)
    except TypeError:return OneHotEncoder(handle_unknown='ignore',sparse=True)
rows=[]; preds=[]
for target in TARGETS:
    models={'NB':CategoricalNB(alpha=1.0),'SVM':Pipeline([('onehot',ohe()),('model',SVC(kernel='rbf',C=1.0,gamma='scale'))]),'DT':Pipeline([('onehot',ohe()),('model',DecisionTreeClassifier(random_state=SPLIT_SEED))])}
    pp={}; models['NB'].fit(tr[FEATURES].astype(int),ytr[target]); pp['NB']=models['NB'].predict(te[FEATURES].astype(int))
    for n in ['SVM','DT']: models[n].fit(tr[FEATURES],ytr[target]); pp[n]=models[n].predict(te[FEATURES])
    for n,p in pp.items(): rows.append({'Target ILS':target,'Model':n,'Accuracy':accuracy_score(yte[target],p),'Balanced Accuracy':balanced_accuracy_score(yte[target],p),'Macro F1':f1_score(yte[target],p,average='macro')})
    q=pd.DataFrame({'record_id':te.record_id,'reference_cluster_id':tcl,'reference_cluster_name':[f'C{x+1:02d}' for x in tcl],'Target ILS':target,'True/Reference':yte[target],'NB':pp['NB'],'SVM':pp['SVM'],'DT':pp['DT']}); preds.append(q)
pd.DataFrame(rows).to_csv(REPRODUCED/'main_lms_prediction_metrics.csv',index=False); pd.concat(preds).to_csv(REPRODUCED/'main_lms_test_predictions.csv',index=False)
