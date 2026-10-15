from pathlib import Path
from io import BytesIO
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from experiment_common import ROOT, REPRODUCED, CARE_CONFIG, K, MAX_ITER
from care_km import care_init,kmodes_fit,assign_nearest
FEATURES=['Course overview','Reading file','Abstract materiale','Concrete material','Visual Materials','Self-assessment','Exercises submit','Quiz submitted','playing','paused','unstarted','buffering']; TARGETS=['Processing','Perception','Input','Understanding']; SEED=161600
zpath=ROOT/'DATA/EXTERNAL_ILS/SOURCE/LS-IEEE-Dataport.zip'
def read(name):
    with zipfile.ZipFile(zpath) as z:
        hit=[n for n in z.namelist() if Path(n).name.lower()==name.lower()][0]; return pd.read_excel(BytesIO(z.read(hit)))
def clean(df): df=df.copy(); df.columns=[str(c).strip() for c in df.columns]; return df
cshs,csms=clean(read('CSHS.xlsx')),clean(read('CSMS.xlsx')); req=['id']+FEATURES+TARGETS
def select(df,n,cohort):
    idx=np.flatnonzero(df[TARGETS].notna().all(axis=1).to_numpy())[:n]; x=df.iloc[idx][req].copy(); x.insert(0,'source_row_index',idx); x.insert(0,'cohort',cohort); return x
ieee=pd.concat([select(cshs,480,'CSHS'),select(csms,256,'CSMS')],ignore_index=True); idx=np.arange(len(ieee)); tridx,teidx=train_test_split(idx,test_size=.30,random_state=SEED,shuffle=True,stratify=ieee.cohort); tr=ieee.iloc[tridx].reset_index(drop=True); te=ieee.iloc[teidx].reset_index(drop=True)
spec={f:sorted(set([float(pd.to_numeric(tr[f]).quantile(1/3)),float(pd.to_numeric(tr[f]).quantile(2/3))])) for f in FEATURES}
def cat(df):
    out=pd.DataFrame(index=df.index)
    for f in FEATURES: out[f]=np.digitize(pd.to_numeric(df[f]).to_numpy(float),np.asarray(spec[f]),right=True)+1
    return out.astype(int)
ctr,cte=cat(tr),cat(te); init=care_init(ctr.to_numpy(object),K,**CARE_CONFIG); res=kmodes_fit(ctr.to_numpy(object),init,max_iter=MAX_ITER); tc,dist=assign_nearest(cte.to_numpy(object),res['modes']); Xtr=ctr.copy(); Xte=cte.copy(); Xtr['CARE_cluster']=res['labels']+1; Xte['CARE_cluster']=tc+1
def ohe():
    try:return OneHotEncoder(handle_unknown='ignore',sparse_output=True)
    except TypeError:return OneHotEncoder(handle_unknown='ignore',sparse=True)
rows=[]; pred=[]
for target in TARGETS:
    ytr=tr[target].astype(str).str.strip().str.upper(); yte=te[target].astype(str).str.strip().str.upper(); models={'NB':CategoricalNB(alpha=1.0),'SVM':Pipeline([('onehot',ohe()),('model',SVC(kernel='rbf',C=1.0,gamma='scale'))]),'DT':Pipeline([('onehot',ohe()),('model',DecisionTreeClassifier(random_state=SEED))])}; pp={}
    models['NB'].fit(Xtr.astype(int),ytr); pp['NB']=models['NB'].predict(Xte.astype(int))
    for n in ['SVM','DT']: models[n].fit(Xtr,ytr); pp[n]=models[n].predict(Xte)
    for n,p in pp.items(): rows.append({'Target ILS':target,'Model':n,'Accuracy':accuracy_score(yte,p),'Balanced Accuracy':balanced_accuracy_score(yte,p),'Macro F1':f1_score(yte,p,average='macro')})
    q=pd.DataFrame({'cohort':te.cohort,'source_row_index':te.source_row_index,'reference_cluster_id':tc,'reference_cluster_name':[f'C{x+1:02d}' for x in tc],'distance_to_frozen_mode':dist,'Target ILS':target,'True/Reference':yte,'NB':pp['NB'],'SVM':pp['SVM'],'DT':pp['DT']}); pred.append(q)
pd.DataFrame(rows).to_csv(REPRODUCED/'external_ieee_prediction_metrics.csv',index=False); pd.concat(pred).to_csv(REPRODUCED/'external_ieee_test_predictions.csv',index=False)
