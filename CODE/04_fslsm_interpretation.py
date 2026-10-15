import pandas as pd
import numpy as np
from experiment_common import ROOT, REPRODUCED, FEATURES, load_lms
from fslsm import interpret
df,X=load_lms()
modes=pd.read_csv(ROOT/'OUTPUT/CLUSTERING/main_lms_full_final_modes.csv',dtype=str)[FEATURES].to_numpy(dtype=object)
a=pd.read_csv(ROOT/'OUTPUT/CLUSTERING/main_lms_full_assignments.csv',dtype={'record_id':str})
labels=df[['record_id']].merge(a,on='record_id',validate='one_to_one')['cluster_id'].to_numpy(int)
evidence=pd.read_csv(ROOT/'OUTPUT/INTERPRETATION/semantic_evidence_matrix.csv')
detail,mapping=interpret(X,modes,labels,FEATURES,evidence)
detail.to_csv(REPRODUCED/'cluster_dimension_evidence.csv',index=False)
mapping.to_csv(REPRODUCED/'cluster_fslsm_mapping.csv',index=False)
print(mapping[['cluster_name','cluster_size','ls_class','preference_signature']].to_string(index=False))
