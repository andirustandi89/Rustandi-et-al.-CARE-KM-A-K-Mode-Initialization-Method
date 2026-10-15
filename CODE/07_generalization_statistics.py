import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from experiment_common import ROOT, REPRODUCED, holm
raw=pd.read_csv(ROOT/'OUTPUT/GENERALIZATION/generalization_resample_runs.csv'); methods=['Random','Huang','Cao','Bai','K-Modes++','HR-2023','CARE-KM']; metrics=[('initial_cost','lower'),('final_cost','lower'),('dbi','lower'),('si','higher'),('iterations','lower'),('total_sec','lower')]
# dataset-level means are the units for cross-dataset inference
mean=raw.groupby(['dataset','method'])[[m for m,_ in metrics]].mean().reset_index(); frows=[]; prows=[]
for metric,direction in metrics:
    p=mean.pivot(index='dataset',columns='method',values=metric).dropna(); stat,pv=friedmanchisquare(*[p[m].to_numpy() for m in methods]); frows.append({'metric':metric,'friedman_chi2':stat,'p_value':pv,'kendall_w':stat/(len(p)*(len(methods)-1)),'n_datasets':len(p),'n_methods':len(methods)})
    temp=[]
    for b in [m for m in methods if m!='CARE-KM']:
        care=p['CARE-KM'].to_numpy(); base=p[b].to_numpy(); diff=(base-care) if direction=='lower' else (care-base); nz=diff[diff!=0]; st,praw=(0.0,1.0) if len(nz)==0 else wilcoxon(nz,alternative='greater'); temp.append({'metric':metric,'direction':direction,'baseline':b,'n_datasets':len(diff),'wins':int((diff>0).sum()),'ties':int((diff==0).sum()),'losses':int((diff<0).sum()),'mean_improvement':diff.mean(),'median_improvement':np.median(diff),'wilcoxon_stat':st,'p_value':praw,'rank_biserial':((diff>0).sum()-(diff<0).sum())/len(diff)})
    adj=holm([x['p_value'] for x in temp]);
    for x,a in zip(temp,adj): x['holm_p']=a; x['significant_0_05']=a<.05; prows.append(x)
pd.DataFrame(frows).to_csv(REPRODUCED/'friedman_kendall.csv',index=False); pd.DataFrame(prows).to_csv(REPRODUCED/'pairwise_wilcoxon_holm.csv',index=False)
