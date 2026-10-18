"""Recompute four-dimensional profiles of the fixed full-data LMS partition."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import fslsm as f
ROOT=Path(__file__).resolve().parents[1]
CODES={'Active':'Ac','Reflective':'Re','Sensing':'Se','Intuitive':'In','Visual':'Vi','Verbal':'Ve','Sequential':'Seq','Global':'Gl'}

def run(out):
    data=pd.read_csv(ROOT/'DATA/main_lms/behavior_categorical.csv')
    modes=pd.read_csv(ROOT/'OUTPUT/experiments/04_carekm_reference_final_modes.csv')[f.MAIN_FEATURES]
    member=pd.read_csv(ROOT/'OUTPUT/experiments/04_carekm_reference_labels.csv')
    d=data.merge(member,on='learner_id',validate='one_to_one')
    rules=f.main_semantic_rules();profiles=[];evidence=[]
    for cid in range(16):
        g=d[d.cluster_id.eq(cid)];row={'cluster':f'C{cid+1:02d}','n':len(g)};chosen=[];mags=[]
        for j,dim in enumerate(f.DIMENSION_ORDER):
            r=rules[rules.dimension.eq(dim)]
            mv=np.array([f.signed_vote(modes.loc[cid,x.feature],x,f.MAIN_FEATURES) for x in r.itertuples()])
            votes=np.array([f.signed_vote(v,x,f.MAIN_FEATURES) for x in r.itertuples() for v in g[x.feature]])
            first,second=f.POLES[dim];mode_balance=int(mv.sum());member_balance=int(votes.sum())
            balance=mode_balance if mode_balance else member_balance
            pole=first if balance>0 else second if balance<0 else None
            a=int((votes>0).sum());b=int((votes<0).sum());neutral=int((votes==0).sum())
            p=(a+.5*neutral)/len(votes)
            units=int(np.floor(11*p+.5));mag=abs(2*units-11)
            row[f.DIM_KEYS[j]]='Unresolved' if pole is None else CODES[pole]+str(mag)
            chosen.append(pole);mags.append(mag)
            evidence.append(dict(cluster_id=cid,dimension=dim,mode_balance=mode_balance,member_balance=member_balance,first_count=a,second_count=b,neutral_count=neutral,direction=pole,magnitude=mag))
        row['base_ls']='Unresolved' if None in chosen else f.LS_LOOKUP[tuple(chosen)]
        row['mean_behavioral_preference_strength']=sum(mags)/4
        profiles.append(row)
    out.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(profiles);result.to_csv(out/'full_lms_profiles.csv',index=False)
    pd.DataFrame(evidence).to_csv(out/'full_lms_evidence.csv',index=False)
    expected=pd.read_csv(ROOT/'OUTPUT/manuscript_tables/table_09_fslsm_profiles.csv')
    assert (result[f.DIM_KEYS].to_numpy()==expected[['Ac/Re','Se/In','Vi/Ve','Seq/Gl']].to_numpy()).all()
    assert result.base_ls.nunique()==9
    print('All 16 profiles match manuscript Table 9; nine Base LS combinations.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'OUTPUT/recomputed/profiles')
    run(p.parse_args().out)
