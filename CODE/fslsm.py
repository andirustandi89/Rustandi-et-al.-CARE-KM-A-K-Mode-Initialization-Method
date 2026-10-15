import itertools
import numpy as np
import pandas as pd
DIMENSIONS=[('Active/Reflective',('Active','Reflective')),('Sensing/Intuitive',('Sensing','Intuitive')),('Visual/Verbal',('Visual','Verbal')),('Sequential/Global',('Sequential','Global'))]
ABBR={'Active':'Act.','Reflective':'Ref.','Sensing':'Sens.','Intuitive':'Int.','Visual':'Vis.','Verbal':'Ver.','Sequential':'Seq.','Global':'Gl.'}
DIM_KEYS={'Active/Reflective':'active_reflective','Sensing/Intuitive':'sensing_intuitive','Visual/Verbal':'visual_verbal','Sequential/Global':'sequential_global'}
LS_LOOKUP={tuple(c):f'LS{i:02d}' for i,c in enumerate(itertools.product(*[p for _,p in DIMENSIONS]),1)}
ALLOWED={1,3,5,7,9,11}

def signed_vote(value, rule):
    first,second=dict(DIMENSIONS)[rule.dimension]; value=int(value)
    if value==2:return 0
    pole=rule.high_supports if value==3 else rule.low_supports
    return 1 if pole==first else -1 if pole==second else (_ for _ in ()).throw(ValueError(pole))

def score11(pos,neg,neutral):
    total=pos+neg+neutral
    if total<=0:return (np.nan,np.nan,'Unresolved')
    pu=int(np.floor(11*(pos+.5*neutral)/total+.5)); pu=max(0,min(11,pu)); raw=2*pu-11; mag=abs(raw)
    if mag not in ALLOWED: raise RuntimeError(raw)
    cat='Balanced' if mag in (1,3) else 'Moderate preference' if mag in (5,7) else 'Strong preference'
    return raw,mag,cat

def interpret(X,modes,labels,features,evidence,rule='PRIMARY_MODE_FIRST'):
    X=np.asarray(X,dtype=object); modes=np.asarray(modes,dtype=object); labels=np.asarray(labels,int); pos={f:i for i,f in enumerate(features)}; sizes=np.bincount(labels,minlength=len(modes))
    detail=[]; maps=[]
    for cid in range(len(modes)):
        selected=[]; abbr=[]; scores=[]; cats=[]; codes=[]; sources=[]
        for dim,poles in DIMENSIONS:
            rules=evidence[evidence.dimension.eq(dim)]; mv=np.array([signed_vote(modes[cid,pos[r.feature]],r) for r in rules.itertuples(index=False)],int); ms=int((mv>0).sum()-(mv<0).sum())
            member=X[labels==cid]; nv=[]
            for r in rules.itertuples(index=False): nv.extend(signed_vote(v,r) for v in member[:,pos[r.feature]])
            nv=np.asarray(nv,int); ns=int((nv>0).sum()-(nv<0).sum()); p,n,z=int((nv>0).sum()),int((nv<0).sum()),int((nv==0).sum())
            first,second=poles
            if rule=='PRIMARY_MODE_FIRST': vals=[(ms,'MODE'),(ns,'MEMBER')]
            elif rule=='MEMBER_FIRST': vals=[(ns,'MEMBER'),(ms,'MODE')]
            elif rule=='MODE_ONLY': vals=[(ms,'MODE')]
            elif rule=='MEMBER_ONLY': vals=[(ns,'MEMBER')]
            else: raise ValueError(rule)
            pole=None; src='BALANCED'
            for val,src0 in vals:
                if val>0: pole=first; src=src0; break
                if val<0: pole=second; src=src0; break
            raw,mag,cat=score11(p,n,z); s=np.nan if pole is None else (mag if pole==first else -mag); code='Balanced' if pole is None else f'{ABBR[pole]}{mag}'
            selected.append(pole); abbr.append('Balanced' if pole is None else ABBR[pole]); scores.append(s); cats.append(cat); codes.append(code); sources.append(src)
            detail.append({'cluster_id':cid,'cluster_name':f'C{cid+1:02d}','cluster_size':int(sizes[cid]),'dimension':dim,'selected_pole':abbr[-1],'decision_source':src,'mode_score':ms,'member_score':ns,'member_positive':p,'member_negative':n,'member_neutral':z,'preference_11eq_score':s,'preference_11eq_magnitude':mag,'preference_category':cat,'preference_code':code})
        unresolved=any(x is None for x in selected); ls='AMBIGUOUS' if unresolved else LS_LOOKUP[tuple(selected)]
        maps.append({'cluster_id':cid,'cluster_name':f'C{cid+1:02d}','cluster_size':int(sizes[cid]),'active_reflective':abbr[0],'sensing_intuitive':abbr[1],'visual_verbal':abbr[2],'sequential_global':abbr[3],'active_reflective_score':scores[0],'sensing_intuitive_score':scores[1],'visual_verbal_score':scores[2],'sequential_global_score':scores[3],'active_reflective_code':codes[0],'sensing_intuitive_code':codes[1],'visual_verbal_code':codes[2],'sequential_global_code':codes[3],'ls_class':ls,'semantic_label':'-'.join(abbr),'preference_signature':'|'.join(codes),'mean_behavioral_preference_strength':float(np.nanmean(np.abs(scores))) if not unresolved else np.nan,'mapping_status':'BALANCED' if unresolved else 'RESOLVED'})
    return pd.DataFrame(detail),pd.DataFrame(maps)
