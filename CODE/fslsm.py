"""FSLSM evidence rules, training-only categorization and classifier settings."""
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import care_km as ck

SPLIT_SEED = 161600

TEST_SIZE = 0.30

K = 16

MAX_ITER = 100

CARE_CONFIG = dict(candidate_rule='separation', candidate_budget=16, max_exchanges=4, min_gain=0.0)

MAIN_FEATURES = [
    'content_visit','content_stay','selfass_visit_proxy','selfass_stay_proxy',
    'exercise_visit_proxy','exercise_stay_proxy','forum_visit','forum_stay','forum_post',
    'quiz_stay_results_proxy','navigation_overview_visit','navigation_overview_stay','return_to_overview_rate'
]

DIM_NAMES = ['Active / Reflective','Sensing / Intuitive','Visual / Verbal','Sequential / Global']

DIM_KEYS = ['active_reflective','sensing_intuitive','visual_verbal','sequential_global']

POLES = {
    'Active/Reflective': ('Active','Reflective'),
    'Sensing/Intuitive': ('Sensing','Intuitive'),
    'Visual/Verbal': ('Visual','Verbal'),
    'Sequential/Global': ('Sequential','Global'),
}

ABBR = {
    'Active':'Act.','Reflective':'Ref.','Sensing':'Sens.','Intuitive':'Int.',
    'Visual':'Vis.','Verbal':'Ver.','Sequential':'Seq.','Global':'Gl.'
}

DIMENSION_ORDER = list(POLES.keys())

LS_LOOKUP = {tuple(combo): f'LS{i:02d}' for i, combo in enumerate(itertools.product(*[POLES[d] for d in DIMENSION_ORDER]), start=1)}

LS_TO_POLES = {v:k for k,v in LS_LOOKUP.items()}

def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=True)

def categorical_nb():
    try:
        return CategoricalNB(alpha=1.0, min_categories=4)
    except TypeError:
        return CategoricalNB(alpha=1.0)

def build_main_classifiers():
    # Prospective main-LMS configuration.
    return {
        'NB': categorical_nb(),
        'DT': Pipeline([('onehot', one_hot_encoder()), ('model', DecisionTreeClassifier(random_state=SPLIT_SEED))]),
        'SVM': Pipeline([('onehot', one_hot_encoder()), ('model', SVC(kernel='rbf', C=1.0, gamma='scale'))]),
    }

def build_external_classifiers():
    # Frozen configuration used by the locked ground-truth Table 2.
    return {
        'NB': categorical_nb(),
        'DT': DecisionTreeClassifier(random_state=SPLIT_SEED, class_weight='balanced'),
        'SVM': Pipeline([('onehot', one_hot_encoder()), ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'))]),
    }

def fit_care(X):
    X=np.asarray(X,dtype=object)
    init=ck.care_init(X,K,**CARE_CONFIG)
    result=ck.kmodes_fit(X,init,max_iter=MAX_ITER)
    return init,result

def signed_vote(value, rule, features):
    value=int(value)
    first,second=POLES[str(rule.dimension)]
    if value==2:
        return 0
    if hasattr(rule,'high_occurrence_indicates'):
        pole = str(rule.high_occurrence_indicates) if value==3 else str(rule.low_occurrence_indicates)
    else:
        pole = str(rule.high_supports) if value==3 else str(rule.low_supports)
    if pole==first: return 1
    if pole==second: return -1
    raise ValueError(f'Unknown semantic pole: {pole}')

def semantic_mapping(X, labels, modes, features, semantic_rules):
    X=np.asarray(X,dtype=object); labels=np.asarray(labels,dtype=int); modes=np.asarray(modes,dtype=object)
    fpos={f:i for i,f in enumerate(features)}
    rows=[]
    for cid in range(K):
        selected=[]; short=[]
        for dim in DIMENSION_ORDER:
            rules=semantic_rules[semantic_rules['dimension'].astype(str).eq(dim)]
            mv=np.asarray([signed_vote(modes[cid,fpos[str(r.feature)]], r, features) for r in rules.itertuples(index=False)],dtype=int)
            ms=int(np.sum(mv>0)-np.sum(mv<0))
            member_X=X[labels==cid]
            nv=[]
            for r in rules.itertuples(index=False):
                j=fpos[str(r.feature)]
                nv.extend(signed_vote(v,r,features) for v in member_X[:,j])
            nv=np.asarray(nv,dtype=int)
            ns=int(np.sum(nv>0)-np.sum(nv<0)) if len(nv) else 0
            first,second=POLES[dim]
            pole = first if ms>0 else second if ms<0 else first if ns>0 else second if ns<0 else None
            selected.append(pole)
            short.append('Balanced' if pole is None else ABBR[pole])
        unresolved=any(x is None for x in selected)
        rows.append({
            'cluster_id':cid,'cluster_name':f'C{cid+1:02d}',
            'cluster_size':int(np.sum(labels==cid)),
            'active_reflective':short[0], 'sensing_intuitive':short[1],
            'visual_verbal':short[2], 'sequential_global':short[3],
            'ls_class':'AMBIGUOUS' if unresolved else LS_LOOKUP[tuple(selected)],
            'mapping_status':'BALANCED' if unresolved else 'RESOLVED',
        })
    return pd.DataFrame(rows)

def main_semantic_rules():
    rows=[]
    def add(f,d,lo,hi): rows.append({'feature':f,'dimension':d,'low_supports':lo,'high_supports':hi})
    add('content_visit','Active/Reflective','Active','Reflective')
    add('content_stay','Active/Reflective','Active','Reflective')
    add('selfass_visit_proxy','Active/Reflective','Reflective','Active')
    add('selfass_stay_proxy','Active/Reflective','Active','Reflective')
    add('exercise_visit_proxy','Active/Reflective','Reflective','Active')
    add('exercise_stay_proxy','Active/Reflective','Reflective','Active')
    add('forum_visit','Active/Reflective','Active','Reflective')
    add('forum_post','Active/Reflective','Reflective','Active')
    add('quiz_stay_results_proxy','Active/Reflective','Active','Reflective')
    add('content_visit','Sensing/Intuitive','Sensing','Intuitive')
    add('content_stay','Sensing/Intuitive','Sensing','Intuitive')
    add('selfass_visit_proxy','Sensing/Intuitive','Intuitive','Sensing')
    add('selfass_stay_proxy','Sensing/Intuitive','Intuitive','Sensing')
    add('exercise_visit_proxy','Sensing/Intuitive','Intuitive','Sensing')
    add('quiz_stay_results_proxy','Sensing/Intuitive','Intuitive','Sensing')
    add('content_visit','Visual/Verbal','Visual','Verbal')
    add('forum_visit','Visual/Verbal','Visual','Verbal')
    add('forum_stay','Visual/Verbal','Visual','Verbal')
    add('forum_post','Visual/Verbal','Visual','Verbal')
    add('navigation_overview_visit','Sequential/Global','Sequential','Global')
    add('navigation_overview_stay','Sequential/Global','Sequential','Global')
    add('return_to_overview_rate','Sequential/Global','Sequential','Global')
    return pd.DataFrame(rows)

def mapping_to_targets(mapping, cluster_ids):
    mp=mapping.set_index('cluster_id')
    target=pd.DataFrame(index=np.arange(len(cluster_ids)))
    rename={
        'active_reflective':'Active / Reflective',
        'sensing_intuitive':'Sensing / Intuitive',
        'visual_verbal':'Visual / Verbal',
        'sequential_global':'Sequential / Global',
    }
    for col,new in rename.items():
        target[new]=[mp.loc[int(c),col] for c in cluster_ids]
    return target

def pole_abbr_to_full(x):
    m={'Act.':'Active','Ref.':'Reflective','Sens.':'Sensing','Int.':'Intuitive','Vis.':'Visual','Ver.':'Verbal','Seq.':'Sequential','Gl.':'Global'}
    return m.get(str(x), 'Balanced')

def ils_code_to_full(code):
    s=str(code)
    for p,full in [('Act.','Active'),('Ref.','Reflective'),('Sens.','Sensing'),('Int.','Intuitive'),('Vis.','Visual'),('Ver.','Verbal'),('Seq.','Sequential'),('Gl.','Global')]:
        if s.startswith(p): return full
    raise ValueError(f'Unexpected ILS code: {code}')

DEGENERATE_MAX_SHARE=0.95

FEATURE_PLAN = [
    {'feature':'content_visit','mode':'GRAF_FIXED','low':75.0,'high':100.0},
    {'feature':'content_stay','mode':'EMPIRICAL','low':50.0,'high':75.0},
    {'feature':'selfass_visit_proxy','mode':'GRAF_FIXED','low':25.0,'high':75.0},
    {'feature':'selfass_stay_proxy','mode':'EMPIRICAL','low':50.0,'high':75.0},
    {'feature':'exercise_visit_proxy','mode':'GRAF_FIXED','low':25.0,'high':75.0},
    {'feature':'exercise_stay_proxy','mode':'EMPIRICAL','low':50.0,'high':75.0},
    {'feature':'forum_visit','mode':'GRAF_DEGENERACY','low':7.0,'high':50.0},
    {'feature':'forum_stay','mode':'GRAF_DEGENERACY','low':5.0,'high':30.0},
    {'feature':'forum_post','mode':'GRAF_SPARSE','low':1.0,'high':10.0},
    {'feature':'quiz_stay_results_proxy','mode':'GRAF_FIXED','low':30.0,'high':60.0},
    {'feature':'navigation_overview_visit','mode':'GRAF_DEGENERACY','low':10.0,'high':20.0},
    {'feature':'navigation_overview_stay','mode':'EMPIRICAL','low':50.0,'high':75.0},
    {'feature':'return_to_overview_rate','mode':'EMPIRICAL','low':np.nan,'high':np.nan},
]

def classify_cut(x,lo,hi):
    x=pd.to_numeric(x,errors='coerce').fillna(0).astype(float)
    return pd.Series(np.where(x<lo,1,np.where(x<hi,2,3)),index=x.index,dtype='int8')

def positive_q(x):
    x=pd.to_numeric(x,errors='coerce').fillna(0).astype(float); pos=x[x>0]
    if pos.empty: raise RuntimeError('Positive quantile unavailable.')
    lo=float(pos.quantile(1/3)); hi=float(pos.quantile(2/3))
    if hi<=lo:
        u=np.sort(pos.unique()); lo=float(np.quantile(u,1/3)); hi=float(np.quantile(u,2/3))
    if hi<=lo: raise RuntimeError('Degenerate train threshold.')
    return lo,hi

def share123(c): return c.value_counts(normalize=True).reindex([1,2,3],fill_value=0.0)

def fit_main_categorizer(train):
    state={}; audit=[]
    for sp in FEATURE_PLAN:
        f=sp['feature']; mode=sp['mode']; x=train[f]; lo=sp['low']; hi=sp['high']; method=mode
        if mode=='EMPIRICAL': lo,hi=positive_q(x)
        elif mode=='GRAF_DEGENERACY':
            c0=classify_cut(x,lo,hi); sh=share123(c0)
            if sh.max()>DEGENERATE_MAX_SHARE or (sh==0).any(): lo,hi=positive_q(x); method='GRAF_FALLBACK_POSITIVE_Q33_Q67'
        elif mode=='GRAF_SPARSE':
            c0=classify_cut(x,lo,hi); sh=share123(c0)
            if sh.max()>DEGENERATE_MAX_SHARE or (sh==0).any():
                pos=pd.to_numeric(x,errors='coerce').fillna(0); pos=pos[pos>0]
                lo=0.0; hi=float(pos.median()) if len(pos) else 1.0; method='ZERO_AWARE_POSITIVE_MEDIAN'
        state[f]={'low':float(lo),'high':float(hi),'method':method}
        audit.append({'feature':f,**state[f]})
    return state,pd.DataFrame(audit)

def apply_main_categorizer(df,state):
    out=pd.DataFrame({'user':df['user'].astype(str)})
    for f in MAIN_FEATURES:
        st=state[f]; x=pd.to_numeric(df[f],errors='coerce').fillna(0).astype(float)
        if st['method']=='ZERO_AWARE_POSITIVE_MEDIAN':
            out[f]=np.where(x<=0,1,np.where(x<st['high'],2,3)).astype(int)
        else:
            out[f]=classify_cut(x,st['low'],st['high']).astype(int)
    return out
