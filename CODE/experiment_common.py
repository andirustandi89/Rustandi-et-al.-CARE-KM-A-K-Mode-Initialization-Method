from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "DATA"
PUBLISHED = ROOT / "OUTPUT"
REPRODUCED = ROOT / "REPRODUCED"
REPRODUCED.mkdir(exist_ok=True)

FEATURES = ['content_visit', 'content_stay', 'selfass_visit_proxy', 'selfass_stay_proxy', 'exercise_visit_proxy', 'exercise_stay_proxy', 'forum_visit', 'forum_stay', 'forum_post', 'quiz_stay_results_proxy', 'navigation_overview_visit', 'navigation_overview_stay', 'return_to_overview_rate']
K=16
MAX_ITER=100
SI_SAMPLE_SIZE=2000
METHODS=["Random","Huang","Cao","Bai","K-Modes++","HR-2023","CARE-KM"]
STOCHASTIC={"Random","Huang","K-Modes++","HR-2023"}
CARE_CONFIG={"candidate_rule":"separation","candidate_budget":16,"max_exchanges":4,"min_gain":0.0}

def load_lms():
    df=pd.read_csv(DATA/"MAIN_LMS"/"main_lms_categorical_6658x13.csv",dtype=str,keep_default_na=False)
    if list(df.columns)!=["record_id"]+FEATURES: raise RuntimeError("Unexpected main LMS schema")
    return df, df[FEATURES].to_numpy(dtype=object)

def sample_hash(idx):
    return hashlib.sha256(np.asarray(idx,dtype=np.int64).tobytes()).hexdigest()[:16]

def holm(p):
    p=np.asarray(p,float); order=np.argsort(p); out=np.empty(len(p)); running=0.0
    for rank,pos in enumerate(order):
        running=max(running,(len(p)-rank)*p[pos]); out[pos]=min(1.0,running)
    return out
