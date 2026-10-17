from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from carekm_core import assign_nearest


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
OUTPUT_FILE = ROOT / "Output" / "main_lms_holdout_predictions.csv"
SPLIT_SEED = 161600

FEATURES = [
    "content_visit", "content_stay", "selfass_visit_proxy", "selfass_stay_proxy",
    "exercise_visit_proxy", "exercise_stay_proxy", "forum_visit", "forum_stay",
    "forum_post", "quiz_stay_results_proxy", "navigation_overview_visit",
    "navigation_overview_stay", "return_to_overview_rate",
]
DIMENSIONS = [
    ("Active / Reflective", "active_reflective"),
    ("Sensing / Intuitive", "sensing_intuitive"),
    ("Visual / Verbal", "visual_verbal"),
    ("Sequential / Global", "sequential_global"),
]
ABBR_TO_FULL = {
    "Act.": "Active", "Ref.": "Reflective", "Sens.": "Sensing", "Int.": "Intuitive",
    "Vis.": "Visual", "Ver.": "Verbal", "Seq.": "Sequential", "Gl.": "Global",
}

FEATURE_PLAN = [
    ("content_visit", "GRAF_FIXED", 75.0, 100.0),
    ("content_stay", "EMPIRICAL", 50.0, 75.0),
    ("selfass_visit_proxy", "GRAF_FIXED", 25.0, 75.0),
    ("selfass_stay_proxy", "EMPIRICAL", 50.0, 75.0),
    ("exercise_visit_proxy", "GRAF_FIXED", 25.0, 75.0),
    ("exercise_stay_proxy", "EMPIRICAL", 50.0, 75.0),
    ("forum_visit", "GRAF_DEGENERACY", 7.0, 50.0),
    ("forum_stay", "GRAF_DEGENERACY", 5.0, 30.0),
    ("forum_post", "GRAF_SPARSE", 1.0, 10.0),
    ("quiz_stay_results_proxy", "GRAF_FIXED", 30.0, 60.0),
    ("navigation_overview_visit", "GRAF_DEGENERACY", 10.0, 20.0),
    ("navigation_overview_stay", "EMPIRICAL", 50.0, 75.0),
    ("return_to_overview_rate", "EMPIRICAL", np.nan, np.nan),
]


def encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def categorical_nb():
    try:
        return CategoricalNB(alpha=1.0, min_categories=4)
    except TypeError:
        return CategoricalNB(alpha=1.0)


def positive_quantiles(series):
    x = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    pos = x[x > 0]
    low = float(pos.quantile(1 / 3))
    high = float(pos.quantile(2 / 3))
    if high <= low:
        unique = np.sort(pos.unique())
        low = float(np.quantile(unique, 1 / 3))
        high = float(np.quantile(unique, 2 / 3))
    if high <= low:
        raise RuntimeError("Threshold train tidak dapat dipisahkan.")
    return low, high


def cut_three(series, low, high):
    x = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    return pd.Series(np.where(x < low, 1, np.where(x < high, 2, 3)), index=series.index, dtype="int8")


def fit_thresholds(train):
    state = {}
    for feature, mode, low, high in FEATURE_PLAN:
        x = train[feature]
        method = mode

        if mode == "EMPIRICAL":
            low, high = positive_quantiles(x)
        elif mode in {"GRAF_DEGENERACY", "GRAF_SPARSE"}:
            initial = cut_three(x, low, high)
            share = initial.value_counts(normalize=True).reindex([1, 2, 3], fill_value=0.0)
            degenerate = share.max() > 0.95 or (share == 0).any()
            if degenerate and mode == "GRAF_DEGENERACY":
                low, high = positive_quantiles(x)
                method = "GRAF_FALLBACK_POSITIVE_Q33_Q67"
            elif degenerate and mode == "GRAF_SPARSE":
                raw = pd.to_numeric(x, errors="coerce").fillna(0.0).astype(float)
                pos = raw[raw > 0]
                low = 0.0
                high = float(pos.median()) if len(pos) else 1.0
                method = "ZERO_AWARE_POSITIVE_MEDIAN"
        state[feature] = (float(low), float(high), method)
    return state


def apply_thresholds(frame, state):
    out = pd.DataFrame({"learner_id": frame["learner_id"].astype(str)})
    for feature in FEATURES:
        low, high, method = state[feature]
        raw = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0).astype(float)
        if method == "ZERO_AWARE_POSITIVE_MEDIAN":
            out[feature] = np.where(raw <= 0, 1, np.where(raw < high, 2, 3)).astype("int8")
        else:
            out[feature] = cut_three(raw, low, high)
    return out


def build_models():
    return {
        "NB": categorical_nb(),
        "DT": Pipeline([
            ("onehot", encoder()),
            ("model", DecisionTreeClassifier(random_state=SPLIT_SEED)),
        ]),
        "SVM": Pipeline([
            ("onehot", encoder()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]),
    }


def full_pole(value):
    return ABBR_TO_FULL.get(str(value), "Balanced")


def main():
    numeric = pd.read_csv(DATA_DIR / "main_lms_numeric.csv", dtype={"learner_id": str})
    split = pd.read_csv(DATA_DIR / "main_lms_split_indices.csv", dtype={"learner_id": str})

    by_id = numeric.set_index("learner_id")
    train_ids = split.loc[split["split"].eq("train"), "learner_id"].tolist()
    test_ids = split.loc[split["split"].eq("test"), "learner_id"].tolist()
    train_raw = by_id.loc[train_ids].reset_index()
    test_raw = by_id.loc[test_ids].reset_index()

    # Seluruh threshold empiris ditentukan dari train lalu dibekukan untuk test.
    state = fit_thresholds(train_raw)
    train = apply_thresholds(train_raw, state)
    test = apply_thresholds(test_raw, state)

    modes_df = pd.read_csv(DATA_DIR / "main_lms_frozen_modes.csv")
    modes_df = modes_df.sort_values("cluster_id")
    modes = modes_df[FEATURES].astype(str).to_numpy(dtype=object)
    mapping = pd.read_csv(DATA_DIR / "main_lms_frozen_semantic_mapping.csv").set_index("cluster_id")

    train_cluster, _ = assign_nearest(train[FEATURES].astype(str).to_numpy(dtype=object), modes)
    test_cluster, test_dist = assign_nearest(test[FEATURES].astype(str).to_numpy(dtype=object), modes)

    reference_train = pd.DataFrame(index=train.index)
    reference_test = pd.DataFrame(index=test.index)
    for title, key in DIMENSIONS:
        reference_train[title] = [full_pole(mapping.loc[int(c), key]) for c in train_cluster]
        reference_test[title] = [full_pole(mapping.loc[int(c), key]) for c in test_cluster]

    out = pd.DataFrame({
        "learner_id": test["learner_id"],
        "reference_cluster_id": test_cluster,
        "distance_to_frozen_mode": test_dist,
    })
    for title, key in DIMENSIONS:
        out[f"reference_{key}"] = reference_test[title]
    out["reference_ls_class"] = [mapping.loc[int(c), "ls_class"] for c in test_cluster]

    score_rows = []
    models = build_models()
    for model_name in ["NB", "DT", "SVM"]:
        accuracies = []
        for title, _ in DIMENSIONS:
            y_train = reference_train[title].astype(str)
            y_test = reference_test[title].astype(str)
            model = build_models()[model_name]

            if model_name == "NB":
                model.fit(train[FEATURES].astype(int).to_numpy(), y_train)
                pred = model.predict(test[FEATURES].astype(int).to_numpy())
            else:
                model.fit(train[FEATURES].astype(int), y_train)
                pred = model.predict(test[FEATURES].astype(int))

            out[f"{model_name}_{title}"] = pred
            accuracies.append(accuracy_score(y_test, pred))
        score_rows.append((model_name, accuracies))

    out.to_csv(OUTPUT_FILE, index=False)

    print("Prospective holdout accuracy:")
    for model_name, acc in score_rows:
        values = "  ".join(f"{100*x:.2f}%" for x in acc)
        print(f"{model_name:3s}: {values}  mean={100*np.mean(acc):.2f}%")
    print(f"\nTrain={len(train)}  Test={len(test)}")
    print(f"Disimpan ke: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
