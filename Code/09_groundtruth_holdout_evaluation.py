from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from carekm_core import assign_nearest


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
OUTPUT_FILE = ROOT / "Output" / "groundtruth_holdout_predictions.csv"
SPLIT_SEED = 161600

FEATURES = [
    "content_visit", "content_stay", "selfass_visit", "selfass_twice_wrong",
    "exercise_visit", "exercise_stay", "ques_detail", "ques_overview", "ques_facts",
    "ques_concepts", "ques_graphics", "ques_text", "ques_interpret", "ques_develop",
    "quiz_revisions", "quiz_stay_results", "navigation_overview_visit", "navigation_overview_stay",
]

DIMENSIONS = [
    ("active_reflective", ("Active", "Reflective")),
    ("sensing_intuitive", ("Sensing", "Intuitive")),
    ("visual_verbal", ("Visual", "Verbal")),
    ("sequential_global", ("Sequential", "Global")),
]
LS_TO_POLES = {
    f"LS{i:02d}": tuple(combo)
    for i, combo in enumerate(itertools.product(*[poles for _, poles in DIMENSIONS]), start=1)
}
ABBR_TO_FULL = {
    "Act.": "Active", "Ref.": "Reflective", "Sens.": "Sensing", "Int.": "Intuitive",
    "Vis.": "Visual", "Ver.": "Verbal", "Seq.": "Sequential", "Gl.": "Global",
}


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


def build_models():
    return {
        "NB": categorical_nb(),
        "DT": DecisionTreeClassifier(random_state=SPLIT_SEED, class_weight="balanced"),
        "SVM": Pipeline([
            ("onehot", encoder()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
        ]),
    }


def full_pole(value):
    return ABBR_TO_FULL.get(str(value), "Balanced")


def main():
    data = pd.read_csv(DATA_DIR / "groundtruth_categorical.csv", dtype={"learner_id": str})
    labels = pd.read_csv(DATA_DIR / "groundtruth_ils_labels.csv", dtype={"learner_id": str})
    split = pd.read_csv(DATA_DIR / "groundtruth_split_indices.csv", dtype={"learner_id": str})

    by_id = data.set_index("learner_id")
    label_by_id = labels.set_index("learner_id")
    train_ids = split.loc[split["split"].eq("train"), "learner_id"].tolist()
    test_ids = split.loc[split["split"].eq("test"), "learner_id"].tolist()
    train = by_id.loc[train_ids].reset_index()
    test = by_id.loc[test_ids].reset_index()
    truth = label_by_id.loc[test_ids].reset_index()

    modes_df = pd.read_csv(DATA_DIR / "groundtruth_frozen_modes.csv").sort_values("cluster_id")
    modes = modes_df[FEATURES].astype(str).to_numpy(dtype=object)
    mapping = pd.read_csv(DATA_DIR / "groundtruth_frozen_semantic_mapping.csv").set_index("cluster_id")

    train_cluster, _ = assign_nearest(train[FEATURES].astype(str).to_numpy(dtype=object), modes)
    test_cluster, test_dist = assign_nearest(test[FEATURES].astype(str).to_numpy(dtype=object), modes)
    train_ls = np.asarray([mapping.loc[int(c), "ls_class"] for c in train_cluster], dtype=object)

    resolved = train_ls != "AMBIGUOUS"
    if not resolved.any():
        raise RuntimeError("Tidak ada label training yang resolved.")

    out = pd.DataFrame({"learner_id": test_ids})
    for dim, _ in DIMENSIONS:
        out[f"ils_{dim}"] = truth[dim].to_numpy()
    out["ils_ls_class"] = truth["ils_ls_class"].to_numpy()
    out["reference_cluster_id"] = test_cluster
    out["distance_to_frozen_mode"] = test_dist

    for dim, _ in DIMENSIONS:
        out[f"carekm_{dim}"] = [full_pole(mapping.loc[int(c), dim]) for c in test_cluster]
    out["carekm_ls_class"] = [mapping.loc[int(c), "ls_class"] for c in test_cluster]

    # Direct CARE-KM agreement against the independent ILS criterion.
    care_acc = []
    for dim, _ in DIMENSIONS:
        care_acc.append(accuracy_score(out[f"ils_{dim}"], out[f"carekm_{dim}"]))

    score_rows = []
    x_train = train.loc[resolved, FEATURES].astype(int).reset_index(drop=True)
    y_train = pd.Series(train_ls[resolved], dtype=str)
    x_test = test[FEATURES].astype(int).reset_index(drop=True)

    for model_name in ["NB", "DT", "SVM"]:
        model = build_models()[model_name]
        if model_name == "NB":
            model.fit(x_train.to_numpy(), y_train)
            pred_ls = model.predict(x_test.to_numpy())
        else:
            model.fit(x_train, y_train)
            pred_ls = model.predict(x_test)

        out[f"{model_name}_predicted_ls_class"] = pred_ls
        dim_acc = []
        bal_acc = []
        macro_f1 = []
        for i, (dim, _) in enumerate(DIMENSIONS):
            pred_dim = np.asarray([LS_TO_POLES[str(ls)][i] for ls in pred_ls], dtype=object)
            out[f"{model_name}_{dim}"] = pred_dim
            y_true = out[f"ils_{dim}"].to_numpy()
            dim_acc.append(accuracy_score(y_true, pred_dim))
            bal_acc.append(balanced_accuracy_score(y_true, pred_dim))
            macro_f1.append(f1_score(y_true, pred_dim, average="macro"))
        score_rows.append((model_name, dim_acc, bal_acc, macro_f1))

    out.to_csv(OUTPUT_FILE, index=False)

    print("CARE-KM agreement vs ILS:")
    print("  " + "  ".join(f"{100*x:.2f}%" for x in care_acc) + f"  mean={np.mean([round(100*x,2) for x in care_acc]):.2f}%")
    print("\nClassifier dimensional accuracy vs ILS:")
    for model_name, acc, bal, f1 in score_rows:
        mean_pct = np.mean([round(100*x, 2) for x in acc])
        print(f"{model_name:3s}: " + "  ".join(f"{100*x:.2f}%" for x in acc) + f"  mean={mean_pct:.2f}%")
        print(f"     balanced accuracy mean={100*np.mean(bal):.2f}%  macro-F1 mean={100*np.mean(f1):.2f}%")

    print(f"\nTrain={len(train)}  Test={len(test)}")
    print(f"Disimpan ke: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
