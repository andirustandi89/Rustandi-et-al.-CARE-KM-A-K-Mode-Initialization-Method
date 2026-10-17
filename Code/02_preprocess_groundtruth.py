from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"

# Fitur yang dipakai pada eksperimen ground-truth.
# navigation_skip tidak dipakai karena konstan pada data training.
THRESHOLDS = {
    "content_visit": (75.0, 100.0),
    "content_stay": (50.0, 75.0),
    "selfass_visit": (25.0, 75.0),
    "selfass_twice_wrong": (25.0, 50.0),
    "exercise_visit": (25.0, 75.0),
    "exercise_stay": (50.0, 75.0),
    "ques_detail": (50.0, 75.0),
    "ques_overview": (50.0, 75.0),
    "ques_facts": (50.0, 75.0),
    "ques_concepts": (50.0, 75.0),
    "ques_graphics": (50.0, 75.0),
    "ques_text": (50.0, 75.0),
    "ques_interpret": (50.0, 75.0),
    "ques_develop": (50.0, 75.0),
    "quiz_revisions": (20.0, 50.0),
    "quiz_stay_results": (30.0, 60.0),
    "navigation_overview_visit": (10.0, 20.0),
    "navigation_overview_stay": (50.0, 75.0),
}


def categorize(series, low, high):
    x = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    return np.where(x < low, 1, np.where(x < high, 2, 3)).astype("int8")


def main():
    source = DATA_DIR / "groundtruth_numeric.csv"
    target = DATA_DIR / "groundtruth_categorical.csv"

    numeric = pd.read_csv(source, dtype={"learner_id": str})
    missing = [f for f in THRESHOLDS if f not in numeric.columns]
    if missing:
        raise RuntimeError(f"Fitur ground-truth tidak lengkap: {missing}")

    out = pd.DataFrame({"learner_id": numeric["learner_id"].astype(str)})
    for feature, (low, high) in THRESHOLDS.items():
        out[feature] = categorize(numeric[feature], low, high)

    if target.exists():
        reference = pd.read_csv(target, dtype={"learner_id": str})
        features = list(THRESHOLDS)
        same_id = reference["learner_id"].astype(str).tolist() == out["learner_id"].astype(str).tolist()
        diff = int((reference[features].astype(int).to_numpy() != out[features].astype(int).to_numpy()).sum())
        same = same_id and diff == 0
        print(f"Kecocokan dengan file kategorikal yang tersedia: {'OK' if same else 'BERBEDA'}")
        if not same:
            raise RuntimeError(f"Hasil kategorisasi berbeda pada {diff} sel.")

    out.to_csv(target, index=False)
    print(f"Selesai: {len(out)} learner, {len(THRESHOLDS)} fitur -> {target.name}")


if __name__ == "__main__":
    main()
