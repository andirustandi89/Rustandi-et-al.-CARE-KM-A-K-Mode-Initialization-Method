from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"

FEATURES = [
    "content_visit",
    "content_stay",
    "selfass_visit_proxy",
    "selfass_stay_proxy",
    "exercise_visit_proxy",
    "exercise_stay_proxy",
    "forum_visit",
    "forum_stay",
    "forum_post",
    "quiz_stay_results_proxy",
    "navigation_overview_visit",
    "navigation_overview_stay",
    "return_to_overview_rate",
]

# Aturan utama yang dipakai untuk membentuk representasi 1/2/3.
# Threshold empiris dihitung hanya dari nilai positif pada fitur terkait.
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

DEGENERATE_MAX_SHARE = 0.95


def positive_quantiles(series):
    x = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    pos = x[x > 0]
    if pos.empty:
        raise RuntimeError("Tidak ada nilai positif untuk menentukan threshold empiris.")

    low = float(pos.quantile(1 / 3))
    high = float(pos.quantile(2 / 3))
    if high <= low:
        unique = np.sort(pos.unique())
        low = float(np.quantile(unique, 1 / 3))
        high = float(np.quantile(unique, 2 / 3))
    if high <= low:
        raise RuntimeError("Threshold empiris tidak dapat dipisahkan menjadi tiga kategori.")
    return low, high


def cut_three(series, low, high):
    x = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    return pd.Series(
        np.where(x < low, 1, np.where(x < high, 2, 3)),
        index=series.index,
        dtype="int8",
    )


def category_share(values):
    return values.value_counts(normalize=True).reindex([1, 2, 3], fill_value=0.0)


def build_categorical(numeric):
    out = pd.DataFrame({"learner_id": numeric["learner_id"].astype(str)})

    for feature, mode, low, high in FEATURE_PLAN:
        x = numeric[feature]

        if mode == "EMPIRICAL":
            low, high = positive_quantiles(x)
            out[feature] = cut_three(x, low, high)
            continue

        initial = cut_three(x, low, high)
        share = category_share(initial)
        degenerate = share.max() > DEGENERATE_MAX_SHARE or (share == 0).any()

        if mode == "GRAF_DEGENERACY" and degenerate:
            low, high = positive_quantiles(x)
            out[feature] = cut_three(x, low, high)
        elif mode == "GRAF_SPARSE" and degenerate:
            raw = pd.to_numeric(x, errors="coerce").fillna(0.0).astype(float)
            pos = raw[raw > 0]
            median = float(pos.median()) if len(pos) else 1.0
            out[feature] = np.where(raw <= 0, 1, np.where(raw < median, 2, 3)).astype("int8")
        else:
            out[feature] = initial

    return out


def main():
    source = DATA_DIR / "main_lms_numeric.csv"
    target = DATA_DIR / "main_lms_categorical.csv"

    numeric = pd.read_csv(source, dtype={"learner_id": str})
    missing = [c for c in ["learner_id"] + FEATURES if c not in numeric.columns]
    if missing:
        raise RuntimeError(f"Kolom tidak lengkap: {missing}")

    categorical = build_categorical(numeric)

    # Jika file referensi sudah ada, cek dulu sebelum ditulis ulang.
    if target.exists():
        reference = pd.read_csv(target, dtype={"learner_id": str})
        same_id = reference["learner_id"].astype(str).tolist() == categorical["learner_id"].astype(str).tolist()
        diff = int((reference[FEATURES].astype(int).to_numpy() != categorical[FEATURES].astype(int).to_numpy()).sum())
        same = same_id and diff == 0
        print(f"Kecocokan dengan file kategorikal yang tersedia: {'OK' if same else 'BERBEDA'}")
        if not same:
            raise RuntimeError(f"Hasil kategorisasi berbeda pada {diff} sel.")

    categorical.to_csv(target, index=False)
    print(f"Selesai: {len(categorical)} learner, {len(FEATURES)} fitur -> {target.name}")


if __name__ == "__main__":
    main()
