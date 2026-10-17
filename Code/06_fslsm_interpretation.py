from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from carekm_core import CARE_CONFIG, care_init, kmodes_fit


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "main_lms_categorical.csv"
OUTPUT_FILE = ROOT / "Output" / "fslsm_cluster_profiles.csv"
FEATURES = [
    "content_visit", "content_stay", "selfass_visit_proxy", "selfass_stay_proxy",
    "exercise_visit_proxy", "exercise_stay_proxy", "forum_visit", "forum_stay",
    "forum_post", "quiz_stay_results_proxy", "navigation_overview_visit",
    "navigation_overview_stay", "return_to_overview_rate",
]
K = 16

DIMENSIONS = [
    ("Active/Reflective", ("Active", "Reflective")),
    ("Sensing/Intuitive", ("Sensing", "Intuitive")),
    ("Visual/Verbal", ("Visual", "Verbal")),
    ("Sequential/Global", ("Sequential", "Global")),
]
POLES = dict(DIMENSIONS)
ABBR = {
    "Active": "Act.", "Reflective": "Ref.", "Sensing": "Sens.", "Intuitive": "Int.",
    "Visual": "Vis.", "Verbal": "Ver.", "Sequential": "Seq.", "Global": "Gl.",
}
DIM_ABBR = {
    "Active/Reflective": "A/R",
    "Sensing/Intuitive": "Sens./Int.",
    "Visual/Verbal": "Vis./Ver.",
    "Sequential/Global": "Seq./Gl.",
}
LS_LOOKUP = {
    tuple(combo): f"LS{i:02d}"
    for i, combo in enumerate(itertools.product(*[pair for _, pair in DIMENSIONS]), start=1)
}

# Arah bukti Low -> High. Nilai 2 (Moderate) diperlakukan netral.
RULES = [
    ("content_visit", "Active/Reflective", "Active", "Reflective"),
    ("content_stay", "Active/Reflective", "Active", "Reflective"),
    ("selfass_visit_proxy", "Active/Reflective", "Reflective", "Active"),
    ("selfass_stay_proxy", "Active/Reflective", "Active", "Reflective"),
    ("exercise_visit_proxy", "Active/Reflective", "Reflective", "Active"),
    ("exercise_stay_proxy", "Active/Reflective", "Reflective", "Active"),
    ("forum_visit", "Active/Reflective", "Active", "Reflective"),
    ("forum_post", "Active/Reflective", "Reflective", "Active"),
    ("quiz_stay_results_proxy", "Active/Reflective", "Active", "Reflective"),
    ("content_visit", "Sensing/Intuitive", "Sensing", "Intuitive"),
    ("content_stay", "Sensing/Intuitive", "Sensing", "Intuitive"),
    ("selfass_visit_proxy", "Sensing/Intuitive", "Intuitive", "Sensing"),
    ("selfass_stay_proxy", "Sensing/Intuitive", "Intuitive", "Sensing"),
    ("exercise_visit_proxy", "Sensing/Intuitive", "Intuitive", "Sensing"),
    ("quiz_stay_results_proxy", "Sensing/Intuitive", "Intuitive", "Sensing"),
    ("content_visit", "Visual/Verbal", "Visual", "Verbal"),
    ("forum_visit", "Visual/Verbal", "Visual", "Verbal"),
    ("forum_stay", "Visual/Verbal", "Visual", "Verbal"),
    ("forum_post", "Visual/Verbal", "Visual", "Verbal"),
    ("navigation_overview_visit", "Sequential/Global", "Sequential", "Global"),
    ("navigation_overview_stay", "Sequential/Global", "Sequential", "Global"),
    ("return_to_overview_rate", "Sequential/Global", "Sequential", "Global"),
]


def signed_vote(value, dimension, low_pole, high_pole):
    value = int(value)
    if value == 2:
        return 0
    first, second = POLES[dimension]
    pole = high_pole if value == 3 else low_pole
    if pole == first:
        return 1
    if pole == second:
        return -1
    raise ValueError(pole)


def nearest_integer(value):
    return int(np.floor(float(value) + 0.5))


def strength_11eq(positive, negative, neutral):
    total = positive + negative + neutral
    if total <= 0:
        return None

    positive_adjusted = positive + 0.5 * neutral
    positive_units = nearest_integer(11.0 * positive_adjusted / total)
    positive_units = min(11, max(0, positive_units))
    negative_units = 11 - positive_units
    magnitude = abs(positive_units - negative_units)

    if magnitude in {1, 3}:
        category = "Balanced"
    elif magnitude in {5, 7}:
        category = "Moderate preference"
    elif magnitude in {9, 11}:
        category = "Strong preference"
    else:
        raise RuntimeError(f"Magnitude 11-equivalent tidak valid: {magnitude}")
    return magnitude, category


def dimension_profile(cluster_id, dimension, x, labels, modes, feature_pos):
    rules = [r for r in RULES if r[1] == dimension]
    mode_votes = np.asarray([
        signed_vote(modes[cluster_id, feature_pos[f]], d, low, high)
        for f, d, low, high in rules
    ])
    mode_score = int(np.sum(mode_votes > 0) - np.sum(mode_votes < 0))

    member = x[labels == cluster_id]
    votes = []
    for f, d, low, high in rules:
        col = feature_pos[f]
        votes.extend(signed_vote(v, d, low, high) for v in member[:, col])
    votes = np.asarray(votes, dtype=int)
    positive = int(np.sum(votes > 0))
    negative = int(np.sum(votes < 0))
    neutral = int(np.sum(votes == 0))
    member_score = positive - negative

    first, second = POLES[dimension]
    if mode_score > 0:
        selected = first
    elif mode_score < 0:
        selected = second
    elif member_score > 0:
        selected = first
    elif member_score < 0:
        selected = second
    else:
        selected = None

    score = strength_11eq(positive, negative, neutral)
    if selected is None or score is None:
        return None, "Balanced", None, "Unresolved"

    magnitude, category = score
    code = f"{ABBR[selected]}{magnitude}"
    return selected, code, magnitude, category


def build_profiles(x, labels, modes):
    feature_pos = {f: i for i, f in enumerate(FEATURES)}
    rows = []

    for cluster_id in range(K):
        selected = []
        codes = []
        magnitudes = []
        categories = []

        for dimension, _ in DIMENSIONS:
            pole, code, magnitude, category = dimension_profile(
                cluster_id, dimension, x, labels, modes, feature_pos
            )
            selected.append(pole)
            codes.append(code)
            magnitudes.append(magnitude)
            categories.append(category)

        if any(p is None for p in selected):
            base_ls = "AMBIGUOUS"
            base_profile = "-".join("Balanced" if p is None else ABBR[p] for p in selected)
            mean_strength = np.nan
            boundary_dimension = ""
            nearest_alternative = ""
        else:
            base_ls = LS_LOOKUP[tuple(selected)]
            base_profile = "-".join(ABBR[p] for p in selected)
            mean_strength = float(np.mean(magnitudes))
            minimum = min(magnitudes)
            boundary_idx = [i for i, value in enumerate(magnitudes) if value == minimum]
            boundary_dimension = "|".join(DIM_ABBR[DIMENSIONS[i][0]] for i in boundary_idx)

            alternatives = []
            if minimum in {1, 3}:
                for i in boundary_idx:
                    alt = list(selected)
                    first, second = DIMENSIONS[i][1]
                    alt[i] = second if alt[i] == first else first
                    alternatives.append(LS_LOOKUP[tuple(alt)])
            nearest_alternative = "|".join(dict.fromkeys(alternatives))

        rows.append({
            "empirical_cluster": f"C{cluster_id + 1:02d}",
            "learners": int(np.sum(labels == cluster_id)),
            "base_ls_class": base_ls,
            "base_fslsm_profile": base_profile,
            "A_R_score": codes[0],
            "A_R_interpretation": categories[0],
            "Sens_Int_score": codes[1],
            "Sens_Int_interpretation": categories[1],
            "Vis_Ver_score": codes[2],
            "Vis_Ver_interpretation": categories[2],
            "Seq_Gl_score": codes[3],
            "Seq_Gl_interpretation": categories[3],
            "mean_strength": mean_strength,
            "boundary_dimension": boundary_dimension,
            "nearest_alternative_ls": nearest_alternative,
            "preference_signature": "|".join(codes),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_FILE, dtype={"learner_id": str})
    x = df[FEATURES].astype(str).to_numpy(dtype=object)

    # CARE-KM deterministik pada full LMS dataset.
    init_modes = care_init(x, K, **CARE_CONFIG)
    result = kmodes_fit(x, init_modes, max_iter=100)
    profiles = build_profiles(x, result["labels"], result["modes"])

    if OUTPUT_FILE.exists():
        reference = pd.read_csv(OUTPUT_FILE)
        check_cols = [
            "empirical_cluster", "learners", "base_ls_class", "A_R_score",
            "Sens_Int_score", "Vis_Ver_score", "Seq_Gl_score", "mean_strength",
        ]
        merged = reference[check_cols].copy()
        test = profiles[check_cols].copy()
        exact_text = (merged.drop(columns=["mean_strength"]).astype(str).to_numpy() == test.drop(columns=["mean_strength"]).astype(str).to_numpy()).all()
        exact_strength = np.allclose(merged["mean_strength"], test["mean_strength"], equal_nan=True)
        print(f"Kecocokan profil dengan output yang tersedia: {'OK' if exact_text and exact_strength else 'BERBEDA'}")

    profiles.to_csv(OUTPUT_FILE, index=False, float_format="%.6f")
    print(profiles.to_string(index=False))
    print(f"\nDisimpan ke: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
