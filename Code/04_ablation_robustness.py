from pathlib import Path
import argparse
import hashlib

import numpy as np
import pandas as pd

from carekm_core import CARE_CONFIG, METHODS, run_method


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "main_lms_categorical.csv"
ABLATION_FILE = ROOT / "Output" / "ablation_results.csv"
ROBUSTNESS_FILE = ROOT / "Output" / "robustness_results.csv"
FEATURES = [
    "content_visit", "content_stay", "selfass_visit_proxy", "selfass_stay_proxy",
    "exercise_visit_proxy", "exercise_stay_proxy", "forum_visit", "forum_stay",
    "forum_post", "quiz_stay_results_proxy", "navigation_overview_visit",
    "navigation_overview_stay", "return_to_overview_rate",
]
K = 16
FRACTION = 0.80


def sample_hash(indices):
    arr = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def run_ablation(x_all, reps=20):
    base_seed = 5161600
    variants = [
        ("A0_Cao", 0),
        ("A1_1_exchange", 1),
        ("A2_2_exchanges", 2),
        ("A3_CARE_4_exchanges", 4),
    ]
    rows = []

    for rep in range(reps):
        rng = np.random.default_rng(base_seed + rep)
        idx = np.sort(rng.choice(len(x_all), size=int(len(x_all) * FRACTION), replace=False))
        x = x_all[idx]
        hash_value = sample_hash(idx)

        for variant, exchanges in variants:
            if exchanges == 0:
                method = "Cao"
                config = None
            else:
                method = "CARE-KM"
                config = dict(CARE_CONFIG)
                config["max_exchanges"] = exchanges

            row, _, _ = run_method(
                method,
                x,
                K,
                seed=base_seed + rep,
                care_config=config,
                si_sample_size=2000,
                si_seed=base_seed + rep,
                max_iter=100,
            )
            row.update({
                "rep": rep,
                "variant": variant,
                "max_exchanges": exchanges,
                "sample_hash": hash_value,
            })
            rows.append(row)
        print(f"Ablation {rep + 1}/{reps} selesai")

    out = pd.DataFrame(rows)
    out.to_csv(ABLATION_FILE, index=False, float_format="%.6f")
    return out


def run_robustness(x_all, reps=30):
    base_seed = 6161600
    rows = []

    for rep in range(reps):
        rng = np.random.default_rng(base_seed + rep)
        idx = np.sort(rng.choice(len(x_all), size=int(len(x_all) * FRACTION), replace=False))
        x = x_all[idx]
        hash_value = sample_hash(idx)

        for method in METHODS:
            row, _, _ = run_method(
                method,
                x,
                K,
                seed=base_seed + rep,
                care_config=CARE_CONFIG,
                si_sample_size=2000,
                si_seed=base_seed + rep,
                max_iter=100,
            )
            row.update({
                "rep": rep,
                "sample_hash": hash_value,
                "sample_n": len(idx),
            })
            rows.append(row)
        print(f"Robustness {rep + 1}/{reps} selesai")

    out = pd.DataFrame(rows)
    out.to_csv(ROBUSTNESS_FILE, index=False, float_format="%.6f")
    return out


def main():
    parser = argparse.ArgumentParser(description="Ablation dan matched-resampling CARE-KM.")
    parser.add_argument("--section", choices=["ablation", "robustness", "both"], default="both")
    parser.add_argument("--ablation-reps", type=int, default=20)
    parser.add_argument("--robustness-reps", type=int, default=30)
    args = parser.parse_args()

    df = pd.read_csv(DATA_FILE, dtype={"learner_id": str})
    x_all = df[FEATURES].astype(str).to_numpy(dtype=object)

    if args.section in {"ablation", "both"}:
        result = run_ablation(x_all, args.ablation_reps)
        print("\nRata-rata ablation:")
        print(result.groupby("variant")[["initial_cost", "final_cost", "dbi", "si", "iterations"]].mean().to_string())

    if args.section in {"robustness", "both"}:
        result = run_robustness(x_all, args.robustness_reps)
        print("\nRata-rata robustness:")
        print(result.groupby("method")[["initial_cost", "final_cost", "dbi", "si", "iterations"]].mean().to_string())


if __name__ == "__main__":
    main()
