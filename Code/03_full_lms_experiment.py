from pathlib import Path
import argparse

import pandas as pd

from carekm_core import CARE_CONFIG, METHODS, STOCHASTIC_METHODS, run_method


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "main_lms_categorical.csv"
OUTPUT_FILE = ROOT / "Output" / "full_lms_clustering_results.csv"
FEATURES = [
    "content_visit", "content_stay", "selfass_visit_proxy", "selfass_stay_proxy",
    "exercise_visit_proxy", "exercise_stay_proxy", "forum_visit", "forum_stay",
    "forum_post", "quiz_stay_results_proxy", "navigation_overview_visit",
    "navigation_overview_stay", "return_to_overview_rate",
]
K = 16
BASE_SEED = 4161600


def summarize(raw):
    metrics = [
        "initial_cost", "final_cost", "dbi", "si", "iterations",
        "min_cluster", "max_cluster", "cluster_cv", "init_sec",
        "cluster_sec", "total_sec",
    ]
    rows = []
    for method, group in raw.groupby("method", sort=False):
        row = {"method": method, "n": len(group)}
        for metric in metrics:
            row[metric] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std() if len(group) > 1 else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    out["rank_final_cost"] = out["final_cost"].rank(method="average", ascending=True)
    out["rank_dbi"] = out["dbi"].rank(method="average", ascending=True)
    out["rank_si"] = out["si"].rank(method="average", ascending=False)
    out["rank_primary"] = out[["rank_final_cost", "rank_dbi", "rank_si"]].mean(axis=1)
    return out.sort_values(["rank_primary", "final_cost"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Menjalankan eksperimen penuh tujuh initializer K-Modes.")
    parser.add_argument("--stochastic-reps", type=int, default=15, help="Jumlah pengulangan untuk initializer stokastik.")
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(DATA_FILE, dtype={"learner_id": str})
    x = df[FEATURES].astype(str).to_numpy(dtype=object)

    rows = []
    for method in METHODS:
        reps = args.stochastic_reps if method in STOCHASTIC_METHODS else 1
        for rep in range(reps):
            seed = BASE_SEED + rep
            print(f"{method:10s}  run {rep + 1}/{reps}")
            row, _, _ = run_method(
                method,
                x,
                K,
                seed=seed,
                care_config=CARE_CONFIG,
                si_sample_size=2000,
                si_seed=seed,
                max_iter=args.max_iter,
            )
            row["rep"] = rep
            rows.append(row)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    summary.to_csv(OUTPUT_FILE, index=False, float_format="%.6f")
    print("\nRingkasan eksperimen:")
    print(summary[["method", "initial_cost", "final_cost", "dbi", "si", "iterations"]].to_string(index=False))
    print(f"\nDisimpan ke: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
