from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Output"


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    full = pd.read_csv(OUT / "full_lms_clustering_results.csv")
    ablation = pd.read_csv(OUT / "ablation_results.csv")
    robustness = pd.read_csv(OUT / "robustness_results.csv")
    profiles = pd.read_csv(OUT / "fslsm_cluster_profiles.csv")
    main_pred = pd.read_csv(OUT / "main_lms_holdout_predictions.csv")
    gt_pred = pd.read_csv(OUT / "groundtruth_holdout_predictions.csv")

    print("TABLE 3 - Full LMS clustering")
    cols = ["method", "final_cost", "dbi", "si"]
    print(full[cols].to_string(index=False))

    print("\nTABLE 4 - Ablation")
    abl = ablation.groupby("max_exchanges").agg(
        initial_cost=("initial_cost", "mean"),
        final_cost=("final_cost", "mean"),
        dbi=("dbi", "mean"),
        si=("si", "mean"),
        iterations=("iterations", "mean"),
    ).reset_index()
    print(abl.round(3).to_string(index=False))

    print("\nTABLE 5 - Robustness")
    rob = robustness.groupby("method").agg(
        initial_cost_mean=("initial_cost", "mean"),
        initial_cost_sd=("initial_cost", "std"),
        final_cost_mean=("final_cost", "mean"),
        final_cost_sd=("final_cost", "std"),
        dbi_mean=("dbi", "mean"),
        dbi_sd=("dbi", "std"),
        si_mean=("si", "mean"),
        si_sd=("si", "std"),
    ).reset_index()
    print(rob.round(4).to_string(index=False))

    print("\nTABLE 9 - FSLSM profiles")
    print(profiles[[
        "empirical_cluster", "learners", "base_ls_class", "A_R_score",
        "Sens_Int_score", "Vis_Ver_score", "Seq_Gl_score", "mean_strength"
    ]].to_string(index=False))

    print("\nTABLE 10 - Main LMS holdout")
    rows = []
    dims = [
        ("Active / Reflective", "reference_active_reflective"),
        ("Sensing / Intuitive", "reference_sensing_intuitive"),
        ("Visual / Verbal", "reference_visual_verbal"),
        ("Sequential / Global", "reference_sequential_global"),
    ]
    for model in ["NB", "DT", "SVM"]:
        acc = []
        for title, ref in dims:
            score = (main_pred[f"{model}_{title}"].astype(str) == main_pred[ref].astype(str)).mean()
            acc.append(score)
        rows.append([model, *[pct(x) for x in acc], pct(sum(acc) / 4)])
    print(pd.DataFrame(rows, columns=["Classifier", *[x[0] for x in dims], "Mean"]).to_string(index=False))

    print("\nTABLE 11 - Questionnaire-grounded evaluation")
    gt_dims = ["active_reflective", "sensing_intuitive", "visual_verbal", "sequential_global"]
    rows = []
    for model in ["NB", "DT", "SVM"]:
        acc = [
            (gt_pred[f"{model}_{d}"].astype(str) == gt_pred[f"ils_{d}"].astype(str)).mean()
            for d in gt_dims
        ]
        rows.append([model, *[pct(x) for x in acc], f"{sum(round(100*x,2) for x in acc)/4:.2f}%"])
    care = [
        (gt_pred[f"carekm_{d}"].astype(str) == gt_pred[f"ils_{d}"].astype(str)).mean()
        for d in gt_dims
    ]
    rows.append(["CARE-KM Agreement", *[pct(x) for x in care], f"{sum(round(100*x,2) for x in care)/4:.2f}%"])
    print(pd.DataFrame(rows, columns=["Method", "Active / Reflective", "Sensing / Intuitive", "Visual / Verbal", "Sequential / Global", "Mean"]).to_string(index=False))


if __name__ == "__main__":
    main()
