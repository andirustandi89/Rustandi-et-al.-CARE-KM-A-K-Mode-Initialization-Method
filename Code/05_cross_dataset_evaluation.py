from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULT_FILE = ROOT / "Output" / "cross_dataset_results.csv"
METHOD_ORDER = ["Random", "Huang", "Cao", "Bai", "K-Modes++", "HR-2023", "CARE-KM"]
METRICS = {
    "initial_cost": True,
    "final_cost": True,
    "dbi": True,
    "si": False,
    "iterations": True,
}


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, pos in enumerate(order):
        value = (len(p) - rank) * p[pos]
        running = max(running, value)
        adjusted[pos] = min(1.0, running)
    return adjusted


def dataset_means(raw):
    cols = list(METRICS)
    return raw.groupby(["dataset", "method"], as_index=False)[cols].mean()


def average_ranks(means):
    rows = []
    for metric, lower_is_better in METRICS.items():
        pivot = means.pivot(index="dataset", columns="method", values=metric)[METHOD_ORDER]
        ranks = pivot.rank(axis=1, method="average", ascending=lower_is_better)
        row = {"metric": metric}
        row.update(ranks.mean(axis=0).to_dict())
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric")


def friedman_table(means):
    rows = []
    for metric in METRICS:
        pivot = means.pivot(index="dataset", columns="method", values=metric)[METHOD_ORDER]
        stat, p = friedmanchisquare(*[pivot[m].to_numpy() for m in METHOD_ORDER])
        n = len(pivot)
        k = len(METHOD_ORDER)
        kendall_w = stat / (n * (k - 1))
        rows.append({"metric": metric, "friedman_statistic": stat, "p_value": p, "kendall_w": kendall_w})
    return pd.DataFrame(rows)


def care_pairwise(means, metric):
    lower_is_better = METRICS[metric]
    pivot = means.pivot(index="dataset", columns="method", values=metric)[METHOD_ORDER]
    care = pivot["CARE-KM"].to_numpy()
    rows = []

    for baseline in METHOD_ORDER:
        if baseline == "CARE-KM":
            continue
        other = pivot[baseline].to_numpy()
        improvement = (other - care) if lower_is_better else (care - other)
        wins = int(np.sum(improvement > 1e-12))
        ties = int(np.sum(np.abs(improvement) <= 1e-12))
        losses = int(np.sum(improvement < -1e-12))
        try:
            _, p = wilcoxon(improvement, alternative="two-sided", zero_method="wilcox", method="auto")
        except ValueError:
            p = 1.0
        rows.append({"baseline": baseline, "wins": wins, "ties": ties, "losses": losses, "p_value": p})

    # Satu keluarga Holm berisi enam baseline untuk setiap metrik.
    adj = holm_adjust([r["p_value"] for r in rows])
    for row, p_adj in zip(rows, adj):
        row["holm_adjusted"] = p_adj
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(RESULT_FILE)
    required = {"dataset", "method", "rep", *METRICS.keys()}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"Kolom hasil cross-dataset tidak lengkap: {sorted(missing)}")

    # File menyimpan hasil per resample. Ringkasan di bawah dihitung ulang dari file tersebut.
    counts = raw.groupby(["dataset", "method"])["rep"].nunique()
    if counts.min() != 30 or counts.max() != 30:
        print("Peringatan: jumlah resample tidak seragam 30 pada seluruh pasangan dataset-metode.")

    means = dataset_means(raw)
    print("Average ranks (10 dataset):")
    print(average_ranks(means).round(3).to_string())

    print("\nFriedman dan Kendall's W:")
    print(friedman_table(means).to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    for metric in ["dbi", "si"]:
        print(f"\nCARE-KM pairwise untuk {metric.upper()}:")
        print(care_pairwise(means, metric).to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    print("\nCatatan: data mentah benchmark pihak ketiga tidak disertakan dalam repositori ini.")
    print("File ini menghitung ulang analisis statistik dari matched-resample results yang tersedia di Output/.")


if __name__ == "__main__":
    main()
