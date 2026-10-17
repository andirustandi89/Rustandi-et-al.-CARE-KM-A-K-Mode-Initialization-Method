from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Output"


def main():
    semantic = pd.read_csv(OUTPUT_DIR / "semantic_stability_results.csv")
    preprocessing = pd.read_csv(OUTPUT_DIR / "preprocessing_sensitivity_results.csv")

    dims = [
        "active_reflective_agreement",
        "sensing_intuitive_agreement",
        "visual_verbal_agreement",
        "sequential_global_agreement",
    ]

    print("Semantic stability, 30 matched resamples:")
    for col in dims:
        label = col.replace("_agreement", "").replace("_", " /").title().replace(" /", "/")
        print(f"  {label:28s}: {100 * semantic[col].mean():.1f}%")
    print(f"  {'Mean dimensional agreement':28s}: {100 * semantic[dims].mean(axis=1).mean():.1f}%")

    print("\nPreprocessing sensitivity:")
    cols = ["configuration", "dbi", "si", "mean_dimensional_agreement", "base_ls_agreement"]
    shown = preprocessing[cols].copy()
    shown["mean_dimensional_agreement"] *= 100
    shown["base_ls_agreement"] *= 100
    print(shown.to_string(index=False, formatters={
        "dbi": lambda x: f"{x:.3f}",
        "si": lambda x: f"{x:.3f}",
        "mean_dimensional_agreement": lambda x: f"{x:.1f}%",
        "base_ls_agreement": lambda x: f"{x:.1f}%",
    }))

    # Pemeriksaan sederhana agar file tidak tertukar dengan hasil eksperimen lain.
    if len(semantic) != 30:
        raise RuntimeError(f"Semantic stability harus berisi 30 resample, ditemukan {len(semantic)}.")
    expected = {"PRIMARY", "EMPIRICAL_Q30_Q70", "EMPIRICAL_Q35_Q65", "GRAF_MINUS_10PCT", "GRAF_PLUS_10PCT"}
    if set(preprocessing["configuration"]) != expected:
        raise RuntimeError("Konfigurasi preprocessing sensitivity tidak lengkap.")


if __name__ == "__main__":
    main()
