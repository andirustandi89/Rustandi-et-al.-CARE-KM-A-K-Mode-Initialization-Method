"""Paired LMS tests with joint Holm correction across 24 comparisons."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from analyze_benchmarks import holm
from run_lms_experiments import METHODS
ROOT=Path(__file__).resolve().parents[1]
def save_csv(df,path):
    df.to_csv(path,index=False,float_format='%.6f')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=ROOT/'OUTPUT/experiments/06_robustness_raw.csv')
    parser.add_argument('--out',type=Path,default=ROOT/'OUTPUT/recomputed/lms_statistics')
    args=parser.parse_args()
    OUT_DIR=args.out
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    from scipy.stats import wilcoxon, rankdata

    RAW_FILE = args.raw
    if not RAW_FILE.exists():
        raise FileNotFoundError(RAW_FILE)
    raw = pd.read_csv(RAW_FILE)

    PRIMARY = {
        'initial_cost': 'lower',
        'final_cost': 'lower',
        'dbi': 'lower',
        'si': 'higher',
    }
    BASELINES = [m for m in METHODS if m != 'CARE-KM']
    rows = []

    for baseline in BASELINES:
        care = raw[raw['method']=='CARE-KM'].set_index('rep').sort_index()
        ref = raw[raw['method']==baseline].set_index('rep').sort_index()
        common = care.index.intersection(ref.index)
        if len(common) != 30:
            raise RuntimeError(f'{baseline}: paired reps = {len(common)}, expected 30')
        if not (care.loc[common,'sample_hash'].astype(str).to_numpy() == ref.loc[common,'sample_hash'].astype(str).to_numpy()).all():
            raise RuntimeError(f'{baseline}: sample hashes tidak matched')

        for metric, direction in PRIMARY.items():
            a = care.loc[common, metric].astype(float).to_numpy()
            b = ref.loc[common, metric].astype(float).to_numpy()
            improvement = (b - a) if direction == 'lower' else (a - b)
            nz = improvement[np.abs(improvement) > 1e-12]

            if len(nz) == 0:
                p_raw = 1.0
                rbc = 0.0
            else:
                try:
                    p_raw = float(wilcoxon(nz, alternative='greater', zero_method='wilcox', method='auto').pvalue)
                except ValueError:
                    p_raw = 1.0
                ranks = rankdata(np.abs(nz), method='average')
                w_plus = float(ranks[nz > 0].sum())
                w_minus = float(ranks[nz < 0].sum())
                rbc = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) else 0.0

            wins = int((improvement > 1e-12).sum())
            ties = int((np.abs(improvement) <= 1e-12).sum())
            losses = int((improvement < -1e-12).sum())

            rng = np.random.default_rng(700000 + 97*BASELINES.index(baseline) + list(PRIMARY).index(metric))
            boot = np.empty(5000)
            for i in range(len(boot)):
                boot[i] = rng.choice(improvement, size=len(improvement), replace=True).mean()
            lo, hi = np.quantile(boot, [0.025,0.975])

            rows.append({
                'baseline': baseline, 'metric': metric, 'direction': direction,
                'n_pairs': len(common),
                'care_mean': float(a.mean()), 'baseline_mean': float(b.mean()),
                'mean_improvement': float(improvement.mean()),
                'wins': wins, 'ties': ties, 'losses': losses,
                'win_rate': float(wins/len(improvement)),
                'rank_biserial': float(rbc),
                'bootstrap_lo': float(lo), 'bootstrap_hi': float(hi),
                'p_raw': p_raw,
            })

    stats = pd.DataFrame(rows)
    stats['p_holm'] = holm(stats['p_raw'].to_numpy())
    stats['significant_0_05'] = stats['p_holm'] < 0.05

    save_csv(stats, OUT_DIR / '07_wilcoxon_holm.csv')
    save_csv(stats[['baseline','metric','mean_improvement','bootstrap_lo','bootstrap_hi']], OUT_DIR / '07_bootstrap_ci.csv')
    save_csv(stats[['baseline','metric','wins','ties','losses','win_rate','rank_biserial']], OUT_DIR / '07_effect_size_wtl.csv')

    print(stats.to_string(index=False))
