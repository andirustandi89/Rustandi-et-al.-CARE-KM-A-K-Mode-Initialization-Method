# CARE-KM Reproducibility Package

This archive accompanies the manuscript **“CARE-KM: A K-Modes Initialization Method for Felder-Silverman Learning Style Profiling and Prediction from LMS Behavior.”** It is deliberately limited to the data, code, split definitions, and run-level outputs needed to audit the results reported in the manuscript. Notebook files, exploratory analyses, plots, cached checkpoints, and variables not used in the paper are excluded.

## Contents

- `DATA/MAIN_LMS/` — anonymized 6,658 × 13 learner matrices before and after the paper's three-level discretization, the frozen primary threshold table, and source provenance.
- `DATA/EXTERNAL_ILS/` — the 736-row questionnaire-label audit table with source row indices and anonymized record IDs. The original IEEE DataPort workbooks are not redistributed.
- `DATA/BENCHMARK/` — UCI source manifest with exact filenames and SHA-256 values. UCI raw archives are not duplicated.
- `SPLITS/` — exact 70/30 main-LMS and external-ILS split membership used for prediction.
- `CODE/` — local Python implementations; no Jupyter/Colab dependency is required.
- `OUTPUT/` — machine-readable run-level results and the minimum derived artifacts needed to reconstruct the manuscript tables.
- `REPRODUCED/` — destination used when the scripts are rerun locally.

## Data provenance and redistribution

### Main LMS
The upstream source is the public Moodle project at:
https://github.com/moisesriestra/moodle-early-performance-prediction

The study preprocessing starts from the `mysql-export` Moodle dump referenced by that repository. The raw dump is not duplicated here. Original Moodle identifiers are removed from this archive and replaced by `L000001`–`L006658`. The two included study matrices expose only the 13 behavioral indicators used by the manuscript.

### External questionnaire-grounded cohort
The external source is **Learning Style Identification** on IEEE DataPort:
https://ieee-dataport.org/documents/learning-style-identification

The archive does not rehost the third-party workbooks. The audit file preserves cohort and source row index, but source IDs are removed and replaced with `I0001`–`I0736`. The study uses the first 480 complete-label CSHS records in source order and all 256 complete-label CSMS records. Train-only discretization thresholds and prediction outputs are included.

### UCI benchmarks
The ten benchmark datasets come from the UCI Machine Learning Repository. `DATA/BENCHMARK/benchmark_manifest.csv` records the UCI dataset page, exact local filename, study dimensions, K, and source SHA-256. Raw UCI archives remain at UCI and should be downloaded according to their current dataset-specific terms.

## Reproduction

Create a virtual environment with Python 3.13 and install:

```bash
python -m pip install -r requirements.txt
```

Validate the archive first:

```bash
python CODE/validate_package.py
```

The primary scripts are:

```bash
python CODE/02_discretize_main_lms.py
python CODE/03_lms_clustering.py --stage full
python CODE/03_lms_clustering.py --stage ablation
python CODE/03_lms_clustering.py --stage robustness
python CODE/04_fslsm_interpretation.py
python CODE/05_sensitivity.py --analysis all
python CODE/06_uci_generalization.py
python CODE/07_generalization_statistics.py
python CODE/08_prediction_main_lms.py
```

For the external IEEE experiment, first place `LS-IEEE-Dataport.zip` in `DATA/EXTERNAL_ILS/SOURCE/`, then run:

```bash
python CODE/09_prediction_external_ieee.py
```

For a complete upstream rebuild of the main LMS learner-level numeric matrix, place the Moodle `mysql-export` dump in `DATA/MAIN_LMS/SOURCE/` and run:

```bash
python CODE/01_raw_moodle_to_numeric.py
```

That upstream script retains the study's event taxonomy, sessionized dwell construction, semantic-coverage checks, and deterministic one-course-per-learner selection. Its working intermediates are written only to `REPRODUCED/RAW_PREPROCESSING/`; they are not part of the public archive.

## Frozen experimental protocol

- CARE-KM: Cao parent modes; separation candidate rule; candidate budget 16; maximum 4 accepted exchanges; minimum gain 0.
- K-Modes: K=16 for the LMS experiments; 100-iteration limit; simple matching dissimilarity.
- Full-data stochastic initializer runs: 15.
- Ablation: 20 matched 80% LMS resamples; base seed 5,161,600.
- Robustness and semantic stability: 30 matched 80% LMS resamples; base seed 6,161,600.
- Cross-dataset evaluation: ten UCI datasets, 30 matched 80% resamples per dataset.
- Prediction split seed: 161,600; 70% training / 30% untouched test before data-dependent fitting.
- Main LMS classifiers use only the 13 behavioral indicators. External IEEE classifiers use the 12 behavior variables plus frozen train-derived CARE-KM cluster membership; observed questionnaire labels remain the targets.

## Scope note

This package intentionally excludes artifacts from the superseded fixed-centroid/Hungarian/FRWD design, exploratory notebooks, runtime-scalability analyses not reported in the current manuscript, figures, model caches, original usernames/IDs, and class-label agreement metrics (ARI/NMI/AMI) that are not used by the manuscript's benchmark validity claims.
