# CARE-KM Reproducibility Package

Reproducibility materials for **CARE-KM Initialization for K-Modes Clustering with an Application to Felder–Silverman Learner Profiles** by Andi Rustandi, Sri Mulyana, and Sigit Priyanta.

CARE-KM refines a complete set of Cao initial modes through a bounded sequence of exchanges that reduce initial assignment cost. Standard K-Modes assignment and mode updating follow the refinement. The study evaluates clustering performance, interprets the resulting learner clusters using the Felder–Silverman Learning Style Model (FSLSM), and evaluates the derived directions in separate holdout analyses.

## Data sources

### Main LMS cohort

The main cohort comprises **6,658 learners and 13 behavioral indicators**. Learner-level records were reconstructed from the public Moodle MySQL export associated with:

> Riestra-González, M., Paule-Ruíz, M. del P., and Ortin, F. (2021). Massive LMS log data analysis for the early prediction of course-agnostic student performance. *Computers & Education*, 163, 104108. https://doi.org/10.1016/j.compedu.2020.104108

The upstream data and acquisition instructions are available in the [Moodle early performance prediction repository](https://github.com/moisesriestra/moodle-early-performance-prediction).

The study uses its own preprocessing pipeline. The 13 indicators summarize content access and duration, self-assessment, exercises, forum activity, quiz-result review, and overview navigation. Each indicator is discretized as Low, Moderate, or High. K-Modes treats these codes as nominal values under simple matching dissimilarity. Learning-style labels are not inputs to clustering.

### Anonymized questionnaire cohort

The questionnaire analysis uses **461 learners with paired LMS behavioral records and Index of Learning Styles (ILS) responses**, as described in the manuscript. The records come from a computer education program at an anonymized higher-education institution in Indonesia. Institution and learner identifiers were removed before analysis.

The anonymized dataset is made available through the following download page:

**[Access the anonymized dataset](https://www.smkn5ppu.sch.id/berita/16/apa-itu-machine-learning-mengenal-cara-komputer-belajar-dari-data)**

Use the dataset download provided on that page. The URL above identifies the access page rather than a direct CSV or archive endpoint. The hosting website is the distribution location; it does not identify the institution from which the cohort originated.

### Categorical benchmarks

The initialization study also evaluates ten datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu): Car Evaluation, Lymphography, Hayes-Roth, Statlog Australian Credit Approval, Chronic Kidney Disease, Annealing, Statlog Heart, Heart Disease Cleveland, the New Thyroid subset of Thyroid Disease, and Nursery.

For each benchmark, the number of clusters is set to the reference-class count as an evaluation configuration. Individual class labels are not used for cluster assignment, mode updating, or the internal-validity measures. The exact source version, preprocessing, and resulting study dimensions are needed to reproduce each benchmark analysis.

## Study protocol

### Clustering

| Setting | Value |
| --- | --- |
| Parent initialization | Cao |
| Dissimilarity | Simple matching dissimilarity |
| Candidate cap per provisional region | H = 16 |
| Maximum accepted exchanges | E = 4 |
| Exchange acceptance | Strictly positive reduction in initial assignment cost |
| Additional minimum-gain threshold | None |
| LMS cluster count | K = 16 |
| Subsequent clustering | Standard K-Modes |
| K-Modes iteration limit | 100 |
| Compared initializers | Random, Huang, Cao, Bai, K-Modes++, HR-2023, CARE-KM |
| Full-data stochastic runs | 15, summarized without best-run selection |
| Exchange-depth ablation | E = 0, 1, 2, 4 on 20 matched 80% LMS samples |
| Robustness analysis | 30 additional matched 80% LMS samples |
| Benchmark evaluation | 10 datasets with 30 matched 80% resamples per dataset |

FSLSM evidence is applied after clustering. The LMS cluster count is motivated by the sixteen possible FSLSM direction combinations; it does not require the empirical clusters to represent sixteen distinct learning-style profiles.

### Holdout analyses

| Cohort | Total learners | Training | Test | Test reference |
| --- | ---: | ---: | ---: | --- |
| Main LMS | 6,658 | 4,660 | 1,998 | Fixed, training-derived CARE-KM directions |
| Questionnaire | 461 | 322 | 139 | Independently obtained ILS directions |

Both analyses split learners before data-dependent fitting. Preprocessing parameters, CARE-KM modes, cluster-level FSLSM interpretations, and classifiers are fitted on the relevant training portion and held fixed for test processing. Test results are not used to select or retune these components.

For the **main LMS cohort**, test learners are assigned to the fixed modes and inherit the corresponding FSLSM directions. Naive Bayes (NB), decision tree (DT), and support vector machine (SVM) classifiers use behavioral indicators to predict those directions. Cluster identity is excluded from their predictors. This analysis measures reproduction of operational labels on held-out learners.

For the **questionnaire cohort**, the split is stratified by the complete ILS class. ILS labels are used for stratification and external evaluation, not for clustering or assigning FSLSM directions to clusters. NB, DT, and SVM are trained on CARE-KM-derived training labels. Their predictions are evaluated against ILS labels on the 139 test learners. Direct agreement between CARE-KM-derived directions and ILS labels is reported separately.

The questionnaire analysis fits the procedure on that cohort's training portion. It does not directly transfer the fitted main-LMS model to the questionnaire cohort.

## Expected package organization

| Location | Purpose |
| --- | --- |
| `CODE/` | Preprocessing, clustering, interpretation, sensitivity, statistics, and prediction implementations |
| `DATA/MAIN_LMS/` | Study matrices and provenance for the main LMS cohort |
| `DATA/EXTERNAL_ILS/` | Questionnaire-cohort inputs and provenance; current analysis uses 461 learners |
| `DATA/BENCHMARK/` | Benchmark source and preprocessing information |
| `SPLITS/` | Saved learner membership for the reported training and test splits |
| `OUTPUT/` | Results used to reconstruct the manuscript tables |
| `MANIFEST.csv` | File inventory and integrity information |
| `protocol.json` | Machine-readable experiment settings |
| `requirements.txt` | Python dependencies |

The questionnaire-cohort files, split definitions, protocol entries, and outputs must all refer to the same 461-learner experiment.

## Reproduction

### Environment

Install dependencies in an isolated Python environment:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

### Main LMS and benchmark workflow

The existing package documents the following entry points:

```bash
python CODE/validate_package.py
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

For an upstream rebuild, obtain the Moodle `mysql-export` dump from the source repository and follow the input-path requirements of `CODE/01_raw_moodle_to_numeric.py`.

Successful execution or package validation should be followed by a comparison of the generated results with the manuscript tables. Validation alone does not establish that every reported result has been reproduced.

### Current questionnaire-cohort workflow

1. Download the anonymized dataset from the [dataset access page](https://www.smkn5ppu.sch.id/berita/16/apa-itu-machine-learning-mengenal-cara-komputer-belajar-dari-data).
2. Check the input schema, paired behavior–ILS records, and saved learner membership against the reported cohort of 461 learners and the 322/139 split.
3. Fit data-dependent preprocessing, CARE-KM, and cluster interpretation on the 322 training learners only.
4. Assign the 139 test learners using the fixed training-derived pipeline.
5. Compare their CARE-KM-derived directions directly with ILS labels for each FSLSM dimension.
6. Train NB, DT, and SVM on the CARE-KM-derived training targets, then evaluate test predictions against ILS labels.
7. Reconstruct Table 11 and the reported balanced accuracy and macro-F1 summaries from the saved predictions.


## Reported reference results

These values identify the results reported in the current manuscript. They are comparison targets for reproduction, not a statement that a fresh execution has been completed.

| Analysis | Reported result |
| --- | --- |
| Full LMS CARE-KM final assignment cost | 23,822 |
| Full LMS CARE-KM Davies–Bouldin index | 1.541 |
| Full LMS CARE-KM silhouette coefficient | 0.120 |
| Main LMS holdout SVM mean dimensional accuracy | 93.12% |
| Questionnaire holdout direct CARE-KM–ILS mean agreement | 79.32% |
| Questionnaire holdout NB mean accuracy against ILS | 83.28% |
| Questionnaire holdout DT mean accuracy against ILS | 80.58% |
| Questionnaire holdout SVM mean accuracy against ILS | 82.20% |
| Questionnaire holdout NB mean balanced accuracy | 69.63% |
| Questionnaire holdout NB mean macro-F1 | 72.00% |

All means in the holdout results are computed across the four FSLSM dimensions. Direct cluster-derived agreement, classifier accuracy, balanced accuracy, and macro-F1 are distinct measures.

## Interpretation and scope

The algorithmic contribution is the refinement of K-Modes initialization. FSLSM profiles describe behavioral evidence in the resulting clusters. Agreement with cluster-derived labels does not independently validate a psychological construct, and correspondence with ILS responses does not establish psychometric equivalence or the benefit of matching instruction to the assigned profiles.

The results are limited to the datasets, splits, and configurations described in the manuscript.

## Citation

When using these materials, cite the accompanying CARE-KM manuscript and the original data sources relevant to the analysis. Use the manuscript's final bibliographic details when available.
