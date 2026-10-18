# CARE-KM Initialization for K-Modes Clustering

Research data, Python implementations, and experimental results accompanying **CARE-KM Initialization for K-Modes Clustering with an Application to Felder–Silverman Learner Profiles**, by Andi Rustandi, Sri Mulyana, and Sigit Priyanta.

We evaluate bounded refinement of Cao initialization before standard K-Modes optimization. The study includes 6,658 LMS learners and ten benchmarks represented categorically. We interpret converged LMS clusters along four Felder–Silverman dimensions and evaluate a separate 461-learner cohort against independently obtained Index of Learning Styles (ILS) responses.

## Repository structure

```text
DATA/
  main_lms/                 13 categorical indicators, fixed split, evidence rules and thresholds
  questionnaire_cohort/     Training and test indicators, ILS labels and preprocessing parameters
  benchmarks/               Source archive registry for the ten benchmark datasets
CODE/                       CARE-KM implementation and executable analyses
OUTPUT/
  manuscript_tables/        Tables 1–11 with final manuscript numbering
  experiments/              Numerical clustering and sensitivity results
  holdout/                  Saved modes, interpretations, assignments and predictions
README.md
requirements.txt
file_manifest.csv           SHA-256 checksums for released files
```

## Main findings

| Analysis | Result |
|---|---:|
| Main LMS CARE-KM final assignment cost | 23,822 |
| Main LMS Davies–Bouldin index | 1.541 |
| Main LMS silhouette coefficient | 0.120 |
| Main holdout SVM mean accuracy | 93.12% |
| Main holdout SVM mean balanced accuracy | 91.39% |
| Questionnaire holdout direct CARE-KM mean agreement | 79.32% |
| Questionnaire holdout Naive Bayes mean accuracy | 83.27% |
| Questionnaire holdout Naive Bayes mean balanced accuracy | 69.63% |

The main holdout measures reproduction of fixed training-derived directions. The questionnaire holdout measures correspondence with independent ILS labels. These are distinct evaluation targets. Behavioral magnitudes describe evidence within clusters; they are not questionnaire scores or confidence estimates.

## Data

### Main LMS cohort

`DATA/main_lms/behavior_categorical.csv` contains one row per learner and the 13 indicators used for clustering. Codes are **1 = Low, 2 = Moderate, 3 = High** and are treated nominally. `learner_id` is a replacement identifier and must never be included as a clustering or classification input. Row order matches the original full-data analysis.

The data were reconstructed from the public Moodle export used by Riestra-González, Paule-Ruíz, and Ortin, *Massive LMS Log Data Analysis for the Early Prediction of Course-Agnostic Student Performance*, Computers & Education 163, 104108 (2021), DOI: `10.1016/j.compedu.2020.104108` (manuscript reference 27).

The source-level names containing `_proxy` are retained to match the experimental implementation. `selfass_visit_proxy`, `selfass_stay_proxy`, `exercise_visit_proxy`, `exercise_stay_proxy`, and `quiz_stay_results_proxy` correspond to the self-assessment, exercise, and quiz-review indicators described in the manuscript. `evidence_rules.csv` records their FSLSM associations.

`split_membership.csv` retains the fixed 4,660/1,998 split and original zero-based row indices. `train_thresholds.csv` contains the thresholds fitted on training learners for the holdout analysis. The original continuous learner matrix and the code reconstructing it from the source export were absent from the supplied experiment archive and are not included here. The released full-data categorical matrix must not be substituted for training-only preprocessing in the main holdout. Saved main-holdout predictions and interpretations support independent metric recalculation; rebuilding that fitted pipeline requires the original numeric matrix.

### Questionnaire cohort

The cohort comprises 461 learners in Computer Education at Universitas Mulawarman, Indonesia, collected through Moodle during the COVID-19 and new-normal periods. `behavior_train.csv` and `behavior_test.csv` contain 322 and 139 learners, respectively. Numeric indicator matrices are also supplied for these same partitions. Category thresholds and fallback decisions were fitted on training data and fixed for test processing.

`ils_labels_and_split.csv` stores the independent ILS codes separately from the behavioral inputs. ILS labels were used for split stratification and external evaluation, not as clustering features or classifier training targets. The released saved split uses seed **161600**. Its membership is preserved directly because changing identifiers or sorting before splitting could change the original sample allocation.

The 19 available core indicators are listed in the behavior files and `feature_dictionary.csv`. The training set determines which constant indicators are excluded. `evidence_rules.csv` lists the predefined behavioral directions; `train_preprocessing_parameters.csv` records flattened training-fitted preprocessing parameters. Rebuilding indicators from raw Moodle events is outside this release; the executable holdout starts from the released, training-fitted categorical matrices.

CARE-KM is fitted on all 322 training learners. Classifiers use the **281** training learners with resolved composite labels. All **139** test learners remain in the primary evaluation. Fourteen direct assignments are unresolved on Visual/Verbal and count as non-matches, with false negatives assigned to the corresponding reference pole. Classifiers are trained on composite cluster-derived profiles in this cohort; their predictions are decomposed into the four dimensions for evaluation.

Replacement identifiers use separate `LMS` and `Q` namespaces consistently across data and outputs. Course keys in the released fitted-parameter table also use replacement codes. Original learner and course identifiers, course names, event logs, and raw database archives are excluded.

### Benchmarks

`DATA/benchmarks/source_archives.csv` lists the ten expected UCI archive filenames, dataset names, dimensions and cluster counts. The source archives were absent from the supplied experiment archive and are not bundled. Obtain the corresponding UCI datasets and retain the specified filenames before rerunning benchmark clustering. The parser handles the source-specific formats and class definitions used in the experiments.

Numeric attributes with more than ten distinct non-missing values are converted to up to five quantile bins; duplicate edges are removed and recognized missing values form a separate category. Benchmark conversion precedes resampling. Each dataset uses its reference-class count for K. Individual class labels do not enter clustering or internal-validity calculations.

The released thirty-resample numerical results are sufficient to recompute average ranks, Friedman tests, Kendall concordance and paired comparisons without downloading the source archives.

## Method and evaluation settings

- Initializers: Random, Huang, Cao, Bai, K-Modes++, HR-2023 and CARE-KM.
- CARE-KM: complete Cao mode set, provisional-region candidate restriction, separation-based shortlists, **H = 16**, and at most **E = 4** positive-gain exchanges.
- Exchange acceptance minimizes global initial assignment cost, not DBI or silhouette.
- SMD is used for assignment, mode updating and categorical DBI. Silhouette uses normalized Hamming dissimilarity, an equivalent constant scaling of SMD.
- Standard K-Modes allows at most 100 iterations. Silhouette samples at most 2,000 observations without replacement with a shared seed in matched comparisons.
- Stochastic full-data initializers use 15 runs; deterministic initializers produce one partition for a fixed matrix.
- Ablation: twenty matched 80% LMS samples, E = 0, 1, 2 and 4. Robustness: thirty additional matched 80% samples.
- Main split: 4,660/1,998. Questionnaire split: 322/139. Both splits precede data-dependent fitting.
- Categorical NB uses alpha 1 and at least four categories per indicator. SVM uses an RBF kernel, C = 1 and gamma = scale. Questionnaire DT and SVM use balanced class weights; main-cohort classifiers use default weights. Hyperparameters are not tuned on test learners.

In the implementation, nearest-mode ties use the first mode. Category-mode ties retain the previous value when possible, then use sorted order. Empty clusters are repaired by moving distant observations from clusters with more than one member. Candidate screening uses stable ordering; a replacement must improve assignment cost. FSLSM direction uses mode evidence first, then member evidence to resolve ties; a dimension remains unresolved if both balances are zero.

## Running the analyses

Use Python 3.12 and install the pinned dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
python CODE/check_results.py
python CODE/run_questionnaire_holdout.py
python CODE/interpret_profiles.py
python CODE/analyze_lms.py
python CODE/analyze_benchmarks.py
```

`check_results.py` checks cohort sizes, categories, split membership, file checksums, Python syntax and the manuscript's holdout summaries against saved predictions. `run_questionnaire_holdout.py` refits CARE-KM and all three classifiers from the released categorical training data, then evaluates untouched test learners. `interpret_profiles.py` recomputes the directions and magnitudes of all 16 full-data clusters and checks them against Table 9. The statistical scripts recompute tests from the archived numerical experiment results.

Run new LMS clustering experiments with:

```bash
python CODE/run_lms_experiments.py --mode full
python CODE/run_lms_experiments.py --mode ablation
python CODE/run_lms_experiments.py --mode robustness
```

Run benchmarks after supplying the source archives:

```bash
python CODE/run_benchmarks.py
python CODE/analyze_benchmarks.py --raw OUTPUT/recomputed/benchmarks/benchmark_resample_raw.csv
```

Rebuild the main holdout after supplying the original numeric matrix with replacement identifiers matching `DATA/main_lms/split_membership.csv`:

```bash
python CODE/run_main_holdout.py --numeric path/to/main_lms_numeric.csv
```

The numeric input must contain `learner_id` and the 13 indicator columns in the categorical data file. Empirical thresholds and fallback decisions are fitted on its training rows only. The script does not fall back to full-data categories.

New computations are saved under `OUTPUT/recomputed/`, preserving the released experimental results. Each run recomputes the selected analysis; partial runs are not resumed. The pinned scikit-learn version is relevant to exact decision-tree predictions; later versions can change individual predictions even when rounded mean accuracy is unchanged.

## Results and manuscript correspondence

`OUTPUT/manuscript_tables/` follows the final manuscript's numbering, independently of the earlier internal experiment filenames:

| Table | Content |
|---|---|
| 1 | Ten benchmark datasets |
| 2 | Main LMS behavioral evidence directions |
| 3 | Full LMS clustering comparison |
| 4 | Exchange-depth ablation |
| 5 | Matched-resample LMS robustness |
| 6 | Average benchmark ranks |
| 7 | Friedman tests and Kendall concordance |
| 8 | Pairwise benchmark DBI and silhouette comparisons |
| 9 | Four-dimensional full-data profiles |
| 10 | Main LMS classifier reproduction |
| 11 | Questionnaire agreement and prediction performance |

These table CSVs preserve manuscript display values, including percentages and rounded numbers. Machine-readable unrounded experiment summaries and per-learner predictions are in the other output folders. Their accuracy and agreement fields use proportions from 0 to 1 unless the column explicitly contains formatted percentages. Full-data modes and mappings must not be confused with the separately fitted holdout modes and mappings.

## Interpretation and limitations

The 16 full-data clusters represent nine Base LS profiles; completeness of the sixteen theoretical combinations is not imposed. Learner-weighted dimensional agreement is 78.0% under resampling, 98.7% under member-first or member-only consolidation, and 72.7–79.3% under the reported preprocessing alternatives. Sensitivity summaries are released as recorded experiment outputs; no dedicated rerun drivers for those sensitivity analyses are included.

The combined refinement procedure is evaluated; region restriction and separation screening are not isolated component effects. Comparative runtime efficiency is not established. In LMS robustness tests, final-cost differences against Random and K-Modes++ and silhouette differences against Bai and K-Modes++ are not significant after joint Holm correction. Benchmark ranks describe ordering rather than improvement magnitudes on individual datasets.

Questionnaire correspondence is cohort-specific and uses one holdout split. Sequential/Global contains only 17 Sequential versus 122 Global test learners. High overall accuracy therefore coexists with limited minority-pole recall. These profiles do not establish psychometric equivalence or benefits from matching instruction to a learning-style profile.

## Attribution

Please cite the accompanying manuscript when using the CARE-KM implementation, processed learner data or experimental outputs. The main LMS source and UCI datasets retain their respective source attribution and distribution conditions. This repository does not assign a new license to those external datasets.
