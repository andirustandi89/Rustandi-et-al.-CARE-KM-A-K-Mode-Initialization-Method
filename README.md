# CARE-KM Reproducibility Materials

This repository contains the data, Python code, and experimental outputs accompanying the study:

**CARE-KM: A K-Modes Initialization Method for Categorical Clustering with a Felder–Silverman Learning-Style Application**

The repository is organized to support inspection and reproduction of the clustering, FSLSM interpretation, robustness, and prediction experiments reported in the manuscript.

## Repository Structure

```text
CARE-KM-Reproducibility/
├── Data/
├── Code/
├── Output/
└── README.md
```

## Data

The `Data` directory contains the processed learner-level datasets, train-test split information, and frozen artifacts used in the experiments.

```text
Data/
├── main_lms_numeric.csv
├── main_lms_categorical.csv
├── groundtruth_numeric.csv
├── groundtruth_categorical.csv
├── groundtruth_ils_labels.csv
├── main_lms_split_indices.csv
├── groundtruth_split_indices.csv
├── main_lms_frozen_modes.csv
├── main_lms_frozen_semantic_mapping.csv
├── groundtruth_frozen_modes.csv
└── groundtruth_frozen_semantic_mapping.csv
```

### Main LMS dataset

`main_lms_numeric.csv` contains learner-level behavioral indicators before categorical discretization.

`main_lms_categorical.csv` contains the categorical behavioral representation used as input to K-Modes and CARE-KM.

The dataset contains **6,658 learners**.

### Questionnaire-grounded dataset

The questionnaire-grounded cohort is represented by:

- `groundtruth_numeric.csv`
- `groundtruth_categorical.csv`
- `groundtruth_ils_labels.csv`

The cohort contains **461 learners** from a **Computer Education Department at an anonymized higher education institution in Indonesia**.

Student identities and institutional identity have been anonymized. Research-specific anonymous learner identifiers are used consistently across the released files.

The ILS-derived labels are stored separately from the behavioral features and are not included as clustering attributes.

## Code

The `Code` directory contains the Python implementation and experiment scripts.

```text
Code/
├── 01_preprocess_main_lms.py
├── 02_preprocess_groundtruth.py
├── carekm_core.py
├── 03_full_lms_experiment.py
├── 04_ablation_robustness.py
├── 05_cross_dataset_evaluation.py
├── 06_fslsm_interpretation.py
├── 07_semantic_sensitivity.py
├── 08_main_lms_holdout_prediction.py
├── 09_groundtruth_holdout_evaluation.py
└── 10_generate_paper_tables.py
```

### Script description

**`01_preprocess_main_lms.py`**  
Processes the learner-level behavioral indicators used for the main LMS experiment.

**`02_preprocess_groundtruth.py`**  
Processes the questionnaire-grounded behavioral dataset and its categorical representation.

**`carekm_core.py`**  
Contains the core CARE-KM implementation, categorical dissimilarity functions, initialization procedure, and K-Modes-related utilities.

**`03_full_lms_experiment.py`**  
Runs the full LMS clustering comparison among the evaluated initialization methods.

**`04_ablation_robustness.py`**  
Runs the CARE-KM ablation and matched-resampling robustness experiments.

**`05_cross_dataset_evaluation.py`**  
Runs the evaluation on the categorical benchmark datasets.

**`06_fslsm_interpretation.py`**  
Applies the post-clustering Felder-Silverman learning-style interpretation procedure.

**`07_semantic_sensitivity.py`**  
Runs the semantic stability and preprocessing sensitivity analyses.

**`08_main_lms_holdout_prediction.py`**  
Runs the untouched 70:30 prediction experiment on the main LMS cohort.

**`09_groundtruth_holdout_evaluation.py`**  
Runs the untouched 70:30 questionnaire-grounded evaluation against ILS-derived ground truth.

**`10_generate_paper_tables.py`**  
Reads the experimental outputs and prepares the principal result tables.

## Output

The `Output` directory contains the experiment results associated with the analyses reported in the manuscript.

```text
Output/
├── classifier_settings.csv
├── full_lms_clustering_results.csv
├── ablation_results.csv
├── robustness_results.csv
├── cross_dataset_results.csv
├── fslsm_cluster_profiles.csv
├── semantic_stability_results.csv
├── preprocessing_sensitivity_results.csv
├── main_lms_holdout_predictions.csv
└── groundtruth_holdout_predictions.csv
```

## Untouched Holdout Design

The prediction experiments use a 70:30 train-test design in which the split is performed before data-dependent fitting.

### Main LMS cohort

```text
6,658 learners
├── 4,660 training learners
└── 1,998 untouched test learners
```

Training-dependent preprocessing parameters, CARE-KM modes, semantic mappings, and classifier parameters are estimated from the training portion and then frozen before the untouched test set is processed.

### Questionnaire-grounded cohort

```text
461 learners
├── 322 training learners
└── 139 untouched test learners
```

The ILS-derived labels are not used to form CARE-KM clusters or determine their semantic identities. They are used for split stratification and external evaluation.

## Running the Code

The scripts are designed to run from the repository root directory using relative paths.

Examples:

```bash
python Code/03_full_lms_experiment.py
```

Run the main LMS holdout experiment:

```bash
python Code/08_main_lms_holdout_prediction.py
```

Run the questionnaire-grounded evaluation:

```bash
python Code/09_groundtruth_holdout_evaluation.py
```

Generate the principal result tables:

```bash
python Code/10_generate_paper_tables.py
```

Input files are read from `Data/`, while experiment results are stored or reproduced in `Output/`.

## Experimental Separation

CARE-KM modifies the initialization stage of K-Modes while retaining the standard categorical assignment and mode-update procedure.

FSLSM interpretation is performed only after the empirical clustering partition has been obtained. Learning-style evidence therefore does not participate in the clustering objective.

The main LMS holdout experiment evaluates the prospective reproducibility of the frozen train-derived interpretation.

The questionnaire-grounded experiment uses independently obtained ILS-derived labels as an external evaluation criterion.

## Data Privacy

The questionnaire-grounded dataset was anonymized before public release.

The released data do not contain:

- student names;
- student numbers;
- Moodle usernames;
- email addresses;
- IP addresses;
- telephone numbers;
- original Moodle user identifiers; or
- the identity of the higher education institution.

Anonymous research identifiers are retained only to preserve the relationships among behavioral records, train-test assignments, and ILS-derived labels.

## Third-Party Data

Third-party raw datasets are not redistributed in this repository.

The main LMS dataset was reconstructed from the publicly available Moodle database associated with the `moodle-early-performance-prediction` repository.

The categorical benchmark datasets used in the cross-dataset evaluation are available from the UCI Machine Learning Repository.

Users should obtain third-party source data from their original providers and comply with the corresponding access and licensing conditions.

## Citation

If you use these materials, please cite the accompanying CARE-KM article:

**CARE-KM: A K-Modes Initialization Method for Categorical Clustering with a Felder–Silverman Learning-Style Application**

## Authors

- Andi Rustandi
- Sri Mulyana
- Sigit Priyanta
