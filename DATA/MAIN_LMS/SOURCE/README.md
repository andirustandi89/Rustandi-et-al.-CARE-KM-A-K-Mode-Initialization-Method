# Main LMS raw source

The raw Moodle dump is not duplicated in this archive. Obtain the source from the upstream project:
https://github.com/moisesriestra/moodle-early-performance-prediction

The study preprocessing starts from the `mysql-export` Moodle dump referenced by that repository. Place the dump in this folder using the filename `mysql-export` before running `CODE/01_raw_moodle_to_numeric.py`.

The public archive already contains the anonymized learner-level numeric and categorical matrices actually used by the manuscript. Original Moodle user identifiers are not released.
