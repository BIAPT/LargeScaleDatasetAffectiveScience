# LargeScaleDatasetAffectiveScience
QC & ML Pipeline for Physiological Signal Dataset

This repository contains code for quality control (QC), preprocessing, and machine learning analysis of a large physiological signal dataset (EDA, BVP, TMP). It includes:

QC & preprocessing scripts: Clean signals, handle baseline/video conditions, normalize, and output processed CSVs.

ML pipeline: Trains Random Forest, XGBoost, SVM, and a VotingClassifier ensemble on pre-extracted features. Performs feature selection (RFE), SMOTE balancing, baseline normalization, train/validation/test splits, and evaluation with classification reports, confusion matrices, and bootstrapped F1 confidence intervals.

A Filepaths.csv is included for easy configuration of input/output directories.

This work is part of Girgis et al. (in submission) and supports reproducible research via OSF.

Please visit https://osf.io/pn9rq/?view_only=25f185f049434b18a7356d85e7f81d5c to access the dataset files.
