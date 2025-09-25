def process_file(file: str):
    """
    Run machine learning pipeline on a single feature CSV file.

    Steps:
    1. Load pre-extracted features (tsfresh output) and preprocess (split participant/video/segment).
    2. Downcast numeric columns, drop NaNs/unnecessary columns.
    3. Separate baseline vs experimental segments and compute video-level median features.
    4. Balance class distribution with SMOTE at video level.
    5. Normalize signals relative to baseline.
    6. Merge participant demographics.
    7. Apply filtering (best_of_cat, mismatch, subjective/objective ratings).
    8. Train/test split with GroupShuffleSplit to prevent participant leakage.
    9. Scale features, apply SMOTE again on training set.
    10. Feature selection using Recursive Feature Elimination (RFE) → 50 features.
    11. Train Random Forest, XGBoost, SVM, and a VotingClassifier ensemble with hyperparameter tuning.
    12. Evaluate models on validation and test sets with classification reports, confusion matrices,
        class distributions, and bootstrapped confidence intervals for macro F1-score.

    Assumptions:
    - Input features were extracted beforehand (e.g., using tsfresh).
    - The input CSV must contain `participant_video_segment`, baseline videos,
      and demographic information will be merged from external file.

    Parameters
    ----------
    file : str
        Filename of the feature CSV to process.

    Returns
    -------
    None
        Prints model performance reports and statistics to console.
    """
    import warnings
    warnings.filterwarnings('ignore')

    import os
    import json
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.feature_selection import RFE
    from imblearn.over_sampling import SMOTE
    from collections import Counter

    try:
        # ---------------- Load and preprocess ---------------- #
        directory = r"G:\Montreal Science Center JG\Features Divided QC Ten Seconds New Temp Z Score Individually\reduced"
        filepath = os.path.join(directory, file)
        df = pd.read_csv(filepath)

        # Parse participant/video/segment IDs
        parts = df['participant_video_segment'].astype(str).str.split("_", expand=True)
        df['video_id'] = parts[0]
        df['participant'] = parts[1] + "_" + parts[2]
        df['segment'] = parts[3]
        df['participant_video'] = df['participant'] + "_" + df['video_id']

        # Numeric preprocessing
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, downcast='float')
        df = df.dropna(subset=numeric_cols)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            numeric_cols = numeric_cols.drop('Unnamed: 0')

        # ---------------- Aggregate features ---------------- #
        experimental_df = df[df['video_id'] != 'baseline'].copy()
        baseline_df = df[df['video_id'] == 'baseline'].copy()

        video_level_features = (
            experimental_df.groupby('participant_video')[numeric_cols].median().reset_index()
        )
        video_level_features['participant'] = video_level_features['participant_video'].apply(
            lambda x: '_'.join(x.split('_')[:-1]))
        video_level_features['video_id'] = video_level_features['participant_video'].apply(
            lambda x: x.split('_')[-1])

        # Encode video_id → label
        le = LabelEncoder()
        video_level_features['label'] = le.fit_transform(video_level_features['video_id'])
        X = video_level_features[numeric_cols].values.astype(np.float32)
        y = video_level_features['label'].values

        # Balance with SMOTE
        print(f"\n📊 Class balance before SMOTE (video-level): {dict(Counter(y))}")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"📈 Class balance after SMOTE (video-level):  {dict(Counter(y_resampled))}")

        # Combine original + resampled
        df_resampled = pd.DataFrame(X_resampled, columns=numeric_cols)
        df_resampled['video_id'] = le.inverse_transform(y_resampled)
        df_resampled['participant'] = [f'smote{i}' for i in range(len(df_resampled))]
        df_resampled['participant_video'] = df_resampled['participant'] + "_" + df_resampled['video_id']
        df_combined = pd.concat([video_level_features.drop(columns=['label']), df_resampled], ignore_index=True)

        # Compute baseline-normalized signals
        video_values = df_combined.groupby("participant_video")[numeric_cols].median().astype(np.float32)
        participant_info = df_combined[['participant_video', 'participant']].drop_duplicates()
        participant_video_avg = video_values.reset_index().merge(participant_info, on='participant_video')
        baseline_avg = baseline_df.groupby("participant")[numeric_cols].median().astype(np.float32).reset_index()
        merged = participant_video_avg.merge(baseline_avg, on='participant', suffixes=('_video', '_baseline'))

        signals_diff = (
            merged[[col + '_video' for col in numeric_cols]].values.astype(np.float32) -
            merged[[col + '_baseline' for col in numeric_cols]].values.astype(np.float32)
        )
        signals = pd.DataFrame(signals_diff, columns=numeric_cols)
        signals['participant_video'] = merged['participant_video'].astype(str)
        signals['video_id'] = signals['participant_video'].apply(lambda x: x.split('_')[-1])

        # ---------------- Merge demographics ---------------- #
        json_df = pd.read_csv("G:/Montreal Science Center JG/demographics/2025-05-13/participant_demographic_data.csv")
        json_df['participant_video'] = json_df['participant_video'].astype(str)
        df = signals.merge(json_df, on='participant_video')

        # Drop unwanted features
        cols_to_drop = [col for col in df.columns if "TMP" in col and "ar_coefficient" in col]
        cols_to_drop.append('EDA_cleaned_normalized__variation_coefficient')
        df = df.drop(columns=cols_to_drop, errors='ignore')
        numeric_cols = [col for col in numeric_cols if col in df.columns]

        # ---------------- Filtering ---------------- #
        df = df[df['best_of_cat'] == 1]
        df = df[df['subjective_rating'] != 0]
        df = df[df['mismatch'] == 1]
        df['objective_rating'] = df['objective_rating'].apply(lambda x: 0 if x == 4 else 1)

        target_class = 'SAD VS ALL'
        print(target_class)
        label = 'objective_rating'
        X = df[numeric_cols].astype(np.float32)
        y = df[label]
        groups = df['participant']
        print('label: ', label)

        # ---------------- Train/validation/test split ---------------- #
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_val_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train_val_raw, X_test_raw = X.iloc[train_val_idx], X.iloc[test_idx]
        y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
        groups_train_val = groups.iloc[train_val_idx]

        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, val_idx = next(gss_val.split(X_train_val_raw, y_train_val, groups=groups_train_val))
        X_train_raw, X_val_raw = X_train_val_raw.iloc[train_idx], X_train_val_raw.iloc[val_idx]
        y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

        # ---------------- Scaling + SMOTE ---------------- #
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        X_train = pd.DataFrame(X_train_scaled, columns=numeric_cols, index=X_train_raw.index)
        X_val = pd.DataFrame(X_val_scaled, columns=numeric_cols, index=X_val_raw.index)
        X_test = pd.DataFrame(X_test_scaled, columns=numeric_cols, index=X_test_raw.index)

        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        print(f"\n📊 Class balance before SMOTE (final training): {dict(Counter(y_train))}")
        print(f"📈 Class balance after SMOTE (final training):  {dict(Counter(y_train_bal))}")

        # ---------------- Feature Selection: RFE ---------------- #
        print("\n🔍 Running RFE for feature selection (50 features)...")
        rfe_selector = RFE(
            estimator=RandomForestClassifier(random_state=42),
            step=10,
            n_features_to_select=50,
            verbose=1
        )
        rfe_selector.fit(X_train_bal, y_train_bal)
        selected_feature_names = [col for col, keep in zip(numeric_cols, rfe_selector.support_) if keep]
        print(f"✅ Selected {len(selected_feature_names)} features via RFE.")

        X_train_selected = X_train_bal[selected_feature_names]
        X_val_selected = X_val[selected_feature_names]
        X_test_selected = X_test[selected_feature_names]

        # ---------------- Classifiers + grids ---------------- #
        base_classifiers = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        ensemble_classifiers = {
            'Voting Classifier': VotingClassifier(
                estimators=[(name, clf) for name, clf in base_classifiers.items()],
                voting='soft')
        }
        param_grids = {
            'Random Forest': {'n_estimators': [100, 300, 500], 'max_depth': [10, 20, 50, 100]},
            'XGBoost': {'n_estimators': [100, 200, 400], 'learning_rate': [0.01, 0.1, 0.75, 1], 'max_depth': [6, 12, 24]},
            'SVM': {'C': [1, 10, 100], 'kernel': ['linear', 'rbf']}
        }
        all_classifiers = {**base_classifiers, **ensemble_classifiers}

        # ---------------- Helpers ---------------- #
        def bootstrap_f1_ci(y_true, y_pred, n_bootstrap=1000, average='macro'):
            rng = np.random.RandomState(42)
            scores = []
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            for _ in range(n_bootstrap):
                idx = rng.choice(len(y_true), len(y_true), replace=True)
                y_t, y_p = y_true[idx], y_pred[idx]
                if len(np.unique(y_t)) < 2:
                    continue
                scores.append(f1_score(y_t, y_p, average=average))
            return np.median(scores), (np.percentile(scores, 2.5), np.percentile(scores, 97.5))

        def print_class_distribution(y_train, y_val, y_test):
            table = pd.DataFrame({
                'Train': y_train.value_counts(),
                'Validation': y_val.value_counts(),
                'Test': y_test.value_counts()
            }).fillna(0).astype(int)
            print("\n📊 Sample Count Per Class (Train / Validation / Test):")
            print(table.to_markdown())

        # ---------------- Training loop ---------------- #
        for name, clf in all_classifiers.items():
            print(f"\n🧪 Training: {name}")
            param_grid = param_grids.get(name, None)

            if param_grid:
                grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
                grid_search.fit(X_train_selected, y_train_bal)
                clf = grid_search.best_estimator_
                print(f"🔍 Best params for {name}: {grid_search.best_params_}")
            else:
                clf.fit(X_train_selected, y_train_bal)

            val_preds = clf.predict(X_val_selected)
            test_preds = clf.predict(X_test_selected)

            print(f"\n📋 {name} Validation Report:")
            print(classification_report(y_val, val_preds))
            print("🧮 Validation Confusion Matrix:")
            print(confusion_matrix(y_val, val_preds))

            print(f"\n📋 {name} Test Report:")
            print(classification_report(y_test, test_preds))
            print("🧮 Test Confusion Matrix:")
            print(confusion_matrix(y_test, test_preds))

            print_class_distribution(y_train, y_val, y_test)

            # Bootstrap CI for test macro F1
            f1_median, (ci_low, ci_high) = bootstrap_f1_ci(y_test, test_preds)
            print(f"\n📈 Test Macro F1-score (bootstrap median): {f1_median:.4f}")
            print(f"📉 95% CI for Macro F1-score: ({ci_low:.4f}, {ci_high:.4f})")

    except Exception as e:
        print(f"❌ Error processing file {file}: {str(e)}")
