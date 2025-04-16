import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # Import Random Forest
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Ignore potential warnings

# Helper function to get feature names (same as in logreg version)
def get_feature_names(config):
    data_cfg = config['data_params']
    num_cols = data_cfg['phenotype_cols_numerical']
    cat_cols = data_cfg['phenotype_cols_categorical']
    num_regions = data_cfg['num_regions']
    num_fc_features = num_regions * (num_regions - 1) // 2
    fc_names = [f"FC_{i}" for i in range(num_fc_features)]
    pheno_names = []
    pheno_names.extend(num_cols)
    try:
        df_pheno_temp = pd.read_csv(data_cfg['phenotype_file'], usecols=cat_cols + [data_cfg['sub_id_col']]) # Need ID for map
        df_pheno_temp = df_pheno_temp.replace([-9999, -9999.0, ''], np.nan)
        # Need imputation before getting categories if NaNs exist
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_pheno_temp[cat_cols] = df_pheno_temp[cat_cols].astype(str)
        df_pheno_temp[cat_cols] = cat_imputer.fit_transform(df_pheno_temp[cat_cols])
        # Get unique categories after imputation to build feature names
        ohe_feature_names_list = []
        for col in cat_cols:
            cats = df_pheno_temp[col].unique()
            ohe_feature_names_list.extend([f"{col}_{cat}" for cat in sorted(cats)])
        pheno_names.extend(ohe_feature_names_list)
    except Exception as e:
        print(f"Warning: Could not reliably get OHE feature names: {e}. Using generic names.")
        pheno_cat_count = 2 + 20 + 2 # Approx count for Sex, Site, EyeStatus
        pheno_names.extend([f"PhenoCat_{i}" for i in range(pheno_cat_count)])
    return fc_names + pheno_names

def get_region_pair_from_fc_index(fc_index, num_regions):
    """Maps a flattened upper triangle index back to the region pair."""
    if fc_index < 0: return None
    indices = np.triu_indices(num_regions, k=1)
    if fc_index >= len(indices[0]): return None
    row_idx = indices[0][fc_index]
    col_idx = indices[1][fc_index]
    return (row_idx, col_idx)

def main():
    start_time = time.time()
    # --- Load Configuration ---
    config_filename = 'config_baseline_rf.yaml'
    print(f"Loading configuration from {config_filename}...")
    try:
        with open(config_filename, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {config_filename}: {e}")
        return

    data_cfg = config['data_params']
    run_cfg = config['run_params']
    rf_cfg = config['random_forest_params'] # Get RF params
    rfe_cfg = config['feature_selection_params']

    output_dir = data_cfg['output_feature_dir']
    feature_path = os.path.join(output_dir, 'features.npy')
    label_path = os.path.join(output_dir, 'labels.npy')
    subjects_path = os.path.join(output_dir, 'subject_ids.npy')

    # --- Load Preprocessed Data ---
    print("Loading preprocessed features and labels...")
    try:
        X = np.load(feature_path)
        y = np.load(label_path).astype(int)
        subject_ids = np.load(subjects_path)
        if X.shape[0] != y.shape[0] or X.shape[0] != len(subject_ids):
             raise ValueError("Mismatch in loaded array lengths.")
        print(f" Feature shape: {X.shape}, Label shape: {y.shape}")
        print(f" Label distribution: {np.bincount(y)}")
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return

    # --- Get Feature Names ---
    print("Attempting to retrieve feature names...")
    feature_names = get_feature_names(config)
    if len(feature_names) != X.shape[1]:
        print(f"Warning: Feature name count ({len(feature_names)}) mismatch data columns ({X.shape[1]}). Using generic names.")
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # --- Cross-Validation Setup ---
    skf = StratifiedKFold(n_splits=run_cfg['n_splits'], shuffle=True, random_state=run_cfg['seed'])
    fold_results = []
    all_feature_importances = [] # Store importances from each fold

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        print(f"\n--- Fold {fold+1}/{run_cfg['n_splits']} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Pipeline Steps ---
        steps = []

        # 1. Scaling (Optional for RF, but good practice if using RFE)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_proc, X_test_proc = X_train_scaled, X_test_scaled
        current_feature_names = np.array(feature_names) # Keep track of names

        # 2. Feature Selection (RFE) - Optional
        if rfe_cfg['use_rfe']:
            print(f"Applying RFE to select {rfe_cfg['n_features_to_select']} features...")
            # Use RandomForest itself or a simpler model for RFE estimator
            estimator_rfe = RandomForestClassifier(
                n_estimators=50, # Fewer trees for speed
                max_depth=5,
                random_state=run_cfg['seed'],
                n_jobs=-1,
                class_weight='balanced'
             )
            n_features = X_train_proc.shape[1]
            n_select = rfe_cfg['n_features_to_select']
            if n_select > n_features: n_select = n_features
            if n_select <= 0: n_select = 1

            selector = RFE(estimator=estimator_rfe,
                           n_features_to_select=n_select,
                           step=rfe_cfg.get('rfe_step', 0.1),
                           verbose=0)
            try:
                print(" Fitting RFE...")
                selector.fit(X_train_proc, y_train)
                selected_indices = selector.get_support(indices=True)
                print(" Transforming features with RFE...")
                X_train_proc = selector.transform(X_train_proc)
                X_test_proc = selector.transform(X_test_proc)
                current_feature_names = current_feature_names[selected_indices] # Filter feature names
                print(f" RFE finished. Selected features shape: {X_train_proc.shape}")
            except Exception as e:
                print(f"Error during RFE: {e}. Skipping RFE for this fold.")
                # Revert
                X_train_proc, X_test_proc = X_train_scaled, X_test_scaled
                current_feature_names = np.array(feature_names)
        else:
            print("Skipping RFE.")

        # 3. Random Forest Classifier
        print("Training Random Forest Classifier...")
        rf_classifier = RandomForestClassifier(
            n_estimators=rf_cfg['n_estimators'],
            max_depth=rf_cfg['max_depth'],
            min_samples_split=rf_cfg['min_samples_split'],
            min_samples_leaf=rf_cfg['min_samples_leaf'],
            max_features=rf_cfg['max_features'],
            criterion=rf_cfg['criterion'],
            class_weight=rf_cfg['class_weight'],
            random_state=run_cfg['seed'],
            n_jobs=rf_cfg.get('n_jobs', -1),
            oob_score=rf_cfg.get('oob_score', False)
        )

        try:
             rf_classifier.fit(X_train_proc, y_train)
             # Store feature importances after fitting
             all_feature_importances.append(
                 pd.Series(rf_classifier.feature_importances_, index=current_feature_names)
             )
        except Exception as e:
             print(f"Error fitting RandomForest in Fold {fold+1}: {e}")
             fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0})
             continue # Skip to next fold

        # --- Evaluation ---
        print("Evaluating on test set...")
        y_pred = rf_classifier.predict(X_test_proc)
        y_proba = rf_classifier.predict_proba(X_test_proc)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        auc = 0.0
        if len(np.unique(y_test)) > 1:
            try: auc = roc_auc_score(y_test, y_proba)
            except ValueError: auc = 0.0
        else: auc = 0.0

        fold_metrics = {'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'auc': auc}
        fold_results.append(fold_metrics)

        print(f"Fold {fold+1} Results: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
        fold_end_time = time.time()
        print(f"Fold {fold+1} elapsed time: {fold_end_time - fold_start_time:.2f} seconds.")

    # --- Aggregate and Print Results ---
    print("\n--- Cross-Validation Results ---")
    # (Result aggregation code...)
    if fold_results:
        valid_results = [res for res in fold_results if res and res.get('acc', -1) > -1]
        num_valid_folds = len(valid_results)
        print(f"Number of successfully completed folds: {num_valid_folds}/{run_cfg['n_splits']}")
        if num_valid_folds > 0:
            avg_acc = np.mean([res.get('acc', 0) for res in valid_results])
            std_acc = np.std([res.get('acc', 0) for res in valid_results])
            avg_f1 = np.mean([res.get('f1', 0) for res in valid_results])
            avg_prec = np.mean([res.get('precision', 0) for res in valid_results])
            avg_rec = np.mean([res.get('recall', 0) for res in valid_results])
            avg_auc = np.mean([res.get('auc', 0) for res in valid_results])
            print(f'Average Accuracy:  {avg_acc:.4f} +/- {std_acc:.4f}')
            print(f'Average F1-Score:  {avg_f1:.4f}')
            print(f'Average Precision: {avg_prec:.4f}')
            print(f'Average Recall:    {avg_rec:.4f}')
            print(f'Average AUC:       {avg_auc:.4f}')
        else: print("No folds completed successfully.")
    else: print("No results recorded.")

    # --- Feature Importance Analysis ---
    if all_feature_importances:
        print("\n--- Average Feature Importances (Random Forest Gini Importance) ---")
        # Average importance across folds (handle potentially different feature sets if RFE used per fold)
        # For simplicity, average importances only if RFE is OFF
        if not rfe_cfg['use_rfe']:
            avg_importances_series = pd.concat(all_feature_importances, axis=1).mean(axis=1)
            avg_importances_series = avg_importances_series.sort_values(ascending=False)

            num_fc_features = data_cfg['num_regions'] * (data_cfg['num_regions'] - 1) // 2

            print(f"\nTop 20 Most Important Features Overall:")
            for i, (feat_name, imp) in enumerate(avg_importances_series.head(20).items()):
                 if feat_name.startswith("FC_"):
                      try:
                           fc_index = int(feat_name.split("_")[1])
                           region_pair = get_region_pair_from_fc_index(fc_index, data_cfg['num_regions'])
                           feature_type = f"FC between regions {region_pair}" if region_pair else f"FC Feature {fc_index}"
                      except:
                           feature_type = feat_name # Fallback
                 else:
                      feature_type = f"Phenotype: {feat_name}"

                 print(f" {i+1}. {feature_type} - Importance: {imp:.4f}")

            print("\n(Note: Mapping FC indices back to region pairs provides interpretability)")
        else:
            print("Average feature importance across folds not calculated when RFE is enabled (selected features may differ per fold).")


    end_time = time.time()
    print(f"\nTotal elapsed time: {end_time - start_time:.2f} seconds.")
    print("--- Baseline Random Forest Run Finished ---")

if __name__ == "__main__":
    main()