import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression # For faster RFE if needed
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Helper function to get feature names (simplified)
def get_feature_names(config):
    data_cfg = config['data_params']
    num_cols = data_cfg.get('phenotype_cols_numerical', [])
    cat_cols = data_cfg.get('phenotype_cols_categorical', [])
    num_regions = data_cfg['num_regions']
    num_fc_features = num_regions * (num_regions - 1) // 2
    fc_names = [f"FC_{i}" for i in range(num_fc_features)]
    pheno_names = []
    pheno_names.extend(num_cols)
    try:
        # This part is still an estimation
        df_pheno_temp = pd.read_csv(data_cfg['phenotype_file'], usecols=cat_cols + [data_cfg['sub_id_col']])
        df_pheno_temp = df_pheno_temp.replace([-9999, -9999.0, ''], np.nan)
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_pheno_temp[cat_cols] = df_pheno_temp[cat_cols].astype(str)
        df_pheno_temp[cat_cols] = cat_imputer.fit_transform(df_pheno_temp[cat_cols])
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(df_pheno_temp[cat_cols])
        pheno_names.extend(ohe.get_feature_names_out(cat_cols))
    except Exception as e:
        print(f"Warning: Could not reliably get OHE feature names: {e}. Using generic names.")
        # Estimate based on known columns
        ohe_count = 0
        if 'SEX' in cat_cols: ohe_count += 2
        if 'SITE_ID' in cat_cols: ohe_count += 20 # Assuming 20 sites
        if 'EYE_STATUS_AT_SCAN' in cat_cols: ohe_count += 2 # Assuming 2 statuses
        pheno_names.extend([f"PhenoCat_{i}" for i in range(ohe_count)])

    return fc_names + pheno_names

def get_region_pair_from_fc_index(fc_index, num_regions):
    if fc_index < 0: return None
    indices = np.triu_indices(num_regions, k=1)
    if fc_index >= len(indices[0]): return None
    row_idx = indices[0][fc_index]
    col_idx = indices[1][fc_index]
    return (row_idx, col_idx)

def main():
    start_time = time.time()
    config_filename = 'config_baseline_xgboost.yaml'
    print(f"Loading configuration from {config_filename}...")
    try:
        with open(config_filename, 'r') as f: config = yaml.safe_load(f)
    except Exception as e: print(f"Error loading {config_filename}: {e}"); return

    data_cfg = config['data_params']
    run_cfg = config['run_params']
    xgb_cfg = config['xgboost_params']
    rfe_cfg = config['feature_selection_params']

    feature_dir = data_cfg['output_feature_dir']
    feature_path = os.path.join(feature_dir, 'features.npy')
    label_path = os.path.join(feature_dir, 'labels.npy')
    subjects_path = os.path.join(feature_dir, 'subject_ids.npy')

    print("Loading preprocessed features and labels...")
    try:
        X = np.load(feature_path)
        y = np.load(label_path).astype(int)
        subject_ids = np.load(subjects_path)
        if X.shape[0] != y.shape[0] or X.shape[0] != len(subject_ids): raise ValueError("Data length mismatch.")
        print(f" Feature shape: {X.shape}, Label shape: {y.shape}")
        print(f" Label distribution: {np.bincount(y)}")
    except Exception as e: print(f"Error loading data: {e}. Run preprocess_baseline.py first."); return

    feature_names_list = get_feature_names(config)
    if len(feature_names_list) != X.shape[1]:
        print(f"Warning: Feature name count mismatch ({len(feature_names_list)} vs {X.shape[1]}). Using generic names.")
        feature_names_list = [f"feature_{i}" for i in range(X.shape[1])]
    feature_names = np.array(feature_names_list)

    n_splits = run_cfg['n_splits']
    if X.shape[0] < n_splits: n_splits = max(2, X.shape[0])
    if n_splits < 2: print("Error: Not enough samples."); return
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run_cfg['seed'])
    fold_results = []
    all_fold_importances = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_proc, X_test_proc = X_train_scaled, X_test_scaled
        current_feature_names = feature_names

        if rfe_cfg.get('use_rfe', False):
            # ... (RFE Logic - keep as is, preferably using LogisticRegression estimator for speed) ...
            n_features_in = X_train_proc.shape[1]
            n_select = int(rfe_cfg.get('n_features_to_select', 1000))
            if n_select > n_features_in: n_select = n_features_in
            if n_select <= 0: n_select = 1
            print(f"Applying RFE to select {n_select} features...")
            estimator_rfe = LogisticRegression(solver='liblinear', C=0.1, penalty='l1', max_iter=500, random_state=run_cfg['seed'])
            selector = RFE(estimator=estimator_rfe, n_features_to_select=n_select, step=rfe_cfg.get('rfe_step', 0.1), verbose=0)
            try:
                selector.fit(X_train_proc, y_train)
                selected_indices = selector.get_support(indices=True)
                X_train_proc = selector.transform(X_train_proc)
                X_test_proc = selector.transform(X_test_proc)
                current_feature_names = feature_names[selected_indices]
                print(f" RFE finished. Selected features shape: {X_train_proc.shape}")
            except Exception as e:
                print(f"Error during RFE: {e}. Skipping RFE for this fold.")
                X_train_proc, X_test_proc = X_train_scaled, X_test_scaled
                current_feature_names = feature_names
        else:
            print("Skipping RFE.")

        print("Training XGBoost Classifier...")
        xgb_classifier = xgb.XGBClassifier(
            objective=xgb_cfg.get('objective', 'binary:logistic'),
            eval_metric=xgb_cfg.get('eval_metric', 'auc'),
            use_label_encoder=xgb_cfg.get('use_label_encoder', False),
            n_estimators=xgb_cfg.get('n_estimators', 100),
            learning_rate=xgb_cfg.get('learning_rate', 0.1),
            max_depth=xgb_cfg.get('max_depth', 3),
            subsample=xgb_cfg.get('subsample', 1.0),
            colsample_bytree=xgb_cfg.get('colsample_bytree', 1.0),
            gamma=xgb_cfg.get('gamma', 0),
            reg_alpha=xgb_cfg.get('reg_alpha', 0),
            reg_lambda=xgb_cfg.get('reg_lambda', 1),
            tree_method=xgb_cfg.get('tree_method', 'auto'),
            random_state=run_cfg['seed'],
            n_jobs=xgb_cfg.get('n_jobs', -1)
        )

        try:
            # *** FIX: REMOVE early stopping args from fit ***
            xgb_classifier.fit(
                X_train_proc, y_train,
                # early_stopping_rounds=... # REMOVED
                # eval_set=...            # REMOVED
                # callbacks=...           # REMOVED
                verbose=False           # Keep verbose False
            )
            # ***********************************************

            # Store feature importances (only makes sense if RFE is off)
            if not rfe_cfg.get('use_rfe', False):
                 if hasattr(xgb_classifier, 'feature_importances_'):
                      if len(current_feature_names) == len(xgb_classifier.feature_importances_):
                           importances = pd.Series(xgb_classifier.feature_importances_, index=current_feature_names)
                           all_fold_importances.append(importances)
                      else: print("Warning: Importances length mismatch.")
                 else: print("Warning: Could not get feature importances.")

        except Exception as e:
             print(f"Error fitting XGBoost in Fold {fold+1}: {e}")
             fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0})
             continue # Important to skip to next fold on error

        # --- Evaluation ---
        print("Evaluating on test set...")
        try:
            y_pred = xgb_classifier.predict(X_test_proc)
            y_proba = xgb_classifier.predict_proba(X_test_proc)[:, 1]
        except Exception as e:
            print(f"Error during prediction in Fold {fold+1}: {e}")
            fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0})
            continue

        # Calculate metrics
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
    # ... (Result aggregation code remains the same) ...
    if fold_results:
        valid_results = [res for res in fold_results if res and res.get('acc', -1) > -1]
        num_valid_folds = len(valid_results)
        print(f"Number of successfully completed folds: {num_valid_folds}/{n_splits}")
        if num_valid_folds > 0:
            avg_acc = np.mean([res.get('acc', 0) for res in valid_results]); std_acc = np.std([res.get('acc', 0) for res in valid_results])
            avg_f1 = np.mean([res.get('f1', 0) for res in valid_results]); avg_prec = np.mean([res.get('precision', 0) for res in valid_results])
            avg_rec = np.mean([res.get('recall', 0) for res in valid_results]); avg_auc = np.mean([res.get('auc', 0) for res in valid_results])
            print(f'Average Accuracy:  {avg_acc:.4f} +/- {std_acc:.4f}'); print(f'Average F1-Score:  {avg_f1:.4f}')
            print(f'Average Precision: {avg_prec:.4f}'); print(f'Average Recall:    {avg_rec:.4f}'); print(f'Average AUC:       {avg_auc:.4f}')
        else: print("No folds completed successfully.")
    else: print("No results recorded.")


    # --- Feature Importance Analysis ---
    if all_fold_importances:
        print("\n--- Average Feature Importances (from XGBoost) ---")
        try:
            avg_importances_series = pd.concat(all_fold_importances, axis=1).mean(axis=1)
            avg_importances_series = avg_importances_series.sort_values(ascending=False)
            num_fc_features = data_cfg['num_regions'] * (data_cfg['num_regions'] - 1) // 2
            print(f"\nTop 20 Most Important Features Overall:")
            for i, (feat_name, imp) in enumerate(avg_importances_series.head(20).items()):
                 if feat_name.startswith("FC_"):
                      try:
                           fc_index = int(feat_name.split("_")[1])
                           region_pair = get_region_pair_from_fc_index(fc_index, data_cfg['num_regions'])
                           feature_type = f"FC between regions {region_pair}" if region_pair else f"FC Feature {fc_index}"
                      except: feature_type = feat_name
                 else: feature_type = f"Phenotype: {feat_name}"
                 print(f" {i+1}. {feature_type} - Importance: {imp:.4f}")
            print("\n(Note: Mapping FC indices back to region pairs provides interpretability)")
        except Exception as fe_e: print(f"Could not process feature importances: {fe_e}")
    elif rfe_cfg.get('use_rfe', False): print("\nFeature importances not averaged when RFE is enabled.")
    else: print("\nNo feature importances were recorded.")

    end_time = time.time()
    print(f"\nTotal elapsed time: {end_time - start_time:.2f} seconds.")
    print("--- Baseline XGBoost Run Finished ---")

if __name__ == "__main__":
    main()