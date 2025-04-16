import os
import numpy as np
import pandas as pd # Needed for feature names
import yaml
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Need OHE logic here if loading raw
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Helper function to get feature names after preprocessing (simplified)
# NOTE: This relies on the exact preprocessing steps in preprocess_baseline.py
# It's better to save feature names during preprocessing if possible.
def get_feature_names(config):
    data_cfg = config['data_params']
    num_cols = data_cfg['phenotype_cols_numerical']
    cat_cols = data_cfg['phenotype_cols_categorical']
    num_regions = data_cfg['num_regions']

    # FC feature names (generic)
    num_fc_features = num_regions * (num_regions - 1) // 2
    fc_names = [f"FC_{i}" for i in range(num_fc_features)]

    # Phenotype feature names (requires loading data briefly to get OHE names)
    pheno_names = []
    pheno_names.extend(num_cols)
    try:
        # Quick load to get categories for OHE names
        df_pheno_temp = pd.read_csv(data_cfg['phenotype_file'], usecols=cat_cols)
        df_pheno_temp = df_pheno_temp.replace([-9999, -9999.0, ''], np.nan)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_pheno_temp[cat_cols] = df_pheno_temp[cat_cols].astype(str) # Ensure string
        df_pheno_temp[cat_cols] = cat_imputer.fit_transform(df_pheno_temp[cat_cols])
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(df_pheno_temp[cat_cols])
        pheno_names.extend(ohe.get_feature_names_out(cat_cols))
    except Exception as e:
        print(f"Warning: Could not reliably get OHE feature names: {e}. Using generic names.")
        # Fallback: Generate generic names if OHE fails
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
    config_filename = 'config_baseline_logreg.yaml' # Use a specific config
    print(f"Loading configuration from {config_filename}...")
    try:
        with open(config_filename, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {config_filename}: {e}")
        return

    data_cfg = config['data_params']
    run_cfg = config['run_params']
    logreg_cfg = config['logreg_params'] # Get LogReg params

    output_dir = data_cfg['output_feature_dir']
    feature_path = os.path.join(output_dir, 'features.npy')
    label_path = os.path.join(output_dir, 'labels.npy')
    subjects_path = os.path.join(output_dir, 'subject_ids.npy') # Load subject IDs

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

    # --- Get Feature Names (Best Effort) ---
    print("Attempting to retrieve feature names...")
    feature_names = get_feature_names(config)
    if len(feature_names) != X.shape[1]:
         print(f"Warning: Number of generated feature names ({len(feature_names)}) does not match data columns ({X.shape[1]}). Importance mapping might be incorrect.")
         feature_names = [f"feature_{i}" for i in range(X.shape[1])] # Use generic names

    # --- Cross-Validation Setup ---
    skf = StratifiedKFold(n_splits=run_cfg['n_splits'], shuffle=True, random_state=run_cfg['seed'])
    fold_results = []
    all_coefs = [] # Store coefficients from each fold

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        print(f"\n--- Fold {fold+1}/{run_cfg['n_splits']} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Pipeline: Scaling + Logistic Regression ---
        # RFE is typically not used with L1 regularization as L1 does selection
        steps = []
        scaler = StandardScaler()
        steps.append(('scaler', scaler))

        print("Training Logistic Regression Classifier...")
        logreg_classifier = LogisticRegression(
            penalty='l1', # Use L1 for sparsity
            C=logreg_cfg['C'],
            solver='liblinear', # Good solver for L1
            random_state=run_cfg['seed'],
            max_iter=logreg_cfg.get('max_iter', 1000) # Increase max iterations if needed
        )
        steps.append(('logreg', logreg_classifier))

        pipeline = Pipeline(steps=steps)

        try:
            pipeline.fit(X_train, y_train)
        except Exception as e:
             print(f"Error fitting pipeline in Fold {fold+1}: {e}")
             fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0})
             continue

        # Store coefficients
        all_coefs.append(pipeline.named_steps['logreg'].coef_.flatten())

        # --- Evaluation ---
        print("Evaluating on test set...")
        y_pred = pipeline.predict(X_test)
        try:
            # predict_proba might fail for some solvers/penalties if classes are separated
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except AttributeError:
            print("Warning: predict_proba not available. Calculating AUC from decision function.")
            try:
                 y_decision = pipeline.decision_function(X_test)
                 # Need scaling or checking if roc_auc_score can handle decision values directly
                 # For now, calculate AUC based on prediction if proba fails
                 y_proba = y_pred # Fallback, AUC will be less meaningful
            except AttributeError:
                 y_proba = y_pred # Fallback

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
    if all_coefs:
        print("\n--- Average Feature Importances (Logistic Regression Coefficients) ---")
        avg_coefs = np.mean(all_coefs, axis=0)
        feature_importances = np.abs(avg_coefs) # Use absolute value for ranking importance
        feature_indices_sorted = np.argsort(feature_importances)[::-1] # Indices sorted by importance

        num_fc_features = data_cfg['num_regions'] * (data_cfg['num_regions'] - 1) // 2
        num_total_features = X.shape[1]

        print(f"Total features: {num_total_features}, FC features: {num_fc_features}, Pheno features: {num_total_features - num_fc_features}")

        # Separate FC and Phenotype
        top_n = 20 # Number of top features to show
        top_indices = feature_indices_sorted[:top_n]

        print(f"\nTop {top_n} Most Important Features (Absolute Coefficient):")
        for i, idx in enumerate(top_indices):
            importance = avg_coefs[idx] # Get signed coefficient
            if idx < num_fc_features:
                region_pair = get_region_pair_from_fc_index(idx, data_cfg['num_regions'])
                feature_type = f"FC between regions {region_pair}" if region_pair else f"FC Index {idx}"
            else:
                pheno_idx = idx - num_fc_features
                feature_type = f"Phenotype: {feature_names[idx]}" if idx < len(feature_names) else f"Phenotype Index {pheno_idx}"

            print(f" {i+1}. {feature_type} (Index: {idx}) - Coef: {importance:.4f}")

        # Count non-zero coefficients (due to L1)
        non_zero_coefs = np.sum(np.abs(avg_coefs) > 1e-6) # Use small tolerance
        print(f"\nNumber of features with non-zero average coefficient (L1 Selection): {non_zero_coefs} / {num_total_features}")

    end_time = time.time()
    print(f"\nTotal elapsed time: {end_time - start_time:.2f} seconds.")
    print("--- Baseline Logistic Regression Run Finished ---")

if __name__ == "__main__":
    main()