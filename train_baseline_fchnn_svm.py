import os
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# RFE likely not needed for low-dimensional fcHNN features
# from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    start_time = time.time()
    # --- Load Configuration ---
    config_filename = 'config_baseline_fchnn.yaml'
    print(f"Loading configuration from {config_filename}...")
    try:
        with open(config_filename, 'r') as f: config = yaml.safe_load(f)
    except Exception as e: print(f"Error loading {config_filename}: {e}"); return

    data_cfg = config['data_params']
    run_cfg = config['run_params']
    svm_cfg = config['svm_params']
    # rfe_cfg = config['feature_selection_params'] # RFE config likely not needed

    # --- Load fcHNN Features ---
    feature_dir = data_cfg['output_feature_dir'] # Directory where preprocess_fchnn saved
    feature_path = os.path.join(feature_dir, 'features.npy')
    label_path = os.path.join(feature_dir, 'labels.npy')

    print("Loading preprocessed fcHNN+Pheno features and labels...")
    try:
        X = np.load(feature_path)
        y = np.load(label_path).astype(int)
        if X.shape[0] != y.shape[0]: raise ValueError("Feature/Label sample mismatch")
        print(f" Feature shape: {X.shape}, Label shape: {y.shape}")
        print(f" Label distribution: {np.bincount(y)}")
    except Exception as e:
        print(f"Error loading preprocessed data: {e}. Run preprocess_baseline_fchnn.py first."); return

    # --- Cross-Validation Setup ---
    n_splits = run_cfg['n_splits']
    if X.shape[0] < n_splits: n_splits = max(2, X.shape[0])
    if n_splits < 2: print("Error: Not enough samples."); return
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run_cfg['seed'])
    fold_results = []

    # --- Optional: Define parameter grid for GridSearchCV ---
    use_grid_search = False # Set to True to enable hyperparameter tuning
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Pipeline: Scaling + SVM ---
        steps = []
        scaler = StandardScaler()
        steps.append(('scaler', scaler))

        svc_classifier = SVC(
            kernel=svm_cfg.get('kernel', 'rbf'),
            probability=True, # Needed for AUC
            random_state=run_cfg['seed'],
            # C and gamma will be set by GridSearchCV if enabled, otherwise use config
            C=svm_cfg.get('C', 1.0) if not use_grid_search else 1.0,
            gamma=svm_cfg.get('gamma', 'scale') if not use_grid_search else 'scale'
        )
        steps.append(('svm', svc_classifier))

        pipeline = Pipeline(steps=steps)

        # --- Fit Model (with optional GridSearchCV) ---
        if use_grid_search:
            print("Running GridSearchCV for SVM hyperparameters...")
            # Use AUC scorer for grid search if possible
            scorer_auc = make_scorer(roc_auc_score, needs_proba=True, error_score=0.0)
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring=scorer_auc, n_jobs=-1, verbose=1) # 3-fold inner CV
            try:
                grid_search.fit(X_train, y_train)
                print(f" Best Params: {grid_search.best_params_}")
                pipeline = grid_search.best_estimator_ # Use best found pipeline
            except Exception as e:
                 print(f"GridSearchCV failed: {e}. Training with default params.")
                 # Fallback to fitting with default params
                 try: pipeline.fit(X_train, y_train)
                 except Exception as fit_e:
                      print(f"Error fitting pipeline in Fold {fold+1}: {fit_e}")
                      fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0})
                      continue
        else:
            # Fit directly without grid search
            print("Training SVM Classifier with parameters from config...")
            try: pipeline.fit(X_train, y_train)
            except Exception as fit_e:
                 print(f"Error fitting pipeline in Fold {fold+1}: {fit_e}")
                 fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0})
                 continue

        # --- Evaluation ---
        print("Evaluating on test set...")
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

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
        print(f"Number of successfully completed folds: {num_valid_folds}/{n_splits}")
        if num_valid_folds > 0:
            avg_acc = np.mean([res.get('acc', 0) for res in valid_results]); std_acc = np.std([res.get('acc', 0) for res in valid_results])
            avg_f1 = np.mean([res.get('f1', 0) for res in valid_results]); avg_prec = np.mean([res.get('precision', 0) for res in valid_results])
            avg_rec = np.mean([res.get('recall', 0) for res in valid_results]); avg_auc = np.mean([res.get('auc', 0) for res in valid_results])
            print(f'Average Accuracy:  {avg_acc:.4f} +/- {std_acc:.4f}'); print(f'Average F1-Score:  {avg_f1:.4f}')
            print(f'Average Precision: {avg_prec:.4f}'); print(f'Average Recall:    {avg_rec:.4f}'); print(f'Average AUC:       {avg_auc:.4f}')
        else: print("No folds completed successfully.")
    else: print("No results recorded.")

    end_time = time.time()
    print(f"\nTotal elapsed time: {end_time - start_time:.2f} seconds.")
    print("--- Baseline fcHNN+SVM Run Finished ---")

if __name__ == "__main__":
    main()