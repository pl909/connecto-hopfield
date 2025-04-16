import os
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Ignore convergence warnings sometimes from SVM/RFE

def main():
    start_time = time.time()
    # --- Load Configuration ---
    print("Loading configuration from config_baseline.yaml...")
    try:
        with open('config_baseline.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config_baseline.yaml: {e}")
        return

    data_cfg = config['data_params']
    run_cfg = config['run_params']
    svm_cfg = config['svm_params']
    rfe_cfg = config['feature_selection_params']

    output_dir = data_cfg['output_feature_dir']
    feature_path = os.path.join(output_dir, 'features.npy')
    label_path = os.path.join(output_dir, 'labels.npy')

    # --- Load Preprocessed Data ---
    print("Loading preprocessed features and labels...")
    try:
        X = np.load(feature_path)
        y = np.load(label_path)
    except FileNotFoundError:
        print(f"Error: Preprocessed data not found at {feature_path} or {label_path}.")
        print("Please run preprocess_baseline.py first.")
        return
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return

    print(f"Loaded features shape: {X.shape}")
    print(f"Loaded labels shape: {y.shape}")
    if X.shape[0] != y.shape[0]:
        print("Error: Mismatch between number of samples in features and labels.")
        return

    # --- Cross-Validation Setup ---
    skf = StratifiedKFold(n_splits=run_cfg['n_splits'], shuffle=True, random_state=run_cfg['seed'])
    fold_results = []

    # Define scorer for RFE if used (use accuracy for faster selection)
    # scorer = make_scorer(roc_auc_score, needs_proba=True) # Use AUC if tuning RFE significantly
    scorer = make_scorer(accuracy_score) # Faster for basic RFE

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        print(f"\n--- Fold {fold+1}/{run_cfg['n_splits']} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Pipeline Steps ---
        steps = []

        # 1. Scaling (Essential for SVM)
        scaler = StandardScaler()
        steps.append(('scaler', scaler)) # Add scaler to pipeline

        # 2. Feature Selection (RFE) - Optional
        if rfe_cfg['use_rfe']:
            print(f"Applying RFE to select {rfe_cfg['n_features_to_select']} features...")
            # Use a simple linear SVM for faster RFE estimation
            estimator_rfe = SVC(kernel="linear", C=0.1, random_state=run_cfg['seed'])
            # RFECV could also be used here for automatic feature number selection, but slower
            selector = RFE(estimator=estimator_rfe,
                           n_features_to_select=rfe_cfg['n_features_to_select'],
                           step=0.1) # Remove 10% of features per iteration
            steps.append(('rfe', selector))
        else:
            print("Skipping RFE.")


        # 3. SVM Classifier
        print("Training SVM Classifier...")
        svc_classifier = SVC(
            kernel=svm_cfg['kernel'],
            C=svm_cfg['C'],
            gamma=svm_cfg['gamma'],
            probability=True, # Needed for AUC calculation
            random_state=run_cfg['seed']
        )
        steps.append(('svm', svc_classifier))

        # Create and Fit Pipeline
        pipeline = Pipeline(steps=steps)

        try:
             pipeline.fit(X_train, y_train)
        except ValueError as ve:
             print(f" Error fitting pipeline in Fold {fold+1}: {ve}. Maybe too few samples or features?")
             # Record failure for this fold
             fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0})
             continue # Skip to next fold


        # --- Evaluation ---
        print("Evaluating on test set...")
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] # Probability of class 1 for AUC

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        auc = 0.0
        if len(np.unique(y_test)) > 1: # Check if more than one class exists
            try:
                auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                print("Warning: Could not calculate AUC (likely only one class in test set).")
                auc = 0.0 # Or handle as NaN
        else:
             print("Warning: Only one class present in test set labels. AUC is ill-defined (setting to 0).")


        fold_metrics = {'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'auc': auc}
        fold_results.append(fold_metrics)

        print(f"Fold {fold+1} Results: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
        fold_end_time = time.time()
        print(f"Fold {fold+1} elapsed time: {fold_end_time - fold_start_time:.2f} seconds.")

    # --- Aggregate and Print Results ---
    print("\n--- Cross-Validation Results ---")
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
        else:
             print("No folds completed successfully with valid metrics.")
    else:
        print("No results recorded across folds.")

    end_time = time.time()
    print(f"\nTotal elapsed time: {end_time - start_time:.2f} seconds.")
    print("--- Baseline SVM Run Finished ---")

if __name__ == "__main__":
    main()