import os
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import time
from src.utils import set_seed, get_device

from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Define MLP Model ---
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        mlp_cfg = config['mlp_params']
        hidden_layers = mlp_cfg.get('hidden_layer_sizes', [128, 64])
        dropout_rate = mlp_cfg.get('dropout_rate', 0.5)
        use_batchnorm = mlp_cfg.get('use_batchnorm', True)
        activation_fn = nn.ReLU if mlp_cfg.get('activation', 'relu').lower() == 'relu' else nn.Tanh # Example

        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(p=dropout_rate))
            last_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(last_dim, output_dim))
        # No activation/softmax needed if using CrossEntropyLoss

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- Training and Evaluation Functions ---
def train_mlp_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
             print("Warning: NaN/Inf loss detected during training. Skipping batch.")
             continue # Skip this batch update

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate_mlp(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    batches = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)

        if criterion is not None:
            loss = criterion(outputs, labels)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                 total_loss += loss.item()
                 batches += 1

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs[:, 1].cpu()) # Probability of class 1

    avg_loss = total_loss / batches if batches > 0 else 0.0

    if not all_labels: # Handle empty loader case
        return 0.0, 0.0, 0.0, 0.0, 0.0, avg_loss

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    auc = 0.0
    if len(np.unique(all_labels)) > 1:
        try: auc = roc_auc_score(all_labels, all_probs)
        except ValueError: auc = 0.0
    else: auc = 0.0

    return acc, f1, precision, recall, auc, avg_loss


# --- Main Function ---
def main():
    start_time = time.time()
    # --- Load Configuration ---
    config_filename = 'config_baseline_mlp.yaml'
    print(f"Loading configuration from {config_filename}...")
    try:
        with open(config_filename, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {config_filename}: {e}")
        return

    data_cfg = config['data_params']
    run_cfg = config['run_params']
    mlp_cfg = config['mlp_params']

    output_dir = data_cfg['output_feature_dir']
    feature_path = os.path.join(output_dir, 'features.npy')
    label_path = os.path.join(output_dir, 'labels.npy')

    # --- Load Preprocessed Data ---
    print("Loading preprocessed features and labels...")
    try:
        X = np.load(feature_path)
        y = np.load(label_path).astype(int)
        print(f" Feature shape: {X.shape}, Label shape: {y.shape}")
        print(f" Label distribution: {np.bincount(y)}")
        if X.shape[0] != y.shape[0]: raise ValueError("Feature/Label sample mismatch")
    except Exception as e:
        print(f"Error loading preprocessed data: {e}. Run preprocess_baseline.py first.")
        return

    # --- Setup ---
    set_seed(run_cfg['seed'])
    DEVICE = get_device(run_cfg.get('device', 'cuda'))

    # --- Cross-Validation ---
    n_splits = run_cfg['n_splits']
    if X.shape[0] < n_splits: n_splits = max(2, X.shape[0])
    if n_splits < 2: print("Error: Not enough samples for CV."); return

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run_cfg['seed'])
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        print(f"\n--- Fold {fold+1}/{n_splits} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Scaling ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- Create PyTorch Datasets and DataLoaders ---
        train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=run_cfg['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=run_cfg['batch_size'] * 2) # Larger batch for eval

        # --- Initialize Model, Optimizer, Criterion, Scheduler ---
        input_dim = X_train_scaled.shape[1]
        output_dim = len(np.unique(y)) # Should be 2 for binary
        model = MLPClassifier(input_dim=input_dim, output_dim=output_dim, config=config).to(DEVICE)

        opt_name = mlp_cfg.get('optimizer', 'adamw').lower()
        if opt_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=run_cfg['learning_rate'], weight_decay=run_cfg['weight_decay'])
        elif opt_name == 'sgd':
             optimizer = optim.SGD(model.parameters(), lr=run_cfg['learning_rate'], weight_decay=run_cfg['weight_decay'], momentum=0.9)
        else: # Default to AdamW
            optimizer = optim.AdamW(model.parameters(), lr=run_cfg['learning_rate'], weight_decay=run_cfg['weight_decay'])

        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', verbose=True,
            factor=mlp_cfg.get('lr_scheduler_factor', 0.1),
            patience=mlp_cfg.get('lr_scheduler_patience', 10),
            min_lr=1e-7
        )

        # --- Training Loop ---
        print(f"Starting training for Fold {fold+1}...")
        best_test_auc = 0.0
        fold_best_metrics = {}
        patience_counter = 0
        early_stopping_patience = run_cfg.get('early_stopping_patience', 20)

        for epoch in range(1, run_cfg['epochs'] + 1):
            train_loss = train_mlp_epoch(model, train_loader, optimizer, criterion, DEVICE)

            # Evaluate periodically
            if epoch % 5 == 0 or epoch == 1 or epoch == run_cfg['epochs']:
                test_acc, test_f1, test_prec, test_rec, test_auc, test_loss = evaluate_mlp(model, test_loader, criterion, DEVICE)
                print(f'F{fold+1} E{epoch:03d} | Tr Loss: {train_loss:.4f} | Te Loss: {test_loss:.4f} | Te Acc: {test_acc:.4f} | Te AUC: {test_auc:.4f}')

                scheduler.step(test_auc) # Step based on validation AUC

                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    fold_best_metrics = {'acc': test_acc, 'f1': test_f1, 'precision': test_prec, 'recall': test_rec, 'auc': test_auc}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        # --- End of Epoch Loop ---
        if not fold_best_metrics:
            print(f"Fold {fold+1} failed or did not improve.")
            fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0})
        else:
            print(f"Fold {fold+1} - Training finished. Best Test Accuracy: {fold_best_metrics.get('acc', 0):.4f}, AUC: {fold_best_metrics.get('auc', 0):.4f}")
            fold_results.append(fold_best_metrics)

        fold_end_time = time.time()
        print(f"Fold {fold+1} elapsed time: {fold_end_time - fold_start_time:.2f} seconds.")

    # --- Aggregate and Print Results ---
    print("\n--- Cross-Validation Results ---")
    if fold_results:
        valid_results = [res for res in fold_results if res and res.get('acc', -1) > -1]
        num_valid_folds = len(valid_results)
        print(f"Number of successfully completed folds: {num_valid_folds}/{n_splits}")
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

    end_time = time.time()
    print(f"\nTotal elapsed time: {end_time - start_time:.2f} seconds.")
    print("--- Baseline MLP Run Finished ---")


if __name__ == "__main__":
    main()