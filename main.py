import yaml
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
import time
import math # For ceil
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import scheduler

# Import project modules
from src.utils import set_seed, get_device
from src.data_loader import load_data, load_and_pad_timeseries, get_fold_max_len
from src.models import TimeSeriesTransformerEncoder, PhenotypeEncoder, GNNClassifier
from src.graph_utils import build_graph
from src.engine import train_epoch, evaluate


# --- Main Execution Function ---
def main():
    start_time = time.time()
    # --- Load Configuration ---
    print("Loading configuration from config.yaml...")
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please create it in the root directory.")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing config.yaml: {e}")
        return

    # --- Print Configuration ---
    print("\n--- Configuration Used ---")
    print(yaml.dump(config, default_flow_style=False, indent=2))
    print("--------------------------\n")

    # --- Setup ---
    run_cfg = config['run_params']
    data_cfg = config['data_params']
    graph_cfg = config['graph_params']
    model_cfg = config['model_params']

    try:
        MAX_LEN_CONFIG = int(data_cfg['max_ts_length'])
        ENCODER_BATCH_SIZE = int(run_cfg['encoder_batch_size'])
        NUM_REGIONS = int(data_cfg['num_regions'])
        if MAX_LEN_CONFIG <= 0 and MAX_LEN_CONFIG != -1 : raise ValueError("max_ts_length must be positive or -1")
        if ENCODER_BATCH_SIZE <= 0 or NUM_REGIONS <= 0: raise ValueError("Batch size, regions must be positive")
        print(f"Using configured max_ts_length setting: {MAX_LEN_CONFIG} (-1 means use max in fold)")
        print(f"Using configured num_regions: {NUM_REGIONS}")
        print(f"Using configured encoder_batch_size: {ENCODER_BATCH_SIZE}")
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error: Invalid or missing config parameter ({e}). Check 'max_ts_length', 'num_regions', and 'encoder_batch_size'.")
        return

    set_seed(run_cfg['seed'])
    DEVICE = get_device(run_cfg.get('device', 'cuda'))

    # --- 1. Load Initial Data ---
    subject_data_dict, num_pheno_features, all_targets = load_data(config)

    if subject_data_dict is None:
        print("Exiting due to data loading errors.")
        return

    subject_ids_list = list(subject_data_dict.keys())
    num_subjects = len(subject_ids_list)

    if num_subjects < run_cfg['n_splits']:
         print(f"Warning: Number of subjects ({num_subjects}) with TS files is less than n_splits ({run_cfg['n_splits']}). Adjusting n_splits.")
         run_cfg['n_splits'] = max(2, num_subjects)

    if num_subjects < 2:
        print(f"Error: Fewer than 2 subjects ({num_subjects}) found with valid data. Cannot perform cross-validation.")
        return

    print(f"Total subjects available for K-Fold analysis: {num_subjects}")
    if num_pheno_features <=0 and model_cfg['pheno_embedding_dim'] > 0:
         print("Warning: No phenotype features were loaded, but pheno_embedding_dim > 0. PhenotypeEncoder may fail or be unused.")

    # --- 2. Cross-Validation Setup ---
    skf = StratifiedKFold(n_splits=run_cfg['n_splits'], shuffle=True, random_state=run_cfg['seed'])
    fold_results = []
    failed_folds = 0

    for fold, (train_idx, test_idx) in enumerate(skf.split(subject_ids_list, all_targets)): # Use list of IDs that HAVE data
        fold_start_time = time.time()
        print(f"\n--- Fold {fold+1}/{run_cfg['n_splits']} ---")

        train_subject_ids_fold = [subject_ids_list[i] for i in train_idx]
        test_subject_ids_fold = [subject_ids_list[i] for i in test_idx]
        fold_subject_ids_initial = train_subject_ids_fold + test_subject_ids_fold

        # --- Determine Max Length for THIS fold ---
        if MAX_LEN_CONFIG == -1:
            current_fold_max_len_for_pe = get_fold_max_len(subject_data_dict, fold_subject_ids_initial, NUM_REGIONS)
            if current_fold_max_len_for_pe is None or current_fold_max_len_for_pe <= 0:
                 print(f"Critical Error: Could not determine valid max TS length for Positional Encoding in Fold {fold+1}. Skipping fold.")
                 failed_folds += 1; fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}); continue
            max_len_for_loader = -1
            print(f"Fold {fold+1} using dynamic max length (up to {current_fold_max_len_for_pe} for PE).")
        else:
            current_fold_max_len_for_pe = MAX_LEN_CONFIG
            max_len_for_loader = MAX_LEN_CONFIG
            print(f"Fold {fold+1} truncating/padding to fixed length: {max_len_for_loader}")

        # --- 3. Load, Normalize, (Optionally) Truncate, and Pad Time Series ---
        fold_padded_ts, fold_ts_masks, failed_load_ids, actual_padding_len = load_and_pad_timeseries(
            subject_data_dict, fold_subject_ids_initial, max_len=max_len_for_loader, num_regions=NUM_REGIONS
        )

        if fold_padded_ts is None:
            print(f"Critical Error: Could not load/pad time series for Fold {fold+1}. Skipping fold.")
            failed_folds += 1; fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}); continue

        fold_subject_ids_loaded = [sid for sid in fold_subject_ids_initial if sid not in failed_load_ids]
        successful_indices_in_batch = [i for i, sid in enumerate(fold_subject_ids_initial) if sid not in failed_load_ids]

        if not fold_subject_ids_loaded:
             print(f"Critical Error: No subjects loaded successfully in Fold {fold+1}. Skipping fold.")
             failed_folds += 1; fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}); continue

        fold_padded_ts = fold_padded_ts[:, successful_indices_in_batch, :]
        fold_ts_masks = fold_ts_masks[successful_indices_in_batch, :]
        num_fold_nodes = len(fold_subject_ids_loaded)
        print(f"Proceeding with {num_fold_nodes} subjects for Fold {fold+1} model training/evaluation.")

        # --- 4. Prepare Phenotype, Targets, and Masks for the *successfully loaded* subjects ---
        try:
             pheno_array_filtered = np.array([subject_data_dict[sid]['pheno'] for sid in fold_subject_ids_loaded])
             if pheno_array_filtered.ndim == 1 and num_pheno_features > 0: pheno_array_filtered = pheno_array_filtered.reshape(1, -1)
             elif pheno_array_filtered.size == 0 and num_pheno_features > 0: fold_pheno_features = torch.zeros((num_fold_nodes, num_pheno_features), dtype=torch.float32)
             elif num_pheno_features > 0 and pheno_array_filtered.shape[1] != num_pheno_features: fold_pheno_features = torch.zeros((num_fold_nodes, num_pheno_features), dtype=torch.float32)
             elif num_pheno_features == 0: fold_pheno_features = torch.empty((num_fold_nodes, 0), dtype=torch.float32)
             else: fold_pheno_features = torch.tensor(pheno_array_filtered, dtype=torch.float32)
             fold_targets = torch.tensor(np.array([subject_data_dict[sid]['target'] for sid in fold_subject_ids_loaded]), dtype=torch.long)
        except Exception as e_pheno:
             print(f"Unexpected error preparing phenotype/targets for fold {fold+1} after filtering: {e_pheno}. Skipping fold.")
             failed_folds += 1; fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}); continue

        train_mask = torch.zeros(num_fold_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_fold_nodes, dtype=torch.bool)
        original_train_set = set(train_subject_ids_fold)
        original_test_set = set(test_subject_ids_fold)
        for i, sub_id in enumerate(fold_subject_ids_loaded):
            if sub_id in original_train_set: train_mask[i] = True
            elif sub_id in original_test_set: test_mask[i] = True

        if train_mask.sum() == 0 or test_mask.sum() == 0:
             print(f"Warning: Fold {fold+1} has zero samples in train ({train_mask.sum()}) or test ({test_mask.sum()}) set after filtering. Skipping fold.")
             failed_folds += 1; fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}); continue

        # --- 5. Initialize Models ---
        # Determine actual max length for PE based on mode
        pe_max_len_this_fold = actual_padding_len if max_len_for_loader == -1 else max_len_for_loader
        ts_encoder = TimeSeriesTransformerEncoder(config, num_regions=NUM_REGIONS, max_len=pe_max_len_this_fold).to(DEVICE)

        pheno_encoder = None
        pheno_embedding_dim = model_cfg['pheno_embedding_dim'] if num_pheno_features > 0 else 0
        if num_pheno_features > 0 and pheno_embedding_dim > 0:
            pheno_encoder = PhenotypeEncoder(config, input_dim=num_pheno_features).to(DEVICE)
            print("Phenotype encoder created.")
        else:
             print("Phenotype encoder skipped.")

        combined_feature_dim = model_cfg['ts_embedding_dim'] + pheno_embedding_dim
        gnn_classifier = GNNClassifier(config, node_feature_dim=combined_feature_dim).to(DEVICE)

        # --- Optimizer, Criterion, Scheduler ---
        model_params = list(ts_encoder.parameters()) + list(gnn_classifier.parameters())
        if pheno_encoder is not None: model_params.extend(list(pheno_encoder.parameters()))
        optimizer = optim.AdamW(model_params, lr=run_cfg['learning_rate'], weight_decay=run_cfg['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        # Use scheduler params from config
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', verbose=True,
            factor=run_cfg.get('lr_scheduler_factor', 0.1),
            patience=run_cfg.get('lr_scheduler_patience', 10),
            min_lr=1e-7 # Keep a minimum LR
        )

        # --- 6. Generate Combined Features (Mini-Batching) ---
        print(f"Generating initial embeddings for graph construction (Batch Size: {ENCODER_BATCH_SIZE})...")
        ts_encoder.eval()
        if pheno_encoder: pheno_encoder.eval()
        all_ts_embeddings = []
        all_pheno_embeddings = []
        num_batches = math.ceil(num_fold_nodes / ENCODER_BATCH_SIZE)

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Encoding Batches", leave=False):
                start_idx = i * ENCODER_BATCH_SIZE
                end_idx = min((i + 1) * ENCODER_BATCH_SIZE, num_fold_nodes)

                ts_batch = fold_padded_ts[:, start_idx:end_idx, :].to(DEVICE)
                mask_batch = fold_ts_masks[start_idx:end_idx, :].to(DEVICE)
                pheno_batch = fold_pheno_features[start_idx:end_idx, :].to(DEVICE)

                batch_ts_embed = ts_encoder(ts_batch, mask_batch)
                all_ts_embeddings.append(batch_ts_embed.cpu())

                if pheno_encoder:
                    if pheno_batch.nelement() > 0:
                         batch_pheno_embed = pheno_encoder(pheno_batch)
                         all_pheno_embeddings.append(batch_pheno_embed.cpu())
                    else:
                         batch_size_actual = end_idx - start_idx
                         all_pheno_embeddings.append(torch.zeros((batch_size_actual, pheno_embedding_dim), dtype=torch.float32))

                if DEVICE == torch.device('cuda'): torch.cuda.empty_cache()

        init_ts_embed = torch.cat(all_ts_embeddings, dim=0)
        if pheno_encoder and all_pheno_embeddings:
            init_pheno_embed = torch.cat(all_pheno_embeddings, dim=0)
            if init_ts_embed.shape[0] == init_pheno_embed.shape[0]:
                 init_combined_features = torch.cat([init_ts_embed, init_pheno_embed], dim=1)
            else:
                 print(f"Error: Embedding dimension mismatch! Skipping fold.")
                 failed_folds += 1; fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}); continue
        else: init_combined_features = init_ts_embed
        print("Initial embeddings generated.")

        # --- 7. Build Graph ---
        edge_index = build_graph(init_combined_features.cpu(), k=graph_cfg['k_neighbors'], device=DEVICE)

        # --- Create PyG Data object ---
        fold_data = Data(
            x_ts=fold_padded_ts, x_pheno=fold_pheno_features, ts_mask=fold_ts_masks,
            y=fold_targets, edge_index=edge_index
        )

        # --- 8. Training Loop ---
        print(f"Starting training for Fold {fold+1}... Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}")
        best_test_auc = 0.0
        fold_best_metrics = {}
        patience_counter = 0
        early_stopping_patience = run_cfg.get('early_stopping_patience', 20) # Get from config

        for epoch in range(1, run_cfg['epochs'] + 1):
            try:
                current_pheno_encoder = pheno_encoder if pheno_encoder is not None else None
                loss = train_epoch(ts_encoder, current_pheno_encoder, gnn_classifier, fold_data, optimizer, criterion, train_mask, DEVICE, epoch) # Pass epoch
            except Exception as train_e:
                print(f"Error during training epoch {epoch} for fold {fold+1}: {train_e}")
                failed_folds +=1; fold_best_metrics = {}; break

            if torch.isnan(torch.tensor(loss)):
                print(f"Error: Loss became NaN during epoch {epoch} for fold {fold+1}. Stopping fold.")
                failed_folds += 1; fold_best_metrics = {}; break

            # Evaluate less frequently later in training? e.g., epoch % 10 == 0
            eval_freq = 5
            if epoch <= 50 or epoch % eval_freq == 0 or epoch == run_cfg['epochs']:
                try:
                    current_pheno_encoder = pheno_encoder if pheno_encoder is not None else None
                    train_acc, train_f1, _, _, train_auc, train_loss = evaluate(ts_encoder, current_pheno_encoder, gnn_classifier, fold_data, train_mask, DEVICE, criterion)
                    test_acc, test_f1, test_prec, test_rec, test_auc, test_loss = evaluate(ts_encoder, current_pheno_encoder, gnn_classifier, fold_data, test_mask, DEVICE, criterion)

                    print(f'F{fold+1} E{epoch:03d} | Loss: {loss:.4f} | Tr Acc: {train_acc:.4f} | Te Acc: {test_acc:.4f} | Te AUC: {test_auc:.4f} | Te F1: {test_f1:.4f}')

                    # Step scheduler based on Test AUC
                    current_lr = optimizer.param_groups[0]['lr'] # Get current LR for info
                    scheduler.step(test_auc)
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr < current_lr:
                         print(f"  Learning rate reduced to {new_lr:.1e}")


                    if test_auc > best_test_auc:
                        best_test_auc = test_auc
                        fold_best_metrics = {'acc': test_acc, 'f1': test_f1, 'precision': test_prec, 'recall': test_rec, 'auc': test_auc}
                        patience_counter = 0 # Reset patience on improvement
                        # Optional: Save best model checkpoint here
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                         print(f"Early stopping triggered at epoch {epoch} due to lack of improvement in Test AUC for {early_stopping_patience} evaluation steps.")
                         break # Exit epoch loop

                except Exception as eval_e:
                    print(f"Error during evaluation at epoch {epoch} for fold {fold+1}: {eval_e}")
                    failed_folds += 1; fold_best_metrics = {}; break

        # --- End of Epoch Loop ---
        if not fold_best_metrics:
             print(f"Fold {fold+1} failed or did not complete training.")
             fold_results.append({'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0})
        else:
            print(f"Fold {fold+1} - Training finished. Best Test Accuracy: {fold_best_metrics.get('acc', 0):.4f}, AUC: {fold_best_metrics.get('auc', 0):.4f}")
            fold_results.append(fold_best_metrics)

        fold_end_time = time.time()
        print(f"Fold {fold+1} elapsed time: {fold_end_time - fold_start_time:.2f} seconds.")


    # --- 9. Aggregate and Print Results ---
    print("\n--- Cross-Validation Results ---")
    if fold_results:
        valid_results = [res for res in fold_results if res and isinstance(res.get('acc'), (int, float)) and res.get('acc', 0) > -1]
        num_valid_folds = len(valid_results)
        print(f"Number of successfully completed folds with valid metrics: {num_valid_folds}/{run_cfg['n_splits']}")

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
    print("--- Run Finished ---")


if __name__ == '__main__':
    main()