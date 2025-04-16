import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def train_epoch(ts_encoder, pheno_encoder, gnn_classifier, fold_data, optimizer, criterion, train_mask, device, epoch):
    """Performs one training epoch with noise augmentation."""
    ts_encoder.train()
    if pheno_encoder is not None: pheno_encoder.train()
    gnn_classifier.train()
    optimizer.zero_grad()

    x_ts_dev = fold_data.x_ts.to(device)
    x_pheno_dev = fold_data.x_pheno.to(device)
    ts_mask_dev = fold_data.ts_mask.to(device)
    edge_index_dev = fold_data.edge_index.to(device)
    y_dev = fold_data.y.to(device)

    try:
        # --- Add Noise Augmentation ---
        if ts_encoder.training: # Check model mode
            noise_level = 0.1 # Hyperparameter: adjust this value (e.g., 0.005, 0.02)
            noise = torch.randn_like(x_ts_dev) * noise_level
            x_ts_augmented = x_ts_dev + noise
        else:
            x_ts_augmented = x_ts_dev # Should not happen if called correctly, but safe fallback
        # ------------------------------

        # Use augmented TS data for encoding
        ts_embed = ts_encoder(x_ts_augmented, ts_mask_dev)

        if pheno_encoder is not None and x_pheno_dev.nelement() > 0:
            pheno_embed = pheno_encoder(x_pheno_dev)
            if torch.isnan(pheno_embed).any() or torch.isinf(pheno_embed).any():
                # print("Warning: NaNs/Infs detected in Phenotype embeddings during training.")
                pheno_embed = torch.nan_to_num(pheno_embed)
        else:
             # Ensure pheno_embed has correct shape even if empty
             output_dim_pheno = 0
             if pheno_encoder is not None: # Try to get output dim if encoder exists
                  try: # Handle Identity case
                      output_dim_pheno = pheno_encoder.net[-1].out_features
                  except:
                       output_dim_pheno = config['model_params']['pheno_embedding_dim'] # Fallback
             pheno_embed = torch.empty((ts_embed.shape[0], output_dim_pheno), device=device)


        if torch.isnan(ts_embed).any() or torch.isinf(ts_embed).any():
            # print("Warning: NaNs/Infs detected in TS embeddings during training.")
            ts_embed = torch.nan_to_num(ts_embed)

        # Ensure batch dims match before cat
        if ts_embed.shape[0] != pheno_embed.shape[0]:
             raise RuntimeError(f"Batch dimension mismatch before concatenation: TS {ts_embed.shape[0]}, Pheno {pheno_embed.shape[0]}")

        combined_features = torch.cat([ts_embed, pheno_embed], dim=1)

    except Exception as e:
        print(f"Error during encoding in train_epoch: {e}")
        raise e # Re-raise to stop training if encoding fails

    out_logits = gnn_classifier(combined_features, edge_index_dev)
    loss = criterion(out_logits[train_mask], y_dev[train_mask])

    if torch.isnan(loss) or torch.isinf(loss):
         print(f"Error: Loss is NaN or Inf at epoch {epoch}. Skipping backward pass.")
         return np.inf # Return indicator of failure

    loss.backward()

    # --- Gradient Norm Check (Optional but useful) ---
    if epoch % 5 == 0 or epoch == 1:
        total_norm = 0
        model_params_for_grad = list(ts_encoder.parameters()) + list(gnn_classifier.parameters())
        if pheno_encoder is not None: model_params_for_grad.extend(list(pheno_encoder.parameters()))

        for p in model_params_for_grad:
            if p.grad is not None:
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except Exception as e_grad:
                    print(f"Warning: Error calculating grad norm for a parameter: {e_grad}")
        total_norm = total_norm ** 0.5
        print(f"  Epoch {epoch}, Grad Norm: {total_norm:.4f}")
    # -------------------------------------------------

    # Optional: Gradient clipping
    # torch.nn.utils.clip_grad_norm_(model_params_for_grad, max_norm=1.0)

    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(ts_encoder, pheno_encoder, gnn_classifier, fold_data, eval_mask, device, criterion=None):
    """Evaluates the model on a given data mask."""
    ts_encoder.eval()
    if pheno_encoder is not None: pheno_encoder.eval()
    gnn_classifier.eval()

    # Ensure data components are on the correct device ONLY when needed
    x_ts_dev = fold_data.x_ts.to(device)
    x_pheno_dev = fold_data.x_pheno.to(device)
    ts_mask_dev = fold_data.ts_mask.to(device)
    # edge_index is likely already on device from graph building
    edge_index_dev = fold_data.edge_index # Assuming already on device
    y_dev = fold_data.y.to(device) # Targets needed on device for loss

    try:
        ts_embed = ts_encoder(x_ts_dev, ts_mask_dev)
        if pheno_encoder is not None and x_pheno_dev.nelement() > 0:
            pheno_embed = pheno_encoder(x_pheno_dev)
            if torch.isnan(pheno_embed).any() or torch.isinf(pheno_embed).any():
                pheno_embed = torch.nan_to_num(pheno_embed)
        else:
             output_dim_pheno = 0
             if pheno_encoder is not None:
                  try: output_dim_pheno = pheno_encoder.net[-1].out_features
                  except: output_dim_pheno = config['model_params']['pheno_embedding_dim'] # Needs config access or hardcode
             pheno_embed = torch.empty((ts_embed.shape[0], output_dim_pheno), device=device)


        if torch.isnan(ts_embed).any() or torch.isinf(ts_embed).any():
            ts_embed = torch.nan_to_num(ts_embed)

        # Ensure dimensions match
        if ts_embed.shape[0] != pheno_embed.shape[0]:
             raise RuntimeError(f"Eval Batch dimension mismatch: TS {ts_embed.shape[0]}, Pheno {pheno_embed.shape[0]}")

        combined_features = torch.cat([ts_embed, pheno_embed], dim=1)

    except Exception as e:
        print(f"Error during encoding in evaluate: {e}")
        # Return zeros or NaNs to indicate failure
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


    out_logits = gnn_classifier(combined_features, edge_index_dev)
    probs = F.softmax(out_logits, dim=-1)
    preds = probs.argmax(dim=-1)

    # --- Evaluate only on the specified mask ---
    eval_mask_dev = eval_mask.to(device=device, dtype=torch.bool)
    labels_eval = y_dev[eval_mask_dev]
    preds_eval = preds[eval_mask_dev]
    probs_eval = probs[eval_mask_dev]

    loss = None
    # Calculate loss only if criterion is provided and there are samples
    if criterion is not None and eval_mask_dev.sum() > 0:
         try:
             loss = criterion(out_logits[eval_mask_dev], labels_eval).item()
             if torch.isnan(torch.tensor(loss)): loss = None # Reset if NaN
         except Exception as e:
             # print(f"Error calculating loss during evaluation: {e}") # Reduce verbosity
             loss = None

    # Move results to CPU for sklearn metrics
    preds_eval_cpu = preds_eval.cpu().numpy()
    labels_eval_cpu = labels_eval.cpu().numpy()
    probs_eval_cpu = probs_eval.cpu().numpy() # Probabilities for AUC

    num_eval_samples = len(labels_eval_cpu)
    if num_eval_samples == 0:
        # print("Warning: Evaluation mask resulted in 0 samples.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, loss if loss is not None else 0.0

    # Calculate metrics
    acc = accuracy_score(labels_eval_cpu, preds_eval_cpu)
    # Use weighted average for potentially imbalanced test folds
    f1 = f1_score(labels_eval_cpu, preds_eval_cpu, average='weighted', zero_division=0)
    precision = precision_score(labels_eval_cpu, preds_eval_cpu, average='weighted', zero_division=0)
    recall = recall_score(labels_eval_cpu, preds_eval_cpu, average='weighted', zero_division=0)

    # Calculate AUC - needs probabilities of the positive class (class 1)
    auc = 0.0
    # Ensure binary classification and valid probabilities
    if probs_eval_cpu.shape[1] == 2 and np.all(np.isfinite(probs_eval_cpu[:, 1])):
        try:
            # Check if there's more than one class present in labels
            if len(np.unique(labels_eval_cpu)) > 1:
                 auc = roc_auc_score(labels_eval_cpu, probs_eval_cpu[:, 1])
            else:
                 # print("AUC calculation skipped: Only one class present in evaluation labels.")
                 auc = 0.0 # Undefined, return 0 or np.nan
        except ValueError as e:
            # print(f"AUC calculation failed: {e}. Probs shape: {probs_eval_cpu.shape}, Labels unique: {np.unique(labels_eval_cpu)}")
            auc = 0.0
    else:
        # print("Warning: Cannot calculate AUC. Probs shape not binary or contains non-finite values.")
        auc = 0.0


    return acc, f1, precision, recall, auc, loss if loss is not None else 0.0 # Return loss as float or 0.0