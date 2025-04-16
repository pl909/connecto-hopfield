# ASD Classification using Transformer, GNN, and Phenotypes

This project implements an advanced approach for Autism Spectrum Disorder (ASD) classification using resting-state fMRI time series data and phenotypic information, typically from datasets like ABIDE.

The core strategy aims for state-of-the-art performance by:
1.  Learning feature representations directly from fMRI time series using a **Transformer Encoder**.
2.  Encoding relevant phenotypic data using a Multi-Layer Perceptron (MLP).
3.  Combining the learned time series features and phenotype features for each subject.
4.  Constructing a population graph using k-Nearest Neighbors (k-NN) based on the similarity of the combined subject features.
5.  Applying a Graph Attention Network (**GATv2**) for classification on the population graph.
6.  Utilizing Stratified K-Fold cross-validation for robust model evaluation.

## Data Requirements

1.  **Phenotype Data:** A CSV file (specified in `config.yaml`, e.g., `Phenotypic_V1_0b_preprocessed1.csv`) containing:
    *   A unique subject ID column (`sub_id_col`).
    *   The subject's site ID column (`site_id_col`, needed to find TS files).
    *   The target diagnosis column (`target_col`, expected: 1 for ASD, 2 for Control).
    *   Relevant numerical phenotypic features (e.g., `AGE_AT_SCAN`, `FIQ` - specified in `phenotype_cols_numerical`).
    *   Relevant categorical phenotypic features (e.g., `SEX`, `SITE_ID` - specified in `phenotype_cols_categorical`).
    *   Missing values coded as -9999 are handled via imputation.
2.  **Time Series Data:** A directory (specified in `config.yaml` -> `data_params.region_dir`) containing **one file per subject**.
    *   **Filename Format:** The code expects filenames in the format: `{SITE_ID}_{SUB_ID_7_DIGIT}_rois_{ATLAS_INFO}.1D` (e.g., `Leuven_1_0050687_rois_cc200.1D`). The `SUB_ID` must be zero-padded to 7 digits. Modify `src/data_loader.py` if your format differs.
    *   **File Content:** Each `.1D` file should be a text file containing space-separated BOLD signal values, where each **row is a time step** and each **column is a region**. The number of columns must match `num_regions` (e.g., 200 for CC200) set in `config.yaml`. Files can have variable numbers of rows (time steps).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory_name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:** Ensure PyTorch/PyG compatibility.
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Carefully edit the `config.yaml` file:
*   **`run_params`**: Set seed, device, CV folds, epochs, LR, weight decay, `encoder_batch_size`, `early_stopping_patience`.
*   **`data_params`**: **CRITICAL:** Set paths, filenames, and ALL column names (`sub_id_col`, `site_id_col`, `target_col`, numerical, categorical) precisely. Set `max_ts_length` (truncation length) and `num_regions` (e.g., 200).
*   **`graph_params`**: Configure `k_neighbors`.
*   **`model_params`**: Adjust model dimensions and hyperparameters.

## Usage

1.  Ensure data is correctly placed and referenced in `config.yaml`.
2.  Run the main training script:
    ```bash
    python main.py
    ```
3.  Monitor output for progress, gradient norms, evaluation metrics, and LR scheduler adjustments.

## Notes
*   **Memory:** Reduce `encoder_batch_size`, `max_ts_length`, or model dimensions in `config.yaml` if `OutOfMemoryError` occurs.
*   **Performance:** Achieving SOTA likely requires significant hyperparameter tuning and potentially SSL pre-training.