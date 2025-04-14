# Attentive Connectome-based Hopfield Network (ACHNN) for ASD Classification

## Project Overview

This project implements an Attentive Connectome-based Hopfield Network (ACHNN) designed to classify individuals with Autism Spectrum Disorder (ASD) from Typically Developing Controls (TDC) using resting-state fMRI data. The model leverages modern Hopfield networks combined with attention mechanisms to identify and learn dynamic patterns in whole-brain functional connectivity that differentiate these groups.

The primary goals are:
1.  **Classification of ASD vs. TDC** based on dynamic resting-state brain activity patterns.
2.  **Analysis of attention mechanisms** to understand how the model identifies relevant brain regions and temporal dynamics for classification.
3.  **Exploration of latent space representations** to gain insights into potentially altered network organization in ASD.

The model is applied to the **ABIDE I & II dataset**, using preprocessed timeseries data.

## Dataset Context

The project uses the **ABIDE (Autism Brain Imaging Data Exchange) I & II dataset**, accessed via the ABIDE Preprocessed initiative. Key features:
- Resting-state fMRI data from hundreds of participants.
- Diagnostic labels: ASD (Group 1) vs. Typically Developing Control (TDC, Group 2).
- Preprocessed derivatives available, including ROI timeseries.

This project utilizes timeseries extracted using the **Craddock 200 (CC200) functional atlas** from a specific pipeline (e.g., CCS) and strategy (e.g., filt\_noglobal) provided by the ABIDE Preprocessed resource.

## Project Structure

project_root/
│
├── configs/
│ └── achnn_config_abide.yaml # Configuration for ABIDE experiment
│
├── src/
│ ├── hflayers/ # Hopfield layer implementation
│ ├── data_loader.py # Data loading (ABIDE specific)
│ ├── models.py # ACHNN model implementation
│ ├── training.py # Training and evaluation functions
│ └── utils.py # Helper functions
│
├── scripts/
│ ├── run_achnn_training.py # Main training script for ABIDE
│ ├── run_achnn_analysis.py # Analysis script for ABIDE results
│ └── test_data_access.py # Script to test ABIDE data access
│
├── results/ # Results directory (created during execution)
│ └── abide_achnn_classification_*/ # Results from each experiment run
│
├── Phenotypic_V1_0b_preprocessed1.csv # ABIDE Phenotypic data file
└── README.md # This file

## Data Directory Structure (Expected Inputs)

-   **Phenotypic Data:** `/home/pl217/connecto-hopfield/Phenotypic_V1_0b_preprocessed1.csv` (as specified in config)
-   **Timeseries Data:** `/home/pl217/connecto-hopfield/results/abide_timeseries/ccs_filt_noglobal_rois_cc200/` (as specified in config) containing `.1D` files named like `Pitt_0050003_rois_cc200.1D`.

## The ACHNN Model

The ACHNN model architecture integrates temporal dynamics and associative memory:

1.  **Linear Embedding Layer**: Projects 200 brain regions (CC200 atlas) to hidden dimension.
2.  **Positional Encoding**: Adds temporal position information.
3.  **Transformer Encoder Blocks**: Process dynamics using self-attention.
4.  **Modern Hopfield Layer**: Associative memory based on continuous modern Hopfield networks.
5.  **Classification Head**: Maps patterns to binary class logits (ASD vs. TDC).

### Key Components:

-   **Self-Attention Mechanism**: Captures temporal dependencies within resting-state windows.
-   **Hopfield Core**: Learns prototype patterns (states) potentially distinguishing groups.
-   **Hopfield Attention**: Weights over stored patterns reveal which learned states are most relevant for classifying ASD vs. TDC.

## Setup Instructions

### Prerequisites

-   Python 3.8+
-   PyTorch 1.9+
-   CUDA-capable GPU (recommended)
-   Downloaded ABIDE CC200 timeseries data (see `download_abide.sh` script - requires verification)
-   ABIDE Phenotypic CSV file in the project root.

### Installation

1.  Clone the repository.
2.  Create and activate a virtual environment.
3.  Install dependencies: `pip install -r requirements.txt`
4.  **Crucially: Update `configs/achnn_config_abide.yaml`** to set correct paths (`base_dir`, `phenotypic_file`, `regional_timeseries_dir`), ensure `num_regions: 200`, `num_classes: 2`, and adjust other parameters as needed.

## Usage

### Testing Data Access

Verify paths and file access:

```bash
python scripts/test_data_access.py configs/achnn_config_abide.yaml

```


### Training the ACHNN Model
Train the model using cross-validation for ASD vs. TDC classification:

```bash
python scripts/run_achnn_training.py configs/achnn_config_abide.yaml
```

This script will:
Load phenotypic data and filter subjects based on included sites.
Load corresponding CC200 timeseries data.
Create sliding windows.
Perform Group K-Fold cross-validation.
Save model checkpoints, metrics, and logs to a new timestamped directory in results/.


### Analyzing a Trained Model
```bash

Analyze the results from a completed training run:

# Replace 'experiment_name_timestamp' with the actual directory created during training
python scripts/run_achnn_analysis.py --experiment_dir results/abide_achnn_classification_timestamp

```


This script will:
Load the best model from the cross-validation folds.
Generate Hopfield attention heatmaps comparing ASD vs. TDC groups.
Visualize the latent space using t-SNE and PCA, colored by group.
Calculate final performance metrics on the validation data used during analysis.


### Understanding the Results Classification Performance

How well does the model distinguish ASD from TDC based on resting-state dynamics? (Accuracy, F1-Score, Confusion Matrix).

### Attention Analysis

Which learned Hopfield patterns are more strongly activated for ASD vs. TDC?
Does the temporal self-attention focus differ between groups?
### Latent Space Analysis

Do ASD and TDC form distinct clusters in the learned representation space?
Does the structure reveal insights into heterogeneity or subtypes?
### Customization

Atlas/Regions: To use a different atlas (e.g., CC400), download the corresponding .1D files, update regional_timeseries_dir and num_regions in the config.
Model Hyperparameters: Adjust dimensions, layers, dropout, learning rate etc., in the config file.
Site Inclusion: Modify the included_sites list in the config.
### Troubleshooting
Import Errors: Ensure src and hflayers are accessible. Check requirements.txt.
Data Loading Errors: Verify paths in config, CSV format, and .1D file format/delimiter. Run test_data_access.py.
CUDA Errors: Ensure PyTorch CUDA version matches driver. Reduce batch_size if out of memory.
Low Performance: Check data quality/filtering, consider different window lengths (seq_len), tune hyperparameters.


### Acknowledgments
Modern Hopfield Network implementation inspired by Ramsauer et al. (2020).
ABIDE Dataset Initiative.
ABIDE Preprocessed data resource.

