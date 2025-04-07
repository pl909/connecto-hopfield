# Attentive Connectome-based Hopfield Network (ACHNN) for Pain State Classification

## Project Overview

This project implements an Attentive Connectome-based Hopfield Network (ACHNN) designed to classify distinct brain states related to pain perception using fMRI data. The model leverages modern Hopfield networks combined with attention mechanisms to identify and learn patterns in whole-brain functional connectivity during different pain conditions.

The primary goals are:
1. **Classification of dynamic pain states** using regional brain activity patterns
2. **Analysis of attention mechanisms** to understand how the model identifies relevant brain regions
3. **Exploration of latent space representations** to gain insights into neural network reconfigurations during pain

The model is applied to the ds000140 dataset, which includes fMRI recordings of participants experiencing controlled heat pain stimuli at varying intensities under different cognitive modulation conditions (passive experience, regulate UP, regulate DOWN).

## Dataset Context

The project uses the **ds000140 dataset** from OpenNeuro, which contains:
- Task-based fMRI with controlled heat pain stimuli at varying intensities
- Experimental conditions with cognitive pain modulation:
  - **Passive experience**: Normal experience of pain stimuli
  - **Regulate UP**: Cognitive amplification of pain perception
  - **Regulate DOWN**: Cognitive reduction of pain perception

Data is preprocessed using the **RPN-Signature pipeline**, which produces:
- Regional timeseries for 122 brain regions (MIST122 atlas)
- Motion parameters and quality control metrics
- Subject-level framewise displacement measures

## Project Structure

```
project_root/
│
├── configs/
│   └── achnn_config.yaml         # Configuration for model, training, and analysis
│
├── src/
│   ├── hflayers/                 # Hopfield layer implementation
│   ├── data_loader.py            # Data loading and processing utilities
│   ├── models.py                 # ACHNN model implementation
│   ├── training.py               # Training and evaluation functions
│   └── utils.py                  # Helper functions
│
├── scripts/
│   ├── run_qc_filter.py          # Quality control filtering script
│   ├── run_achnn_training.py     # Main training script
│   ├── run_achnn_analysis.py     # Analysis script for trained models
│   └── test_data_access.py       # Script to test data accessibility
│
├── results/                      # Results directory (created during execution)
│   └── experiment_*/             # Results from each experiment
│
└── README.md                     # This file
```

## Data Directory Structure

The expected data structure from the RPN-Signature preprocessing pipeline is:

```
/usr/project/xtmp/ds000140-proc/
│
├── regional-timeseries/          # Regional timeseries in tab-separated format
│   └── sub-XX_task-pain_run-XX_timeseries.tsv
│
├── func_preproc/                 # Functional derivatives
│   ├── popFD_max.txt             # Max FD values per subject
│   ├── pop_percent_scrubbed.txt  # Percent of volumes scrubbed per subject
│   │
│   └── mc_fd/                    # Framewise displacement timeseries
│       └── sub-XX_task-pain_run-XX_FD.txt
│
├── QC/                           # Quality check images and visualizations
│   ├── FD/                       # Framewise displacement plots
│   ├── regional_timeseries/      # Carpet plots of atlas-based timeseries
│   └── ...                       # Other QC outputs
│
└── sub-XX/                       # Subject directories with BIDS format
    └── func/                     # Functional data
        └── sub-XX_task-pain_run-XX_events.tsv  # Event files
```

## The ACHNN Model

The ACHNN model architecture integrates temporal dynamics and associative memory through:

1. **Linear Embedding Layer**: Projects 122 brain regions to hidden dimension
2. **Positional Encoding**: Adds temporal position information to the sequence
3. **Transformer Encoder Blocks**: Process dynamic temporal information using self-attention
4. **Modern Hopfield Layer**: Implements an associative memory based on continuous modern Hopfield networks
5. **Classification Head**: Maps retrieved patterns to pain condition classes

### Key Components:

- **Self-Attention Mechanism**: Captures temporal dependencies among brain regions over time
- **Hopfield Core**: Functions as an associative memory that learns prototype patterns of brain states
- **Stored Patterns**: Learnable patterns that represent distinct brain states or configurations
- **Hopfield Attention**: Attention weights over stored patterns reveal what patterns are most relevant for each condition

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA-capable GPU (recommended for training)
- Access to the preprocessed ds000140 dataset

### Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-directory]
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Update configuration:
   Edit `configs/achnn_config.yaml` to set your data paths and model parameters.

## Usage

### Testing Data Access

First, verify that all required data files are accessible:

```bash
python scripts/test_data_access.py configs/achnn_config.yaml
```

This script checks for existence and readability of critical data files including:
- Regional timeseries files
- Framewise displacement files
- QC summary files
- Event files

### Quality Control Filtering

Filter subjects based on motion metrics from RPN-Signature outputs:

```bash
python scripts/run_qc_filter.py configs/achnn_config.yaml
```

This script:
- Loads the maximum FD and percent scrubbed values for each subject
- Applies thresholds to exclude high-motion subjects
- Saves a list of included subjects to `results/included_subjects.txt`

### Training the ACHNN Model

Train the model with cross-validation:

```bash
python scripts/run_achnn_training.py configs/achnn_config.yaml
```

This script will:
- Load data for subjects that passed QC
- Create sliding windows from the timeseries data
- Perform cross-validation, training the ACHNN model on each fold
- Save model checkpoints, metrics, and visualizations
- Generate aggregated results across all folds

### Analyzing a Trained Model

Analyze the attention patterns and latent space of a trained model:

```bash
python scripts/run_achnn_analysis.py --experiment_dir results/experiment_name
```

This script will:
- Load a trained model
- Generate attention heatmaps showing which stored patterns are activated for each condition
- Visualize the latent space using t-SNE and PCA
- Compare attention patterns between pain intensities and modulation conditions

## Understanding the Results

### Classification Performance

The model classifies different pain states based on patterns of brain activity. Key metrics include:
- Accuracy and F1-score for each condition
- Confusion matrix showing classification patterns

### Attention Analysis

The Hopfield attention analysis reveals:
- Which stored patterns are activated for each pain condition
- How attention patterns differ between pain intensities (high vs. medium vs. low)
- How pain modulation (UP vs. DOWN vs. passive) affects pattern activation

### Latent Space Analysis

The latent space visualizations show:
- Clustering of similar brain states
- Transitions between states
- Separation between different experimental conditions

## Customization

### Modifying Event Labels

To adapt the model for different experimental conditions:
1. Update the `conditions_to_classify` in `config.yaml`
2. Modify the `_map_condition_label` method in `data_loader.py` to properly extract labels from your events.tsv files

### Adjusting the Model Architecture

To modify the model architecture:
1. Adjust hyperparameters in `achnn_config.yaml`:
   - Change the number of Hopfield stored patterns
   - Modify transformer layers and attention heads
   - Tune dropout rates and learning parameters

### Running on Different Data

To apply the model to a different dataset:
1. Ensure your data is preprocessed into regional timeseries format
2. Update the paths in `achnn_config.yaml`
3. Adjust the data loading functions in `data_loader.py` to match your file structure

## Troubleshooting

### Common Issues

1. **ImportError for HopfieldCore**: 
   - Ensure the `hflayers` directory is in your Python path
   - Check that PyTorch version is compatible (1.9+ recommended)

2. **CUDA Out of Memory**:
   - Reduce batch size in `achnn_config.yaml`
   - Decrease model dimensions (hidden_dim, hopfield_pattern_dim)

3. **Data Access Issues**:
   - Run `test_data_access.py` to verify all required files are accessible
   - Check that file paths in `achnn_config.yaml` match the actual data structure
   - Ensure file naming patterns match between the config and actual files

4. **Poor Classification Performance**:
   - Ensure labels are correctly aligned with fMRI data considering HRF delay
   - Verify motion scrubbing parameters and QC thresholds
   - Adjust window size (seq_len) to better capture temporal dynamics

## Acknowledgments

- The Modern Hopfield Network implementation is based on "Hopfield Networks is All You Need" (Ramsauer et al., 2020)
- The ds000140 dataset: OpenNeuro dataset (https://openneuro.org/datasets/ds000140)
- RPN-Signature preprocessing pipeline was used for brain parcellation and denoising

## Citations

```
# Add relevant citations here, including:
# - Original ds000140 dataset paper
# - Modern Hopfield Networks paper
# - RPN-Signature pipeline paper
# - Your own work when published
```