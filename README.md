# AIedes

**AIedes** is a data-driven framework for estimating mosquito presence and abundance using neural networks. 

---

## 📦 Installation

We recommend creating a dedicated `conda` environment to ensure compatibility with dependencies.

### 1. Clone the repository

```bash
git clone https://github.com/f7/AIedes.git
cd AIedes
```

### 2. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate aiedes
```

### 2.1. Install PyTorch

Before installing the package, ensure that PyTorch is installed in your environment. You can install PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/). For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Replace the URL with the appropriate one for your system and hardware (e.g., GPU support).

### 2.2. Install the package

You have two options for installing the package:

#### Option 1: Standard installation

For a standard installation, use the following command:

```bash
pip install .
```

This will install the package in a non-editable mode.

#### Option 2: Developer installation

If you are a developer and want to install the package in editable mode, use the following command:

```bash
pip install -e .
```

### 2.3. Download Required Datasets

Two datasets are required to run the AIedes experiments:

#### Required Datasets:
1. **ECDC Albopictus 2023 Cordex 2020** (`.pkl` format)
   - **Purpose**: Classifier experiments (presence/absence prediction)
   - **Location**: Place in `experiments/classifier/data/`

2. **AIMSurv Albopictus 2020 ERA5-Land** (`.pkl` format)
   - **Purpose**: Counter experiments (abundance estimation)
   - **Location**: Place in `experiments/counter/data/`

#### Data Source:
These datasets are officially published and available from:

**European Commission, Joint Research Centre (JRC) (2026)**: *Harmonised climate and Aedes albopictus arboviral vector mosquito surveillance datasets for the European continent*. European Commission, Joint Research Centre (JRC) [Dataset] doi: [10.2905/e4e68952-a02f-460d-8f85-eb630742741d](https://doi.org/10.2905/e4e68952-a02f-460d-8f85-eb630742741d)

#### Dataset Structure:
```
experiments/
├── classifier/data/
│   └── ECDC_Albopictus_2023_Cordex_2020.pkl  # For presence/absence experiments
└── counter/data/
    └── AIMSurv_Albopictus_2020_ERA5_Land.pkl  # For abundance experiments
```

**Note**: The datasets contain harmonized climate and surveillance data for Aedes albopictus across European regions, integrating ECDC surveillance records with CORDEX/ERA5-Land climate reanalysis data.

### 3. Running Experiments

AIedes provides two main experiment types: **Classifier** (presence/absence prediction) and **Counter** (abundance estimation). Each has multiple execution options from quick tests to comprehensive grid searches.

#### 🧪 **Quick Start - Testing Your Setup**

Before running full experiments, test your installation:

```bash
# Test classifier pipeline
cd experiments/classifier
python run_exp.py --num_epochs 10 --input_variables temp_tp

# Test counter pipeline
cd ../counter
./test.sh  # Smoke test with 1 epoch
```

#### 🎯 **Classifier Experiments** (Presence/Absence Prediction)

**Basic Usage:**
```bash
cd experiments/classifier

# Single experiment with default parameters
python run_exp.py --input_variables temp_tp --num_epochs 500

# Custom architecture and features
python run_exp.py \
    --input_variables temp \
    --layers "[64, 32]" \
    --num_epochs 1000 \
    --lr 0.001 \
    --batch_size 512 \
    --use_nuts3_split True
```

**Grid Search Options:**
```bash
# Temperature-only experiments across architectures
./run_exp_temp.sh

# Temperature + precipitation experiments
./run_exp_temp_tp.sh
```

**Key Parameters:**
- `--input_variables`: `"temp"`, `"tp"`, or `"temp_tp"` (climate features)
- `--layers`: Network architecture, e.g., `"[20,20]"`, `"[64,32,16]"`
- `--use_nuts3_split`: `True` for geographic cross-validation, `False` for random splits
- `--num_epochs`: Training epochs (default: 500)
- `--lr`: Learning rate (default: 0.002)
- `--batch_size`: Batch size (default: 1024)

#### 📊 **Counter Experiments** (Abundance Estimation)

**Basic Usage:**
```bash
cd experiments/counter

# Single experiment with default parameters
python run.py --model_type positive_ff --num_epochs 1000

# Advanced model with custom loss function
python run.py \
    --model_type hurdle \
    --loss_type FrequencyWeightedZINBLoss \
    --hidden_dim 64 \
    --num_layers 3 \
    --dropout 0.2 \
    --num_epochs 3000 \
    --variables "t2m_mean_znorm d2m_mean_znorm tp_sum_znorm p1zn p2zn"
```

**Grid Search Experiments:**
```bash
# Main comprehensive grid search (recommended)
./run_all.sh
```

**Key Parameters:**
- `--model_type`: `"positive_ff"`, `"positive_lstm"`, `"positive_mha"`, `"hurdle"`
- `--loss_type`: `"ZINBLoss"`, `"FrequencyWeightedMSLE"`, `"FrequencyWeightedMSE"`, etc.
- `--hidden_dim`: Hidden layer size (default: 30)
- `--num_layers`: Number of layers (default: 3)
- `--variables`: Climate features to use (supports statistical aggregations)
- `--dropout`: Dropout rate (default: 0.01)
- `--device`: `"cpu"`, `"cuda"`, or `"mps"` (auto-detected)

**Note:** The `./run_all.sh` script runs comprehensive experiments with multiple loss types (`ZINBLoss`, `FrequencyWeightedMSLE`, `FrequencyWeightedMSE`), dropout rates, layer configurations, and variable combinations automatically.

#### 🔧 **Advanced Configuration**

**Variable Usage:**
```bash
# Standard normalized climate variables + previous rates
--variables "t2m_mean_znorm d2m_mean_znorm tp_sum_znorm p1zn p2zn"

# With statistical aggregations (weekly means)
--variables "t2m_mean_znorm:mean d2m_mean_znorm:mean tp_sum_znorm:mean p1zn p2zn"

# Minimal feature set (only previous rates)
--variables "p1zn p2zn"
```

**Variable Types:**
- `*_znorm`: Z-score normalized climate features (created at runtime)
- `p1zn`: Previous period 1 rates (normalized, with presence flag)
- `p2zn`: Previous period 2 rates (normalized, with presence flag)
- `:mean`: Apply weekly mean aggregation to the variable

**Model Architectures:**
- `positive_ff`: Feed-forward with positive constraints
- `positive_lstm`: LSTM for temporal patterns
- `positive_mha`: Multi-head attention mechanism
- `hurdle`: Zero-inflated hurdle model

**Loss Functions for Count Data:**
- `MSE`: Standard mean squared error
- `FrequencyWeightedZINBLoss`: Weighted zero-inflated negative binomial
- `Hurdle`: Two-stage hurdle model
- `WeightedMSE`: Class-balanced MSE

#### 📁 **Expected Outputs**

**Classifier Results:**
```
experiments/classifier/results/
├── *.pkl                    # Model results and metrics
└── estimated_risk/          # Risk estimation outputs
```

**Counter Results:**
```
experiments/counter/
├── test_results/            # Test outputs
├── *.log                    # Training logs
└── results_*/               # Experiment results by configuration
```

**Result Files Include:**
- `.pth`: Trained model weights (PyTorch)
- `.pkl`: Complete results with metrics, predictions, and metadata
- `.png`: Training loss plots and visualizations
- `.log`: Detailed training logs

#### 🚀 **Performance Tips**

- **GPU Acceleration**: Models automatically use CUDA if available
- **Batch Sizes**: Start with defaults, increase if you have sufficient memory
- **Early Stopping**: Models use validation-based early stopping (patience=50)
- **Cross-Validation**: Use `--use_nuts3_split True` for geographic validation
- **Grid Search**: Use provided shell scripts for hyperparameter optimization

#### ⚠️ **Common Issues**

- Ensure conda environment is activated: `conda activate aiedes`
- For memory issues, reduce `--batch_size`
- Check dataset placement in correct directories
- Use `./test.sh` for initial pipeline verification