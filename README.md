# GRPO-ToT: Hybrid Reinforcement Learning with Tree-of-Thought

This repository implements a hybrid approach combining Group Relative Policy Optimization (GRPO) with Tree-of-Thought (ToT) reasoning for enhanced Large Language Model (LLM) performance. The implementation leverages the VERL (Volcano Engine Reinforcement Learning for LLMs) library for GRPO capabilities and integrates a custom Tree-of-Thought module.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Running the Implementation](#running-the-implementation)
- [Evaluation](#evaluation)
- [Extending the Implementation](#extending-the-implementation)

## Overview

The GRPO-ToT hybrid implementation combines the computational efficiency of GRPO with the expanded reasoning capabilities of Tree-of-Thought. This approach enables:

1. **Memory-efficient training** by eliminating the need for a value network
2. **Enhanced reasoning** through systematic exploration of multiple reasoning paths
3. **Improved performance** on complex tasks requiring multi-step reasoning

The implementation is designed to work with various datasets, including mathematical reasoning (GSM8K), code generation (HumanEval), and other reasoning tasks.

## Key Features

- **Dual-Phase Reward Mechanism**: Balances exploration and exploitation through configurable reward components
- **Path Importance Weighting (PIW)**: Selectively distills the most valuable reasoning paths
- **Comprehensive Evaluation Metrics**: Includes pass@k, Reasoning Diversity Index, Computational Efficiency Score, and Generalization Gap
- **Modular Design**: Easily extensible to new datasets and tasks

## Project Structure

```
grpo_tot_implementation/
├── verl/                           # VERL library (cloned from GitHub)
├── datasets/                       # Dataset preparation and storage
│   ├── dataset_selection.md        # Documentation of selected datasets
│   ├── prepare_datasets.py         # Dataset preparation script
│   ├── gsm8k/                      # GSM8K dataset files
│   ├── humaneval/                  # HumanEval dataset files
│   └── mbpp/                       # MBPP dataset files
├── grpo_tot_integration.py         # Core integration of GRPO and ToT
├── training_pipeline.py            # End-to-end training pipeline
├── run_demo.py                     # Demo script for running the pipeline
├── run_demo.sh                     # Shell script for running demos
├── config_gsm8k.json               # Configuration for GSM8K dataset
├── config_humaneval.json           # Configuration for HumanEval dataset
├── config_mbpp.json                # Configuration for MBPP dataset
├── requirements.txt                # Python dependencies
├── verl_documentation.md           # Documentation of VERL library
├── tot_module_design.md            # Design document for ToT module
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Git

### Setup Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/grpo-tot-implementation.git
   cd grpo-tot-implementation
   ```

2. **Clone the VERL library**:

   ```bash
   git clone https://github.com/volcengine/verl.git
   ```

3. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Install VERL library**:
   ```bash
   cd verl
   pip install -e .
   cd ..
   ```

## Dataset Preparation

The implementation supports various datasets for training and evaluation. Use the provided script to prepare the datasets:

```bash
cd datasets
python prepare_datasets.py
```

This script will:

1. Download the datasets from Hugging Face (GSM8K, HumanEval, MBPP)
2. Preprocess them for use with the GRPO-ToT pipeline
3. Save them in the appropriate format (parquet files)
4. Generate a summary of the prepared datasets

If the datasets cannot be downloaded automatically, the script will create synthetic data for demonstration purposes.

## Configuration

The implementation uses JSON configuration files to specify training parameters. Three example configurations are provided:

- `config_gsm8k.json`: Configuration for mathematical reasoning (GSM8K)
- `config_humaneval.json`: Configuration for code generation (HumanEval)
- `config_mbpp.json`: Configuration for simple code generation (MBPP)

Key configuration sections include:

### Data Configuration

```json
"data": {
    "train_files": ["path/to/train.parquet"],
    "val_files": ["path/to/val.parquet"],
    "test_files": ["path/to/test.parquet"],
    "train_batch_size": 32,
    "val_batch_size": 32,
    "test_batch_size": 32,
    "max_prompt_length": 1024,
    "max_response_length": 1024
}
```

### Model Configuration

```json
"model": {
    "model_path": "Qwen/Qwen2-7B-Instruct",
    "model_type": "hf",
    "use_remove_padding": true,
    "enable_gradient_checkpointing": true
}
```

### Training Configuration

```json
"training": {
    "epochs": 15,
    "save_freq": 5,
    "eval_freq": 2,
    "output_dir": "output/gsm8k",
    "lr": 1e-6,
    "ppo_mini_batch_size": 256,
    "ppo_micro_batch_size_per_gpu": 16,
    "use_kl_loss": true,
    "kl_loss_coef": 0.001
}
```

### Tree-of-Thought Configuration

```json
"tot": {
    "branching_factor": 5,
    "max_depth": 12,
    "initial_phase": "exploration",
    "phase_transition_epoch": 8,
    "exploration_temp": 0.8,
    "backtracking_threshold": 0.3
}
```

## Running the Implementation

### Demo Run

For a quick demonstration of the implementation, use the provided shell script:

```bash
chmod +x run_demo.sh
./run_demo.sh
```

This will run the demo for all three datasets (GSM8K, HumanEval, MBPP) and save the results in the respective output directories.

### Full Training

To run the full training pipeline with a specific configuration:

```bash
python training_pipeline.py --config config_gsm8k.json
```

Replace `config_gsm8k.json` with the appropriate configuration file for your dataset.

### Custom Dataset

To use a custom dataset:

1. Prepare your dataset in parquet format with columns: `prompt`, `response`, and optional `metadata`
2. Update the configuration file to point to your dataset files
3. Run the training pipeline with your configuration

## Evaluation

The implementation includes comprehensive evaluation metrics:

- **pass@k**: Success rate when considering the top-k generations
- **Reasoning Diversity Index**: Measure of diversity in reasoning paths
- **Computational Efficiency Score**: Efficiency of computation relative to performance
- **Generalization Gap**: Performance difference between in-distribution and out-of-distribution examples

Evaluation results are saved in the output directory specified in the configuration file.

## Extending the Implementation

### Adding New Datasets

To add a new dataset:

1. Prepare the dataset in parquet format with the required columns
2. Create a new configuration file based on the existing examples
3. Update the dataset paths in the configuration file

### Modifying the Tree-of-Thought Module

The ToT module is designed to be modular and extensible. Key components that can be modified include:

- `ToTExplorer`: Controls the exploration strategy
- `ToTRewardShaper`: Implements the dual-phase reward mechanism
- `PathImportanceWeighter`: Handles the weighting of reasoning paths

### Integrating with Different Models

The implementation supports various models through the VERL library. To use a different model:

1. Update the `model_path` and `model_type` in the configuration file
2. Ensure the model is compatible with the VERL library
3. Adjust hyperparameters as needed for the specific model

## Citation

If you use this implementation in your research, please cite:

```
@misc{grpo-tot-2025,
  author = {DngBack},
  title = {GRPO-ToT: Hybrid Reinforcement Learning with Tree-of-Thought},
  year = {2025},
  publisher = {},
  journal = {},
  howpublished = {}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
