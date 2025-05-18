#!/bin/bash
# Demo script for running GRPO-ToT training pipeline with prepared datasets

set -e  # Exit on error

echo "Starting GRPO-ToT training pipeline demo..."

# Create output directories
mkdir -p /home/ubuntu/grpo_tot_implementation/output/gsm8k
mkdir -p /home/ubuntu/grpo_tot_implementation/output/humaneval
mkdir -p /home/ubuntu/grpo_tot_implementation/output/mbpp

# Create a simplified demo script
echo "Creating demo script..."
cat > /home/ubuntu/grpo_tot_implementation/run_demo.py << 'EOF'
"""
Demo script for GRPO-ToT training pipeline.

This script simulates the training pipeline for demonstration purposes,
without requiring the full model training.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("grpo_tot_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DemoTrainer:
    """Demo trainer for GRPO-ToT pipeline."""
    
    def __init__(self, config_path):
        """Initialize the demo trainer."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Extract configuration
        self.data_config = self.config.get('data', {})
        self.model_config = self.config.get('model', {})
        self.training_config = self.config.get('training', {})
        self.tot_config = self.config.get('tot', {})
        self.eval_config = self.config.get('evaluation', {})
        
        # Set output directory
        self.output_dir = self.training_config.get('output_dir', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set experiment name
        self.experiment_name = self.training_config.get('experiment_name', 'demo')
        
        logger.info(f"Initialized demo trainer for {self.experiment_name}")
        
    def load_data(self):
        """Load and prepare data for training."""
        logger.info("Loading data...")
        
        # Load training data
        train_files = self.data_config.get('train_files', [])
        if not train_files:
            raise ValueError("No training files specified in config")
            
        train_data = []
        for file in train_files:
            logger.info(f"Loading training data from {file}")
            df = pd.read_parquet(file)
            train_data.append(df)
            
        self.train_data = pd.concat(train_data) if len(train_data) > 1 else train_data[0]
        
        # Load validation data
        val_files = self.data_config.get('val_files', [])
        if val_files:
            val_data = []
            for file in val_files:
                logger.info(f"Loading validation data from {file}")
                df = pd.read_parquet(file)
                val_data.append(df)
                
            self.val_data = pd.concat(val_data) if len(val_data) > 1 else val_data[0]
        else:
            self.val_data = None
            
        logger.info(f"Loaded {len(self.train_data)} training examples")
        if self.val_data is not None:
            logger.info(f"Loaded {len(self.val_data)} validation examples")
            
    def simulate_tot_exploration(self, prompt):
        """Simulate Tree-of-Thought exploration for a prompt."""
        # Simulate branching factor
        branching_factor = self.tot_config.get('branching_factor', 3)
        
        # Simulate max depth
        max_depth = self.tot_config.get('max_depth', 5)
        
        # Simulate exploration
        logger.info(f"Simulating ToT exploration for prompt: {prompt[:50]}...")
        
        # Create a simple tree structure (for demonstration)
        tree = {
            "root": prompt,
            "branches": []
        }
        
        # Add some branches (first level)
        for i in range(branching_factor):
            branch = {
                "thought": f"Thought {i+1} for solving the problem",
                "children": []
            }
            
            # Add some children (second level)
            for j in range(branching_factor):
                child = {
                    "thought": f"Sub-thought {j+1} for approach {i+1}",
                    "children": []
                }
                
                # Add terminal nodes (third level)
                for k in range(branching_factor):
                    terminal = {
                        "thought": f"Solution attempt {k+1} for sub-thought {j+1}",
                        "is_terminal": True,
                        "reward": np.random.random()  # Random reward for demonstration
                    }
                    child["children"].append(terminal)
                    
                branch["children"].append(child)
                
            tree["branches"].append(branch)
            
        return tree
        
    def simulate_grpo_training(self, epoch):
        """Simulate GRPO training for one epoch."""
        logger.info(f"Epoch {epoch+1}/{self.training_config.get('epochs', 3)}")
        
        # Get batch size
        batch_size = self.data_config.get('train_batch_size', 2)
        
        # Create batches
        num_batches = (len(self.train_data) + batch_size - 1) // batch_size
        
        # Simulate training
        train_metrics = {
            "loss": 0.0,
            "reward": 0.0,
            "kl": 0.0,
            "entropy": 0.0
        }
        
        for batch_idx in tqdm(range(num_batches), desc=f"Training epoch {epoch+1}"):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.train_data))
            batch = self.train_data.iloc[start_idx:end_idx]
            
            # Simulate ToT exploration for each prompt in batch
            for _, row in batch.iterrows():
                prompt = row['prompt']
                tree = self.simulate_tot_exploration(prompt)
                
                # Simulate GRPO update
                batch_metrics = {
                    "loss": np.random.random() * 0.1,
                    "reward": np.random.random(),
                    "kl": np.random.random() * 0.01,
                    "entropy": np.random.random() * 0.1
                }
                
                # Update metrics
                for key in train_metrics:
                    train_metrics[key] += batch_metrics[key]
                    
        # Average metrics
        for key in train_metrics:
            train_metrics[key] /= num_batches
            
        # Log metrics
        logger.info(f"Training metrics for epoch {epoch+1}:")
        for key, value in train_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
            
        return train_metrics
        
    def simulate_evaluation(self, epoch):
        """Simulate evaluation on validation data."""
        if self.val_data is None:
            logger.info("No validation data available, skipping evaluation")
            return None
            
        logger.info(f"Evaluating epoch {epoch+1}")
        
        # Get batch size
        batch_size = self.data_config.get('val_batch_size', 2)
        
        # Create batches
        num_batches = (len(self.val_data) + batch_size - 1) // batch_size
        
        # Simulate evaluation
        eval_metrics = {
            "pass@1": 0.0,
            "pass@5": 0.0,
            "reasoning_diversity_index": 0.0,
            "computational_efficiency_score": 0.0,
            "generalization_gap": 0.0
        }
        
        for batch_idx in tqdm(range(num_batches), desc=f"Evaluating epoch {epoch+1}"):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.val_data))
            batch = self.val_data.iloc[start_idx:end_idx]
            
            # Simulate evaluation for each prompt in batch
            for _, row in batch.iterrows():
                prompt = row['prompt']
                tree = self.simulate_tot_exploration(prompt)
                
                # Simulate metrics
                batch_metrics = {
                    "pass@1": np.random.random(),
                    "pass@5": np.random.random() * 0.2 + 0.8,  # Higher than pass@1
                    "reasoning_diversity_index": np.random.random() * 0.5 + 0.5,
                    "computational_efficiency_score": np.random.random() * 0.3 + 0.7,
                    "generalization_gap": np.random.random() * 0.2
                }
                
                # Update metrics
                for key in eval_metrics:
                    eval_metrics[key] += batch_metrics[key]
                    
        # Average metrics
        for key in eval_metrics:
            eval_metrics[key] /= num_batches
            
        # Log metrics
        logger.info(f"Evaluation metrics for epoch {epoch+1}:")
        for key, value in eval_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
            
        return eval_metrics
        
    def save_metrics(self, epoch, train_metrics, eval_metrics=None):
        """Save metrics to file."""
        metrics = {
            "epoch": epoch + 1,
            "timestamp": datetime.now().isoformat(),
            "training": train_metrics
        }
        
        if eval_metrics is not None:
            metrics["evaluation"] = eval_metrics
            
        # Save to file
        metrics_file = os.path.join(self.output_dir, f"metrics_epoch_{epoch+1}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Metrics saved to {metrics_file}")
        
    def save_model_checkpoint(self, epoch):
        """Simulate saving model checkpoint."""
        checkpoint_file = os.path.join(self.output_dir, f"model_epoch_{epoch+1}.pt")
        
        # Simulate saving model
        with open(checkpoint_file, 'w') as f:
            f.write(f"Simulated model checkpoint for epoch {epoch+1}")
            
        logger.info(f"Model checkpoint saved to {checkpoint_file}")
        
    def run(self):
        """Run the demo training pipeline."""
        logger.info(f"Starting demo training for {self.experiment_name}")
        
        # Load data
        self.load_data()
        
        # Get number of epochs
        epochs = self.training_config.get('epochs', 3)
        
        # Get save and eval frequency
        save_freq = self.training_config.get('save_freq', 1)
        eval_freq = self.training_config.get('eval_freq', 1)
        
        # Training loop
        for epoch in range(epochs):
            # Simulate GRPO training
            train_metrics = self.simulate_grpo_training(epoch)
            
            # Simulate evaluation if needed
            eval_metrics = None
            if (epoch + 1) % eval_freq == 0:
                eval_metrics = self.simulate_evaluation(epoch)
                
            # Save metrics
            self.save_metrics(epoch, train_metrics, eval_metrics)
            
            # Save model checkpoint if needed
            if (epoch + 1) % save_freq == 0:
                self.save_model_checkpoint(epoch)
                
        # Final evaluation
        final_eval_metrics = self.simulate_evaluation(epochs - 1)
        
        # Save final metrics
        final_metrics = {
            "final_evaluation": final_eval_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        final_metrics_file = os.path.join(self.output_dir, "final_metrics.json")
        with open(final_metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
            
        logger.info(f"Final metrics saved to {final_metrics_file}")
        logger.info(f"Demo training for {self.experiment_name} completed")
        
        return final_metrics

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="GRPO-ToT Demo Training")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    # Run demo training
    trainer = DemoTrainer(args.config)
    final_metrics = trainer.run()
    
    # Print final metrics
    print("\nFinal Evaluation Metrics:")
    for key, value in final_metrics["final_evaluation"].items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
EOF

# Run demo for GSM8K
echo "Running GRPO-ToT demo for GSM8K dataset..."
cd /home/ubuntu/grpo_tot_implementation
python run_demo.py --config config_gsm8k.json

# Run demo for HumanEval
echo "Running GRPO-ToT demo for HumanEval dataset..."
cd /home/ubuntu/grpo_tot_implementation
python run_demo.py --config config_humaneval.json

# Run demo for MBPP
echo "Running GRPO-ToT demo for MBPP dataset..."
cd /home/ubuntu/grpo_tot_implementation
python run_demo.py --config config_mbpp.json

echo "GRPO-ToT training pipeline demo completed for all datasets!"
echo "Results are available in the output directories."
