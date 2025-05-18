#!/bin/bash
# Validation script for GRPO-ToT hybrid implementation

set -e  # Exit on error

echo "Starting validation of GRPO-ToT hybrid implementation..."

# Create necessary directories
mkdir -p /home/ubuntu/data/gsm8k
mkdir -p /home/ubuntu/data/math
mkdir -p /home/ubuntu/grpo_tot_implementation/output

# Create sample data for testing
echo "Creating sample data for validation..."
python3 - << 'EOF'
import pandas as pd
import numpy as np
import os

# Create sample GSM8K data
gsm8k_train = pd.DataFrame({
    'prompt': [
        "If John has 5 apples and gives 2 to Mary, how many apples does John have left?",
        "Sarah has $10. She buys a book for $3 and a pen for $2. How much money does she have left?",
        "A train travels at 60 miles per hour. How far will it travel in 2.5 hours?"
    ],
    'response': [
        "John has 5 apples initially. He gives 2 apples to Mary. So John has 5 - 2 = 3 apples left.",
        "Sarah has $10 initially. She spends $3 on a book and $2 on a pen. So she spends $3 + $2 = $5 in total. Therefore, she has $10 - $5 = $5 left.",
        "The train travels at 60 miles per hour. In 2.5 hours, it will travel 60 * 2.5 = 150 miles."
    ]
})

gsm8k_test = pd.DataFrame({
    'prompt': [
        "Tom has 8 marbles. He loses 3 marbles. How many marbles does Tom have now?",
        "A box contains 24 red balls and 36 blue balls. What fraction of the balls are red?"
    ],
    'response': [
        "Tom has 8 marbles initially. He loses 3 marbles. So Tom has 8 - 3 = 5 marbles now.",
        "The box contains 24 red balls and 36 blue balls. The total number of balls is 24 + 36 = 60. The fraction of red balls is 24/60 = 2/5."
    ]
})

# Create sample math data
math_train = pd.DataFrame({
    'prompt': [
        "Solve for x: 2x + 5 = 15",
        "Find the area of a circle with radius 4 cm.",
        "If f(x) = x^2 - 3x + 2, find f(2)."
    ],
    'response': [
        "We have 2x + 5 = 15. Subtracting 5 from both sides, we get 2x = 10. Dividing both sides by 2, we get x = 5.",
        "The area of a circle is given by A = πr^2. With radius r = 4 cm, the area is A = π * 4^2 = 16π cm^2.",
        "f(x) = x^2 - 3x + 2. Substituting x = 2, we get f(2) = 2^2 - 3*2 + 2 = 4 - 6 + 2 = 0."
    ]
})

math_test = pd.DataFrame({
    'prompt': [
        "Solve the equation: 3(x - 2) = 18",
        "Find the derivative of f(x) = x^3 - 2x^2 + 4x - 7"
    ],
    'response': [
        "We have 3(x - 2) = 18. Dividing both sides by 3, we get x - 2 = 6. Adding 2 to both sides, we get x = 8.",
        "The derivative of f(x) = x^3 - 2x^2 + 4x - 7 is f'(x) = 3x^2 - 4x + 4."
    ]
})

# Save to parquet files
gsm8k_train.to_parquet('/home/ubuntu/data/gsm8k/train.parquet', index=False)
gsm8k_test.to_parquet('/home/ubuntu/data/gsm8k/test.parquet', index=False)
math_train.to_parquet('/home/ubuntu/data/math/train.parquet', index=False)
math_test.to_parquet('/home/ubuntu/data/math/test.parquet', index=False)

print("Sample data created successfully.")
EOF

# Create a simplified validation script
echo "Creating validation script..."
cat > /home/ubuntu/grpo_tot_implementation/validate.py << 'EOF'
"""
Validation script for GRPO-ToT hybrid implementation.
This script performs a simplified validation of the implementation without requiring full training.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add implementation directory to path
sys.path.append('/home/ubuntu/grpo_tot_implementation')

# Import our modules
from grpo_tot_integration import (
    ToTNode, ToTTree, ToTExplorer, ToTRewardShaper, 
    PathImportanceWeighter, ToTRolloutProvider, ToTGRPOAdvantageEstimator
)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def validate_tot_node():
    """Validate ToTNode functionality."""
    logger.info("Validating ToTNode...")
    
    # Create a simple tree structure
    root = ToTNode("What is 2+2?")
    child1 = root.add_child("To solve 2+2, I need to add 2 and 2 together.")
    child2 = root.add_child("I'll use another approach to solve 2+2.")
    
    grandchild1 = child1.add_child("2+2 = 4")
    grandchild2 = child2.add_child("2+2 = 1+1+1+1 = 4")
    
    # Validate tree structure
    assert len(root.children) == 2, f"Expected 2 children, got {len(root.children)}"
    assert len(child1.children) == 1, f"Expected 1 child, got {len(child1.children)}"
    assert len(child2.children) == 1, f"Expected 1 child, got {len(child2.children)}"
    
    # Validate path to root
    path = grandchild1.get_path_to_root()
    assert len(path) == 3, f"Expected path length 3, got {len(path)}"
    assert path[0] == root, "First node in path should be root"
    assert path[1] == child1, "Second node in path should be child1"
    assert path[2] == grandchild1, "Third node in path should be grandchild1"
    
    logger.info("ToTNode validation passed!")
    return True

def validate_tot_explorer():
    """Validate ToTExplorer functionality."""
    logger.info("Validating ToTExplorer...")
    
    # Create a mock policy model
    class MockPolicyModel:
        def generate(self, prompt, n=5):
            return [f"{prompt} - thought {i}" for i in range(n)]
    
    # Initialize explorer with mock policy
    explorer = ToTExplorer(MockPolicyModel(), branching_factor=3, max_depth=2)
    
    # Create a tree and explore it
    tree = ToTTree("What is 2+2?", max_depth=2, branching_factor=3)
    terminal_nodes = explorer.explore(tree)
    
    # Validate exploration results
    assert len(terminal_nodes) > 0, "Expected at least one terminal node"
    assert all(node.is_terminal for node in terminal_nodes), "All nodes in terminal_nodes should be marked as terminal"
    
    logger.info("ToTExplorer validation passed!")
    return True

def validate_tot_reward_shaper():
    """Validate ToTRewardShaper functionality."""
    logger.info("Validating ToTRewardShaper...")
    
    # Create a simple verifiable reward function
    def mock_verifiable_reward(thought):
        return 1.0 if "4" in thought else 0.0
    
    # Initialize reward shaper
    reward_shaper = ToTRewardShaper(mock_verifiable_reward, phase="exploration")
    
    # Create some test nodes
    correct_node = ToTNode("2+2 = 4")
    incorrect_node = ToTNode("2+2 = 5")
    
    # Calculate rewards
    correct_reward = reward_shaper.calculate_reward(correct_node)
    incorrect_reward = reward_shaper.calculate_reward(incorrect_node)
    
    # Validate rewards
    assert correct_reward > incorrect_reward, f"Expected correct_reward > incorrect_reward, got {correct_reward} <= {incorrect_reward}"
    
    # Test phase transition
    reward_shaper.set_phase("exploitation")
    new_correct_reward = reward_shaper.calculate_reward(correct_node)
    
    # In exploitation phase, verifiable reward should have higher weight
    assert new_correct_reward > correct_reward, f"Expected higher reward in exploitation phase, got {new_correct_reward} <= {correct_reward}"
    
    logger.info("ToTRewardShaper validation passed!")
    return True

def validate_path_importance_weighter():
    """Validate PathImportanceWeighter functionality."""
    logger.info("Validating PathImportanceWeighter...")
    
    # Initialize weighter
    weighter = PathImportanceWeighter()
    
    # Create a simple path
    root = ToTNode("What is 2+2?")
    child = root.add_child("I'll add 2 and 2 together.")
    terminal = child.add_child("2+2 = 4")
    terminal.reward = 1.0  # Correct answer
    
    path = terminal.get_path_to_root()
    
    # Calculate weight
    weight = weighter.calculate_weight(path)
    
    # Validate weight
    assert weight > 0, f"Expected positive weight, got {weight}"
    
    # Create another path with incorrect answer
    root2 = ToTNode("What is 2+2?")
    child2 = root2.add_child("I'll try a different approach.")
    terminal2 = child2.add_child("2+2 = 5")
    terminal2.reward = 0.0  # Incorrect answer
    
    path2 = terminal2.get_path_to_root()
    
    # Calculate weight
    weight2 = weighter.calculate_weight(path2)
    
    # Validate weight comparison
    assert weight > weight2, f"Expected weight > weight2, got {weight} <= {weight2}"
    
    logger.info("PathImportanceWeighter validation passed!")
    return True

def validate_integration():
    """Validate integration between components."""
    logger.info("Validating component integration...")
    
    # Load configuration
    config = load_config('/home/ubuntu/grpo_tot_implementation/config.json')
    tot_config = config.get('tot', {})
    
    # Create mock policy model
    class MockPolicyModel:
        def generate(self, prompt, n=5):
            return [f"{prompt} - thought {i}" for i in range(n)]
    
    # Initialize components
    explorer = ToTExplorer(
        MockPolicyModel(),
        branching_factor=tot_config.get('branching_factor', 5),
        max_depth=tot_config.get('max_depth', 12)
    )
    
    def mock_verifiable_reward(thought):
        return 1.0 if "correct" in thought.lower() else 0.0
    
    reward_shaper = ToTRewardShaper(
        mock_verifiable_reward,
        phase=tot_config.get('initial_phase', 'exploration')
    )
    
    rollout_provider = ToTRolloutProvider(
        MockPolicyModel(),
        explorer,
        reward_shaper
    )
    
    # Test rollout generation
    prompts = ["Solve: 2+2=?", "Calculate: 3*4"]
    rollouts, rewards = rollout_provider.generate_rollouts(prompts, n_rollouts=3)
    
    # Validate rollouts
    assert len(rollouts) == len(prompts), f"Expected {len(prompts)} rollout groups, got {len(rollouts)}"
    assert all(len(group) == 3 for group in rollouts), "Expected 3 rollouts per prompt"
    
    # Test advantage estimation
    advantage_estimator = ToTGRPOAdvantageEstimator({'epsilon': 0.2})
    advantages = advantage_estimator.estimate_advantages(rollouts[0], rewards[0])
    
    # Validate advantages
    assert len(advantages) == len(rewards[0]), f"Expected {len(rewards[0])} advantages, got {len(advantages)}"
    
    logger.info("Component integration validation passed!")
    return True

def run_validation():
    """Run all validation tests."""
    logger.info("Starting validation of GRPO-ToT hybrid implementation...")
    
    validation_functions = [
        validate_tot_node,
        validate_tot_explorer,
        validate_tot_reward_shaper,
        validate_path_importance_weighter,
        validate_integration
    ]
    
    results = []
    for func in validation_functions:
        try:
            result = func()
            results.append(result)
        except Exception as e:
            logger.error(f"Validation failed in {func.__name__}: {str(e)}")
            results.append(False)
    
    # Summarize results
    passed = sum(results)
    total = len(validation_functions)
    
    logger.info(f"Validation complete: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All validation tests passed!")
        return True
    else:
        logger.warning(f"Some validation tests failed ({total-passed}/{total})")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
EOF

# Run validation
echo "Running validation tests..."
cd /home/ubuntu/grpo_tot_implementation
python3 validate.py

# Check if validation was successful
if [ $? -eq 0 ]; then
    echo "Validation successful! The GRPO-ToT hybrid implementation is functioning as expected."
else
    echo "Validation failed. Please check the logs for details."
    exit 1
fi

# Create a README file
echo "Creating README file..."
cat > /home/ubuntu/grpo_tot_implementation/README.md << 'EOF'
# GRPO-ToT Hybrid Implementation

This repository contains an implementation of a hybrid system that combines Group Relative Policy Optimization (GRPO) with Tree-of-Thought (ToT) reasoning for enhanced LLM performance.

## Overview

The implementation integrates the VERL (Volcano Engine Reinforcement Learning for LLMs) library's GRPO capabilities with a custom Tree-of-Thought module to achieve both high sampling efficiency and expanded reasoning boundaries.

## Repository Structure

- `verl/`: The VERL library (cloned from https://github.com/volcengine/verl)
- `verl_documentation.md`: Documentation of the VERL library structure and GRPO components
- `tot_module_design.md`: Design document for the custom Tree-of-Thought module
- `grpo_tot_integration.py`: Integration of GRPO and ToT components
- `training_pipeline.py`: End-to-end training and evaluation pipeline
- `config.json`: Configuration file for the system
- `validate.py`: Validation script to verify functionality
- `README.md`: This file

## Key Components

### 1. Tree-of-Thought Module

The custom ToT module enables systematic exploration through branching and backtracking, expanding the reasoning boundary of language models. Key components include:

- `ToTNode`: Represents a single node in the reasoning tree
- `ToTTree`: Manages the overall tree structure and search process
- `ToTExplorer`: Handles the exploration strategy within the tree
- `ToTRewardShaper`: Implements the dual-phase reward mechanism

### 2. GRPO Integration

The implementation integrates with VERL's GRPO capabilities through:

- `ToTRolloutProvider`: Custom rollout provider for VERL that uses ToT exploration
- `ToTGRPOAdvantageEstimator`: Custom advantage estimator that combines ToT with GRPO

### 3. Training Pipeline

The training pipeline provides end-to-end functionality:

- `DataManager`: Handles data loading and preprocessing
- `ModelManager`: Handles model initialization and loading
- `EvaluationManager`: Handles evaluation metrics and reporting
- `TrainingPipeline`: Main training pipeline for the GRPO-ToT hybrid system

## Usage

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure the system by editing `config.json`

3. Run the training pipeline:
   ```
   python training_pipeline.py --config config.json
   ```

4. Validate the implementation:
   ```
   python validate.py
   ```

## Configuration

The system is configured through `config.json`, which includes settings for:

- Data loading and preprocessing
- Model initialization
- Training parameters
- ToT-specific parameters
- Evaluation metrics

## Performance Considerations

- The implementation is designed to be memory-efficient, with GRPO eliminating the need for a value network
- Tree search is optimized with pruning strategies and parallel exploration
- The system scales to large models and can be distributed across multiple GPUs

## Future Improvements

- Integration with multi-agent reasoning scenarios
- Extension to embodied reasoning tasks
- Development of more sophisticated adaptive computation mechanisms
EOF

echo "Creating requirements.txt file..."
cat > /home/ubuntu/grpo_tot_implementation/requirements.txt << 'EOF'
torch>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
tqdm>=4.62.0
transformers>=4.30.0
pyarrow>=7.0.0
wandb>=0.13.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
EOF

echo "Validation complete! All files have been created and the implementation has been validated."
echo "The GRPO-ToT hybrid implementation is ready for use."
