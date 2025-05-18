"""
Dataset preparation script for GRPO-ToT hybrid implementation.

This script downloads and preprocesses the selected datasets (GSM8K, HumanEval, MBPP)
and converts them to the format required by the GRPO-ToT training pipeline.
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directories
os.makedirs("gsm8k", exist_ok=True)
os.makedirs("humaneval", exist_ok=True)
os.makedirs("mbpp", exist_ok=True)

def prepare_gsm8k():
    """Prepare GSM8K dataset."""
    logger.info("Preparing GSM8K dataset...")
    
    try:
        # Load GSM8K dataset from Hugging Face
        gsm8k = load_dataset("gsm8k", "main")
        logger.info("Successfully loaded GSM8K from Hugging Face")
    except Exception as e:
        logger.warning(f"Error loading GSM8K from Hugging Face: {e}")
        logger.info("Creating synthetic GSM8K data for demonstration purposes")
        
        # Create synthetic data for demonstration
        synthetic_train = [
            {
                "question": "If John has 5 apples and gives 2 to Mary, how many apples does John have left?",
                "answer": "John has 5 apples initially. He gives 2 apples to Mary. So John has 5 - 2 = 3 apples left."
            },
            {
                "question": "Sarah has $10. She buys a book for $3 and a pen for $2. How much money does she have left?",
                "answer": "Sarah has $10 initially. She spends $3 on a book and $2 on a pen. So she spends $3 + $2 = $5 in total. Therefore, she has $10 - $5 = $5 left."
            },
            {
                "question": "A train travels at 60 miles per hour. How far will it travel in 2.5 hours?",
                "answer": "The train travels at 60 miles per hour. In 2.5 hours, it will travel 60 * 2.5 = 150 miles."
            }
        ]
        
        synthetic_test = [
            {
                "question": "Tom has 8 marbles. He loses 3 marbles. How many marbles does Tom have now?",
                "answer": "Tom has 8 marbles initially. He loses 3 marbles. So Tom has 8 - 3 = 5 marbles now."
            },
            {
                "question": "A box contains 24 red balls and 36 blue balls. What fraction of the balls are red?",
                "answer": "The box contains 24 red balls and 36 blue balls. The total number of balls is 24 + 36 = 60. The fraction of red balls is 24/60 = 2/5."
            }
        ]
        
        gsm8k = {
            "train": synthetic_train,
            "test": synthetic_test
        }
    
    # Process train split
    train_data = []
    for i, item in enumerate(tqdm(gsm8k["train"], desc="Processing GSM8K train")):
        # Extract question and answer
        question = item["question"]
        answer = item["answer"]
        
        # Format as prompt with instruction
        prompt = f"Solve the following math problem step by step:\n\n{question}"
        
        # Add to train data
        train_data.append({
            "prompt": prompt,
            "response": answer,
            "metadata": {
                "id": item.get("id", f"gsm8k_train_{i}")
            }
        })
    
    # Process test split
    test_data = []
    for i, item in enumerate(tqdm(gsm8k["test"], desc="Processing GSM8K test")):
        # Extract question and answer
        question = item["question"]
        answer = item["answer"]
        
        # Format as prompt with instruction
        prompt = f"Solve the following math problem step by step:\n\n{question}"
        
        # Add to test data
        test_data.append({
            "prompt": prompt,
            "response": answer,
            "metadata": {
                "id": item.get("id", f"gsm8k_test_{i}")
            }
        })
    
    # Convert to pandas DataFrame and save as parquet
    pd.DataFrame(train_data).to_parquet("gsm8k/train.parquet", index=False)
    pd.DataFrame(test_data).to_parquet("gsm8k/test.parquet", index=False)
    
    logger.info(f"GSM8K dataset prepared: {len(train_data)} train examples, {len(test_data)} test examples")
    return len(train_data), len(test_data)

def prepare_humaneval():
    """Prepare HumanEval dataset."""
    logger.info("Preparing HumanEval dataset...")
    
    try:
        # Load HumanEval dataset from Hugging Face
        humaneval = load_dataset("openai_humaneval")
        logger.info("Successfully loaded HumanEval from Hugging Face")
    except Exception as e:
        logger.warning(f"Error loading HumanEval from Hugging Face: {e}")
        logger.info("Creating synthetic HumanEval data for demonstration purposes")
        
        # Create synthetic data for demonstration
        synthetic_data = [
            {
                "task_id": "HumanEval/0",
                "prompt": "def double(x):\n    \"\"\"\n    Given a number x, return x * 2.\n    >>> double(2)\n    4\n    >>> double(5)\n    10\n    \"\"\"\n",
                "canonical_solution": "    return x * 2\n",
                "test": "def check(candidate):\n    assert candidate(2) == 4\n    assert candidate(5) == 10\n    assert candidate(0) == 0\n    assert candidate(-5) == -10\n"
            },
            {
                "task_id": "HumanEval/1",
                "prompt": "def factorial(n):\n    \"\"\"\n    Return the factorial of n.\n    >>> factorial(5)\n    120\n    >>> factorial(0)\n    1\n    \"\"\"\n",
                "canonical_solution": "    if n == 0:\n        return 1\n    return n * factorial(n-1)\n",
                "test": "def check(candidate):\n    assert candidate(0) == 1\n    assert candidate(1) == 1\n    assert candidate(5) == 120\n    assert candidate(10) == 3628800\n"
            },
            {
                "task_id": "HumanEval/2",
                "prompt": "def is_prime(n):\n    \"\"\"\n    Return True if n is a prime number, False otherwise.\n    >>> is_prime(2)\n    True\n    >>> is_prime(4)\n    False\n    \"\"\"\n",
                "canonical_solution": "    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
                "test": "def check(candidate):\n    assert candidate(2) == True\n    assert candidate(3) == True\n    assert candidate(4) == False\n    assert candidate(11) == True\n    assert candidate(121) == False\n"
            }
        ]
        
        humaneval = {"test": synthetic_data}
    
    # Process the dataset
    data = []
    for item in tqdm(humaneval["test"], desc="Processing HumanEval"):
        # Extract task, prompt and canonical solution
        task_id = item["task_id"]
        prompt = item["prompt"]
        canonical_solution = item["canonical_solution"]
        test_cases = item["test"]
        
        # Format as prompt with instruction
        formatted_prompt = f"Write a Python function to solve the following problem:\n\n{prompt}"
        
        # Add to data
        data.append({
            "prompt": formatted_prompt,
            "response": canonical_solution,
            "metadata": {
                "id": task_id,
                "test_cases": test_cases
            }
        })
    
    # Split into train and test (80/20 split)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Convert to pandas DataFrame and save as parquet
    pd.DataFrame(train_data).to_parquet("humaneval/train.parquet", index=False)
    pd.DataFrame(test_data).to_parquet("humaneval/test.parquet", index=False)
    
    logger.info(f"HumanEval dataset prepared: {len(train_data)} train examples, {len(test_data)} test examples")
    return len(train_data), len(test_data)

def prepare_mbpp():
    """Prepare MBPP dataset."""
    logger.info("Preparing MBPP dataset...")
    
    try:
        # Load MBPP dataset from Hugging Face
        mbpp = load_dataset("mbpp")
        logger.info("Successfully loaded MBPP from Hugging Face")
    except Exception as e:
        logger.warning(f"Error loading MBPP from Hugging Face: {e}")
        logger.info("Creating synthetic MBPP data for demonstration purposes")
        
        # Create synthetic data for demonstration
        synthetic_train = [
            {
                "task_id": "mbpp_1",
                "text": "Write a function to find the sum of the first n natural numbers.",
                "code": "def sum_n(n):\n    return n * (n + 1) // 2",
                "test_list": ["assert sum_n(5) == 15", "assert sum_n(10) == 55"]
            },
            {
                "task_id": "mbpp_2",
                "text": "Write a function to check if a string is a palindrome.",
                "code": "def is_palindrome(s):\n    return s == s[::-1]",
                "test_list": ["assert is_palindrome('radar') == True", "assert is_palindrome('hello') == False"]
            }
        ]
        
        synthetic_test = [
            {
                "task_id": "mbpp_3",
                "text": "Write a function to count the number of vowels in a string.",
                "code": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
                "test_list": ["assert count_vowels('hello') == 2", "assert count_vowels('python') == 1"]
            }
        ]
        
        mbpp = {
            "train": synthetic_train,
            "test": synthetic_test
        }
    
    # Process train split
    train_data = []
    for item in tqdm(mbpp["train"], desc="Processing MBPP train"):
        # Extract task, prompt and canonical solution
        task_id = item["task_id"]
        text = item["text"]
        code = item["code"]
        test_cases = item["test_list"]
        
        # Format as prompt with instruction
        prompt = f"Write a Python function to solve the following problem:\n\n{text}"
        
        # Add to train data
        train_data.append({
            "prompt": prompt,
            "response": code,
            "metadata": {
                "id": task_id,
                "test_cases": test_cases
            }
        })
    
    # Process test split
    test_data = []
    for item in tqdm(mbpp["test"], desc="Processing MBPP test"):
        # Extract task, prompt and canonical solution
        task_id = item["task_id"]
        text = item["text"]
        code = item["code"]
        test_cases = item["test_list"]
        
        # Format as prompt with instruction
        prompt = f"Write a Python function to solve the following problem:\n\n{text}"
        
        # Add to test data
        test_data.append({
            "prompt": prompt,
            "response": code,
            "metadata": {
                "id": task_id,
                "test_cases": test_cases
            }
        })
    
    # Convert to pandas DataFrame and save as parquet
    pd.DataFrame(train_data).to_parquet("mbpp/train.parquet", index=False)
    pd.DataFrame(test_data).to_parquet("mbpp/test.parquet", index=False)
    
    logger.info(f"MBPP dataset prepared: {len(train_data)} train examples, {len(test_data)} test examples")
    return len(train_data), len(test_data)

def main():
    """Main function to prepare all datasets."""
    logger.info("Starting dataset preparation...")
    
    # Prepare GSM8K
    gsm8k_train, gsm8k_test = prepare_gsm8k()
    
    # Prepare HumanEval
    humaneval_train, humaneval_test = prepare_humaneval()
    
    # Prepare MBPP
    mbpp_train, mbpp_test = prepare_mbpp()
    
    # Create dataset summary
    summary = {
        "gsm8k": {
            "train_examples": gsm8k_train,
            "test_examples": gsm8k_test,
            "train_path": "gsm8k/train.parquet",
            "test_path": "gsm8k/test.parquet"
        },
        "humaneval": {
            "train_examples": humaneval_train,
            "test_examples": humaneval_test,
            "train_path": "humaneval/train.parquet",
            "test_path": "humaneval/test.parquet"
        },
        "mbpp": {
            "train_examples": mbpp_train,
            "test_examples": mbpp_test,
            "train_path": "mbpp/train.parquet",
            "test_path": "mbpp/test.parquet"
        }
    }
    
    # Save summary
    with open("dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Dataset preparation complete!")
    logger.info(f"Summary saved to dataset_summary.json")

if __name__ == "__main__":
    main()
