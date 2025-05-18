# Tree-of-Thought Module Design

## Overview
This document outlines the design of a custom Tree-of-Thought (ToT) module for integration with the VERL library's GRPO implementation. The ToT module enables systematic exploration through branching and backtracking, expanding the reasoning boundary of language models while maintaining the efficiency benefits of GRPO.

## Architecture

### 1. Core Components

#### 1.1 ToTNode
```python
class ToTNode:
    """Represents a single node in the Tree-of-Thought."""
    def __init__(self, thought, parent=None, depth=0):
        self.thought = thought          # The text content of this thought
        self.parent = parent            # Parent node reference
        self.children = []              # Child nodes
        self.value = None               # Estimated value of this node
        self.reward = None              # Reward received for this node
        self.depth = depth              # Depth in the tree
        self.is_terminal = False        # Whether this is a terminal node
        self.metadata = {}              # Additional metadata
```

#### 1.2 ToTTree
```python
class ToTTree:
    """Manages the overall tree structure and search process."""
    def __init__(self, root_thought, max_depth=12, branching_factor=5):
        self.root = ToTNode(root_thought)
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.current_node = self.root
        self.terminal_nodes = []
        self.exploration_history = []
```

#### 1.3 ToTExplorer
```python
class ToTExplorer:
    """Handles the exploration strategy within the tree."""
    def __init__(self, policy_model, value_estimator, branching_factor=5):
        self.policy_model = policy_model          # GRPO-trained policy
        self.value_estimator = value_estimator    # Value estimation function
        self.branching_factor = branching_factor
        self.exploration_temp = 0.8               # Temperature for exploration
```

#### 1.4 ToTRewardShaper
```python
class ToTRewardShaper:
    """Implements the dual-phase reward mechanism."""
    def __init__(self, verifiable_reward_fn, phase="exploration"):
        self.verifiable_reward_fn = verifiable_reward_fn
        self.phase = phase
        self.alpha = 0.7 if phase == "exploitation" else 0.3  # Verifiable reward weight
        self.beta = 0.2 if phase == "exploitation" else 0.5   # Novelty reward weight
        self.gamma = 0.1 if phase == "exploitation" else 0.2  # Count-based penalty weight
        self.seen_patterns = Counter()                        # Track reasoning patterns
```

### 2. Integration Interfaces

#### 2.1 VERLToTAdapter
```python
class VERLToTAdapter:
    """Adapts the ToT module to work with VERL's interfaces."""
    def __init__(self, tot_explorer, tot_reward_shaper):
        self.tot_explorer = tot_explorer
        self.tot_reward_shaper = tot_reward_shaper
        
    def generate_rollouts(self, prompts, n_rollouts):
        """Interface with VERL's rollout generation."""
        # Implementation details
        
    def calculate_advantages(self, rollouts, rewards):
        """Interface with VERL's advantage calculation."""
        # Implementation details
```

#### 2.2 ToTRolloutGenerator
```python
class ToTRolloutGenerator:
    """Generates rollouts using tree search."""
    def __init__(self, tot_explorer, max_tree_size=1000):
        self.tot_explorer = tot_explorer
        self.max_tree_size = max_tree_size
        
    def generate(self, prompt, n_rollouts):
        """Generate n_rollouts using tree search."""
        # Implementation details
```

## Workflow

### 1. Initialization
1. Initialize the ToT module with the GRPO-trained policy model
2. Configure the exploration parameters (branching factor, max depth)
3. Set up the reward shaping mechanism with appropriate weights

### 2. Tree Construction and Exploration
1. For each input prompt, create a root node
2. Use the policy model to generate candidate thoughts
3. Expand the tree by selecting the most promising thoughts
4. Continue exploration until reaching terminal nodes or max depth
5. Backtrack and explore alternative branches as needed

### 3. Reward Calculation
1. Calculate verifiable rewards for terminal nodes
2. Apply novelty rewards based on the uniqueness of reasoning paths
3. Apply count-based penalties to discourage repetitive patterns
4. Combine rewards according to the current phase (exploration/exploitation)

### 4. Integration with GRPO
1. Convert tree exploration results into a format compatible with VERL
2. Feed the results into VERL's advantage estimation (GRPO)
3. Update the policy model based on the calculated advantages

## Implementation Details

### 1. Tree Expansion Strategy
```python
def expand_node(self, node):
    """Expand a node by generating and selecting candidate thoughts."""
    if node.depth >= self.max_depth or node.is_terminal:
        return []
        
    # Generate candidate thoughts using the policy model
    candidates = self.policy_model.generate_thoughts(node.thought, self.branching_factor)
    
    # Score candidates
    scored_candidates = []
    for thought in candidates:
        value = self.value_estimator.estimate(node.thought, thought)
        scored_candidates.append((thought, value))
    
    # Select top-m candidates
    sorted_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
    selected_candidates = sorted_candidates[:self.branching_factor]
    
    # Create child nodes
    child_nodes = []
    for thought, value in selected_candidates:
        child = ToTNode(thought, parent=node, depth=node.depth + 1)
        child.value = value
        node.children.append(child)
        child_nodes.append(child)
    
    return child_nodes
```

### 2. Backtracking Mechanism
```python
def backtrack(self, node, threshold=0.3):
    """Backtrack from the current node if its value is below threshold."""
    if node.parent is None:
        return node  # At root, can't backtrack further
    
    # Find sibling nodes that haven't been explored
    siblings = [child for child in node.parent.children if child not in self.exploration_history]
    
    if not siblings:
        # No unexplored siblings, backtrack to parent
        return self.backtrack(node.parent, threshold)
    
    # Find the highest-value unexplored sibling
    best_sibling = max(siblings, key=lambda x: x.value if x.value is not None else 0)
    
    if best_sibling.value > threshold:
        return best_sibling
    else:
        # No promising siblings, backtrack further
        return self.backtrack(node.parent, threshold)
```

### 3. Dual-Phase Reward Calculation
```python
def calculate_reward(self, node):
    """Calculate the combined reward for a node."""
    # Verifiable reward (e.g., correctness)
    rv = self.verifiable_reward_fn(node.thought)
    
    # Novelty reward
    reasoning_pattern = self.extract_reasoning_pattern(node)
    rn = self.calculate_novelty(reasoning_pattern)
    
    # Count-based penalty
    self.seen_patterns[reasoning_pattern] += 1
    rp = self.calculate_penalty(reasoning_pattern)
    
    # Combined reward
    combined_reward = self.alpha * rv + self.beta * rn - self.gamma * rp
    
    return combined_reward
```

### 4. Path Importance Weighting
```python
def calculate_path_importance(self, path):
    """Calculate importance weight for a reasoning path."""
    # Solution correctness
    wc = 1.0 if path[-1].reward > 0.5 else 0.0
    
    # Reasoning novelty
    reasoning_pattern = self.extract_reasoning_pattern(path)
    wn = self.calculate_novelty(reasoning_pattern)
    
    # Path efficiency
    we = 1.0 / len(path) if len(path) > 0 else 0.0
    
    # Generalization potential (estimated)
    wg = self.estimate_generalization(path)
    
    # Combined weight
    weight = self.lambda_c * wc + self.lambda_n * wn + self.lambda_e * we + self.lambda_g * wg
    
    return weight
```

## Integration with VERL

### 1. Custom Rollout Provider
```python
class ToTRolloutProvider:
    """Custom rollout provider for VERL."""
    def __init__(self, tot_explorer, tot_reward_shaper):
        self.tot_explorer = tot_explorer
        self.tot_reward_shaper = tot_reward_shaper
    
    def generate_rollouts(self, prompts, n_rollouts):
        """Generate rollouts using ToT exploration."""
        all_rollouts = []
        for prompt in prompts:
            # Create and explore tree
            tree = ToTTree(prompt)
            terminal_nodes = self.tot_explorer.explore(tree)
            
            # Convert terminal nodes to rollouts
            rollouts = []
            for node in terminal_nodes[:n_rollouts]:
                path = self.get_path_to_root(node)
                rollout = self.convert_path_to_rollout(path)
                rollouts.append(rollout)
            
            all_rollouts.append(rollouts)
        
        return all_rollouts
```

### 2. Custom Advantage Estimator
```python
class ToTGRPOAdvantageEstimator:
    """Custom advantage estimator that combines ToT with GRPO."""
    def __init__(self, tot_reward_shaper):
        self.tot_reward_shaper = tot_reward_shaper
    
    def estimate_advantages(self, rollouts, rewards):
        """Estimate advantages using GRPO with ToT-enhanced rewards."""
        # Group rewards
        grouped_rewards = self.group_rewards(rewards)
        
        # Calculate mean and std for each group
        group_stats = {}
        for group_id, group_rewards in grouped_rewards.items():
            group_stats[group_id] = {
                'mean': np.mean(group_rewards),
                'std': np.std(group_rewards) + 1e-8  # Avoid division by zero
            }
        
        # Calculate advantages
        advantages = []
        for i, reward in enumerate(rewards):
            group_id = self.get_group_id(rollouts[i])
            stats = group_stats[group_id]
            advantage = (reward - stats['mean']) / stats['std']
            advantages.append(advantage)
        
        return advantages
```

## Configuration and Hyperparameters

### 1. Tree Search Parameters
- `max_depth`: Maximum depth of the tree (default: 12)
- `branching_factor`: Number of branches to explore at each node (default: 5)
- `exploration_temperature`: Temperature for exploration vs. exploitation (default: 0.8)
- `backtracking_threshold`: Value threshold for backtracking (default: 0.3)

### 2. Reward Mechanism Parameters
- `alpha_exploration`: Weight for verifiable reward during exploration phase (default: 0.3)
- `beta_exploration`: Weight for novelty reward during exploration phase (default: 0.5)
- `gamma_exploration`: Weight for count-based penalty during exploration phase (default: 0.2)
- `alpha_exploitation`: Weight for verifiable reward during exploitation phase (default: 0.7)
- `beta_exploitation`: Weight for novelty reward during exploitation phase (default: 0.2)
- `gamma_exploitation`: Weight for count-based penalty during exploitation phase (default: 0.1)
- `phase_transition_schedule`: Schedule for transitioning from exploration to exploitation

### 3. Path Importance Weighting Parameters
- `lambda_c`: Weight for solution correctness (default: 0.4)
- `lambda_n`: Weight for reasoning novelty (default: 0.3)
- `lambda_e`: Weight for path efficiency (default: 0.2)
- `lambda_g`: Weight for generalization potential (default: 0.1)

## Performance Considerations

### 1. Memory Efficiency
- Implement pruning strategies to limit tree size
- Use lazy evaluation of node values
- Implement depth-limited search when memory constraints are tight

### 2. Computational Efficiency
- Parallelize tree exploration where possible
- Implement early stopping for unpromising branches
- Cache intermediate results to avoid redundant computation

### 3. Scalability
- Design for distributed exploration across multiple GPUs
- Implement load balancing for tree exploration
- Support dynamic adjustment of exploration parameters based on available resources
