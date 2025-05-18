# TODO List: GRPO-ToT Hybrid Framework Implementation

## 1. Model Integration

- [ ] Replace all placeholder policy models with a real LLM (e.g., HuggingFace Transformers)
- [ ] Implement a wrapper class for the LLM with a `.generate()` method

## 2. Thought Generation

- [ ] Update `ToTExplorer.generate_thoughts` to call the real LLM
- [ ] Support batch generation, temperature, top-k, etc.

## 3. Value Estimation

- [ ] Implement a real value estimator (value head or reward proxy)

## 4. Verifiable Reward Function

- [ ] Implement `verifiable_reward_fn` for your domain (code, math, etc.)
- [ ] Integrate reward function into `ToTRewardShaper`

## 5. Dual-Phase Reward

- [ ] Ensure `ToTRewardShaper` supports dynamic phase switching
- [ ] Log/track phase transitions during training

## 6. Data Loading

- [ ] Replace dataset stubs with real data loading (GSM8K, HumanEval, etc.)
- [ ] Implement batching and shuffling

## 7. Distillation with PIW

- [ ] Use `PathImportanceWeighter` to compute weights for all trajectories
- [ ] Implement student model and weighted cross-entropy loss
- [ ] Train student on weighted dataset

## 8. Evaluation

- [ ] Implement real evaluation metrics (pass@k, RDI, CES, GG)
- [ ] Log and save all results

## 9. Config and CLI

- [ ] Make all hyperparameters configurable via JSON/YAML
- [ ] Support easy experiment management via CLI

## 10. Testing and Logging

- [ ] Add unit tests for each module
- [ ] Add detailed logging for debugging and analysis
