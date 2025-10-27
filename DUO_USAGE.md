# DUO Sampling Integration

## Overview

DUO (Diverse, Uncertain, On-Policy) sampling has been integrated as a query selection technique in the PEBBLE framework.

## Usage

To use DUO sampling, set `feed_type=6` in your configuration:

```bash
python train_PEBBLE.py feed_type=6
```

## Available Sampling Techniques

| feed_type | Method | Description |
|-----------|--------|-------------|
| 0 | Uniform | Random query selection |
| 1 | Disagreement | Based on ensemble disagreement |
| 2 | Entropy | Based on prediction entropy |
| 3 | KCenter | K-center greedy in state space |
| 4 | KCenter_Disagree | K-center + disagreement |
| 5 | KCenter_Entropy | K-center + entropy |
| **6** | **DUO** | **On-policy + uncertainty + diversity** |

## DUO Implementation Details

DUO implements a three-stage filtering pipeline:

1. **On-Policy (ξO)**: Priority sampling based on trajectory likelihood under current policy
   - Currently uses uniform sampling (policy log probs not computed)
   - Can be extended by computing policy log probabilities

2. **Uncertain (ξU)**: Filters consensual predictions and prioritizes epistemic uncertainty
   - Removes queries where all ensemble members agree
   - Measures uncertainty as preference interval length

3. **Diverse (ξD)**: Clustering-based selection in reward difference space
   - Represents queries in predicted reward difference space
   - Uses adaptive K-means clustering with elbow method

## Dependencies

DUO requires additional packages:
```bash
pip install scikit-learn>=1.3.2 kneed>=0.8.5
```

These are already added to `requirements.txt`.

## Configuration

DUO uses the same configuration parameters as other sampling methods:
- `reward_batch`: Number of queries to select (mb_size)
- `large_batch`: Multiplier for initial candidate pool
- `ensemble_size`: Number of reward models in ensemble (affects uncertainty estimation)
- `segment`: Length of trajectory segments for queries

## Extending with Policy Information

To enable full on-policy filtering, you can modify the code to compute policy log probabilities:

```python
# In learn_reward method, before calling duo_sampling:
policy_log_probs = []
for trajectory in self.reward_model.inputs:
    log_probs = []
    for obs, action in trajectory:
        # Compute log probability under current policy
        log_prob = self.agent.get_log_prob(obs, action)
        log_probs.append(log_prob)
    policy_log_probs.append(log_probs)

# Then call with policy information
labeled_queries = self.reward_model.duo_sampling(policy_log_probs=policy_log_probs)
```

## Paper Reference

Based on: "DUO: Diverse, Uncertain, On-Policy Query Generation and Selection for Reinforcement Learning from Human Feedback"
