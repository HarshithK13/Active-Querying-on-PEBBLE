# Setup Fixes Applied

## Issues Fixed

### 1. TensorFlow/TensorBoard Compatibility Issue

**Problem**: AttributeError with `module 'tensorflow' has no attribute 'io'`

**Solution**: Modified `logger.py` to use `tensorboardX` as fallback when `torch.utils.tensorboard` has compatibility issues.

```python
try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, AttributeError):
    from tensorboardX import SummaryWriter
```

### 2. Missing Dependencies

**Installed**:
- `hydra-core==1.3.2` - For configuration management
- `scikit-learn==1.3.2` - For K-means clustering in DUO
- `kneed==0.8.5` - For elbow method in adaptive K selection
- `tensorboard==2.14.0` - Compatible version
- `protobuf==4.25.8` - Compatible with TensorFlow 2.14

## DUO Sampling Integration

DUO sampling is now available as `feed_type=6` in the training pipeline.

### Usage

```bash
# Run with DUO sampling
./scripts/walker_walk/500/oracle/run_PEBBLE.sh 6

# Or directly
python train_PEBBLE.py env=walker_walk seed=12345 feed_type=6 [other args...]
```

### Available Sampling Methods

| feed_type | Method |
|-----------|--------|
| 0 | Uniform |
| 1 | Disagreement |
| 2 | Entropy |
| 3 | KCenter |
| 4 | KCenter_Disagree |
| 5 | KCenter_Entropy |
| 6 | **DUO** (Diverse, Uncertain, On-Policy) |

## Verification

All components tested and working:
- ✓ Logger import successful
- ✓ RewardModel with DUO import successful
- ✓ DUO sampling functional test passed
- ✓ Training script loads correctly

## Next Steps

You can now run your experiments with DUO sampling:

```bash
./scripts/walker_walk/500/oracle/run_PEBBLE.sh 6
```

The DUO method will:
1. Generate on-policy queries (currently using uniform sampling as fallback)
2. Filter by epistemic uncertainty (preference interval length)
3. Select diverse queries via adaptive K-means in reward difference space
