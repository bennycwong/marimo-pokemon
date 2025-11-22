# Pokemon Card Dataset Splits

## Split Information

Created from the full Pokemon card dataset with **stratified sampling by type**.

### Split Sizes
- **Training**: 11,200 cards (70%)
- **Validation**: 2,400 cards (15%)
- **Test**: 2,400 cards (15%)

### Files
- `pokemon_train.csv` - Training data (use for model training)
- `pokemon_validation.csv` - Validation data (use for hyperparameter tuning)
- `pokemon_test.csv` - Test data (**ONLY use for final evaluation!**)
- `split_metadata.json` - Split metadata and statistics

### Important Notes

⚠️ **Test Set Protocol**:
- The test set should **NEVER** be used during model development
- Only evaluate on test set **once** at the very end
- This prevents overfitting to the test set
- If test performance is bad, do NOT retune using test set!

✅ **Proper Usage**:
1. Train models on `pokemon_train.csv`
2. Tune hyperparameters using `pokemon_validation.csv`
3. Compare models using `pokemon_validation.csv`
4. Select best model based on validation performance
5. **Finally**, evaluate selected model **once** on `pokemon_test.csv`

### Stratification

All splits maintain the same class distribution as the original dataset:
- Stratified by Pokemon type
- Ensures balanced representation across all splits
- Prevents bias from rare types

### Usage in Course

**Module 1-2**: Use training set for EDA and feature engineering
**Module 3**: Train models on training set, validate on validation set
**Module 4**: Detailed evaluation on validation set
**Module 5**: Deploy model trained on train+validation, report test performance
**Module 8 (Capstone)**: Full workflow using all three splits properly

### Reproducibility

All splits use `random_seed=42` for reproducibility.
Re-running the split script will produce identical splits.

Generated: 2025-11-21 17:00:38
