"""
Create train/validation/test splits from the full Pokemon card dataset.

This creates stratified splits that maintain class balance across all sets,
ensuring the model is trained and evaluated fairly.

Splits:
- Training: 70% (~11,200 cards) - For model training
- Validation: 15% (~2,400 cards) - For hyperparameter tuning and model selection
- Test: 15% (~2,400 cards) - Final evaluation ONLY (never touch during training!)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("POKEMON CARD DATASET SPLITTER")
print("="*70)
print("\nThis will create train/validation/test splits from the full dataset.")
print("Splits will be stratified by Pokemon type to maintain class balance.\n")

# Load full dataset
data_path = Path("data/pokemon_cards.csv")
if not data_path.exists():
    print(f"❌ Error: {data_path} not found!")
    print("   Run: uv run python data/generate_comprehensive_dataset.py")
    exit(1)

df = pd.read_csv(data_path)
print(f"✅ Loaded {len(df):,} Pokemon cards from {data_path}")

# Create output directory
output_dir = Path("data/splits")
output_dir.mkdir(exist_ok=True)
print(f"✅ Created output directory: {output_dir}")

# Split strategy: 70% train, 15% validation, 15% test
# Use stratification by type to maintain class balance
print("\n" + "-"*70)
print("CREATING SPLITS (Stratified by Type)")
print("-"*70)

# First split: 70% train, 30% temp (which becomes 15% val, 15% test)
X = df.drop(columns=['type'])
y = df['type']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: Split temp into validation (50%) and test (50%)
# This gives us 15% validation and 15% test from the original dataset
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Reconstruct full dataframes
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print(f"\n✅ Split sizes:")
print(f"   Training:   {len(train_df):,} cards ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Validation: {len(val_df):,} cards ({len(val_df)/len(df)*100:.1f}%)")
print(f"   Test:       {len(test_df):,} cards ({len(test_df)/len(df)*100:.1f}%)")

# Verify stratification worked
print("\n" + "-"*70)
print("VERIFYING CLASS BALANCE (Top 5 Types)")
print("-"*70)

orig_dist = df['type'].value_counts(normalize=True).head()
train_dist = train_df['type'].value_counts(normalize=True).head()
val_dist = val_df['type'].value_counts(normalize=True).head()
test_dist = test_df['type'].value_counts(normalize=True).head()

comparison_df = pd.DataFrame({
    'Original': orig_dist,
    'Train': train_dist,
    'Validation': val_dist,
    'Test': test_dist
})

print(comparison_df.round(4))
print("\n✅ Class distributions are balanced across all splits!")

# Check rarity distribution
print("\n" + "-"*70)
print("RARITY DISTRIBUTION CHECK")
print("-"*70)

rarity_comparison = pd.DataFrame({
    'Original': df['rarity'].value_counts(normalize=True),
    'Train': train_df['rarity'].value_counts(normalize=True),
    'Validation': val_df['rarity'].value_counts(normalize=True),
    'Test': test_df['rarity'].value_counts(normalize=True)
})

print(rarity_comparison.round(4))

# Check legendary distribution
print("\n" + "-"*70)
print("LEGENDARY STATUS CHECK")
print("-"*70)

legendary_stats = pd.DataFrame({
    'Original': df['is_legendary'].value_counts(normalize=True),
    'Train': train_df['is_legendary'].value_counts(normalize=True),
    'Validation': val_df['is_legendary'].value_counts(normalize=True),
    'Test': test_df['is_legendary'].value_counts(normalize=True)
})

print(legendary_stats.round(4))

# Price statistics
print("\n" + "-"*70)
print("PRICE DISTRIBUTION CHECK")
print("-"*70)

price_stats = pd.DataFrame({
    'Original': df['price_usd'].describe(),
    'Train': train_df['price_usd'].describe(),
    'Validation': val_df['price_usd'].describe(),
    'Test': test_df['price_usd'].describe()
})

print(price_stats.round(2))

# Save splits
print("\n" + "-"*70)
print("SAVING SPLITS")
print("-"*70)

train_path = output_dir / "pokemon_train.csv"
val_path = output_dir / "pokemon_validation.csv"
test_path = output_dir / "pokemon_test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"✅ Saved training set:   {train_path}")
print(f"✅ Saved validation set: {val_path}")
print(f"✅ Saved test set:       {test_path}")

# Create a metadata file
metadata = {
    'dataset_info': {
        'total_cards': len(df),
        'train_cards': len(train_df),
        'val_cards': len(val_df),
        'test_cards': len(test_df),
        'split_ratio': '70/15/15',
        'stratification': 'By Pokemon type',
        'random_seed': 42,
    },
    'feature_columns': list(df.columns),
    'target_column': 'type',
    'num_classes': df['type'].nunique(),
    'class_names': sorted(df['type'].unique().tolist()),
}

import json
metadata_path = output_dir / "split_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved metadata:       {metadata_path}")

# Create a README
readme_content = f"""# Pokemon Card Dataset Splits

## Split Information

Created from the full Pokemon card dataset with **stratified sampling by type**.

### Split Sizes
- **Training**: {len(train_df):,} cards (70%)
- **Validation**: {len(val_df):,} cards (15%)
- **Test**: {len(test_df):,} cards (15%)

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

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

readme_path = output_dir / "README.md"
with open(readme_path, 'w') as f:
    f.write(readme_content)

print(f"✅ Saved README:         {readme_path}")

print("\n" + "="*70)
print("✅ SPLITS CREATED SUCCESSFULLY!")
print("="*70)

print(f"""
📁 Location: {output_dir}/

Files created:
1. pokemon_train.csv ({len(train_df):,} cards) - Train your models here
2. pokemon_validation.csv ({len(val_df):,} cards) - Tune hyperparameters here
3. pokemon_test.csv ({len(test_df):,} cards) - ⚠️  Final evaluation ONLY!
4. split_metadata.json - Split information
5. README.md - Usage instructions

⚠️  IMPORTANT: Never touch the test set until final evaluation!

You can now use these splits in your ML course modules!
""")

print("\nNext steps:")
print("  1. Use pokemon_train.csv for Modules 1-3 (training)")
print("  2. Use pokemon_validation.csv for Module 3-4 (tuning/selection)")
print("  3. Use pokemon_test.csv ONLY ONCE for Module 5 (final eval)")
print()
