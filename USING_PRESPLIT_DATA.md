# Using Pre-Split Data in Course Modules

You now have professionally split train/validation/test datasets! Here's how to use them in your notebooks.

## 📁 What You Have

Located in `data/splits/`:
- **`pokemon_train.csv`** - 11,200 cards (70%) - Train your models
- **`pokemon_validation.csv`** - 2,400 cards (15%) - Tune hyperparameters
- **`pokemon_test.csv`** - 2,400 cards (15%) - ⚠️ Final evaluation ONLY!

All splits are stratified by Pokemon type to maintain class balance.

---

## 🚀 Quick Start

### Option 1: Use the Utility Functions (Recommended)

```python
# Import the utility functions
import sys
sys.path.append('data')
from load_splits import load_split_data, load_all_splits, load_xy_split

# Load training data
train_df = load_split_data("train")
print(f"Loaded {len(train_df)} training cards")

# Load all three splits at once
train_df, val_df, test_df = load_all_splits()

# Load with X/y separation (ready for sklearn)
X_train, y_train = load_xy_split("train", target_col="type")
X_val, y_val = load_xy_split("validation", target_col="type")
```

### Option 2: Load Directly with Pandas

```python
import pandas as pd

# Load splits
train_df = pd.read_csv("data/splits/pokemon_train.csv")
val_df = pd.read_csv("data/splits/pokemon_validation.csv")
test_df = pd.read_csv("data/splits/pokemon_test.csv")

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
```

---

## 📚 Using in Course Modules

### Module 1: Data Engineering

```python
# Load training data for validation and cleaning
train_df = load_split_data("train")

# Perform data quality checks on training set
# Build validation schemas
# Create cleaning pipelines
```

### Module 2: EDA & Feature Engineering

```python
# Use training data for exploration
train_df = load_split_data("train")

# Perform EDA
# Engineer features
# Create preprocessing pipelines (fit ONLY on training data!)

# After feature engineering, prepare for modeling
feature_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
                'generation', 'is_legendary', 'total_stats', 'physical_bias', ...]

X_train, y_train = load_xy_split("train", target_col="type", feature_cols=feature_cols)
X_val, y_val = load_xy_split("validation", target_col="type", feature_cols=feature_cols)
```

### Module 3: Model Training

```python
# Load train and validation sets
X_train, y_train = load_xy_split("train", target_col="type")
X_val, y_val = load_xy_split("validation", target_col="type")

# Train models on training set
model.fit(X_train, y_train)

# Evaluate on validation set
val_score = model.score(X_val, y_val)

# Tune hyperparameters using validation set
# Select best model based on validation performance
```

### Module 4: Model Evaluation

```python
# Use validation set for detailed evaluation
X_val, y_val = load_xy_split("validation", target_col="type")

# Get predictions
y_pred = model.predict(X_val)

# Perform error analysis on validation set
# Create confusion matrices
# Analyze misclassifications
```

### Module 5: Deployment & Final Evaluation

```python
# After selecting your final model, evaluate ONCE on test set
X_test, y_test = load_xy_split("test", target_col="type")

# Final evaluation (do this ONLY ONCE!)
test_score = model.score(X_test, y_test)
y_pred_test = model.predict(X_test)

print(f"Final Test Score: {test_score:.2%}")
```

### Module 8: Capstone (Price Prediction)

```python
# For regression task (predicting price_usd)
X_train, y_train = load_xy_split("train", target_col="price_usd")
X_val, y_val = load_xy_split("validation", target_col="price_usd")

# After final model selection
X_test, y_test = load_xy_split("test", target_col="price_usd")

# Evaluate
from sklearn.metrics import mean_absolute_error, r2_score
test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print(f"Test MAE: ${mae:.2f}")
print(f"Test R²: {r2:.3f}")
```

---

## ⚠️ Important Rules

### The Golden Rule of Test Sets

**NEVER touch the test set until final evaluation!**

❌ **Don't**:
- Look at test set during EDA
- Use test set to select features
- Use test set to tune hyperparameters
- Use test set to compare models
- Evaluate on test set multiple times

✅ **Do**:
- Use training set for EDA and feature engineering
- Use validation set for hyperparameter tuning
- Use validation set for model selection
- Use validation set for detailed error analysis
- Use test set **ONLY ONCE** for final evaluation

### Why This Matters

If you use the test set during development, you'll overfit to it, and your performance estimates will be overly optimistic. The test set must remain "unseen" to give an honest estimate of real-world performance.

---

## 🔄 Workflow Summary

```
1. Explore → Use training set
2. Engineer features → Fit on training, transform train & val
3. Train models → Fit on training set
4. Tune hyperparameters → Validate on validation set
5. Compare models → Use validation scores
6. Select best model → Based on validation performance
7. Final evaluation → Use test set ONCE
```

---

## 📊 Advantages of Pre-Split Data

✅ **Consistency**: Everyone uses the same splits (reproducible research)
✅ **No leakage**: Splits are done before any EDA or preprocessing
✅ **Balanced**: Stratified by type to maintain class distribution
✅ **Realistic**: 70/15/15 split matches industry standards
✅ **Large enough**: 11,200 training cards is substantial

---

## 🔧 Regenerating Splits

If you want to create new splits (with a different seed):

```bash
# Edit random_seed in data/create_splits.py
# Then run:
uv run python data/create_splits.py
```

---

## 💡 Pro Tips

1. **Save preprocessing artifacts**: When you fit scalers/encoders on training data, save them to apply to validation/test

```python
from sklearn.preprocessing import StandardScaler
import pickle

# Fit on training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Transform validation and test using the SAME scaler
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

2. **Use pipelines**: Sklearn pipelines prevent leakage automatically

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Fit on training (scaler is fit on training data only)
pipe.fit(X_train, y_train)

# Evaluate on validation (scaler is applied, not refit)
val_score = pipe.score(X_val, y_val)
```

3. **Track which set you're using**: Add comments or variables

```python
current_set = "training"  # Make it explicit
print(f"Working with {current_set} set")
```

---

## 📖 Further Reading

- [sklearn train_test_split documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Why you need three sets, not two](https://machinelearningmastery.com/difference-test-validation-datasets/)
- [Common data leakage mistakes](https://www.kaggle.com/code/alexisbcook/data-leakage)

---

**Happy Training! 🚀**

Remember: Train on train, tune on validation, evaluate ONCE on test!
