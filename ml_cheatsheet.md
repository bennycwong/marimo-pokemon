# ML Engineering Cheatsheet

Quick reference guide for professional ML engineering practices.

---

## 📊 Data Engineering

### Data Quality Checks
```python
# Always validate your data
import pandera as pa

schema = pa.DataFrameSchema({
    "feature": pa.Column(float, pa.Check.ge(0)),
    "target": pa.Column(str, pa.Check.isin(['A', 'B', 'C']))
})

validated_df = schema.validate(df)
```

### Train/Val/Test Split
```python
# ALWAYS split before preprocessing!
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
```

### Preprocessing Pipeline (No Leakage!)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Fit on train only!
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)  # Fit on train
predictions = pipeline.predict(X_val)  # Transform val automatically
```

---

## 🔧 Feature Engineering

### Common Patterns
```python
# Ratios and interactions
df['ratio'] = df['numerator'] / (df['denominator'] + 1)  # +1 to avoid div by zero

# Aggregations
df['total'] = df[['col1', 'col2', 'col3']].sum(axis=1)

# Binning
df['category'] = pd.cut(df['continuous'], bins=[0, 25, 50, 100], labels=['low', 'med', 'high'])

# Domain features (use your knowledge!)
df['is_weekend'] = df['day_of_week'].isin([5, 6])
```

### Avoiding Data Leakage
```python
# ❌ WRONG - Uses target to create feature
df['target_mean'] = df.groupby('category')['target'].transform('mean')

# ✅ CORRECT - Only use features available at prediction time
df['category_count'] = df.groupby('category')['id'].transform('count')
```

---

## 🤖 Model Training

### Quick Model Comparison
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier()
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.2%} ± {scores.std():.2%}")
```

### Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

print(f"CV Score: {scores.mean():.2%} ± {scores.std():.2%}")
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

search = RandomizedSearchCV(
    model, param_dist, n_iter=20, cv=5, random_state=42
)
search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
print(f"Best score: {search.best_score_:.2%}")
```

---

## 📈 Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import classification_report, confusion_matrix

# Detailed per-class metrics
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```

### When to Use Which Metric

| Problem Type | Primary Metric | Why |
|-------------|----------------|-----|
| Balanced classes | Accuracy | All classes equally important |
| Imbalanced classes | F1-Score / ROC-AUC | Accounts for imbalance |
| Spam detection | Precision | False positives annoying |
| Cancer detection | Recall | Missing cases dangerous |
| Ranking | MAP, NDCG | Order matters |
| Regression | RMSE, MAE | Continuous output |

### Error Analysis
```python
# Find misclassified samples
errors = df[y_true != y_pred]

# Analyze error patterns
error_analysis = errors.groupby(['true_class', 'predicted_class']).size()
print(error_analysis.sort_values(ascending=False).head(10))

# Statistical analysis
errors[numeric_cols].describe()
```

---

## 🚀 Model Deployment

### Save Model + Metadata
```python
import pickle
import json

# Save model
with open('model_v1.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save metadata
metadata = {
    'version': 'v1.0.0',
    'features': feature_names,
    'classes': model.classes_.tolist(),
    'training_date': '2025-01-19'
}

with open('model_v1_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### Input Validation
```python
def validate_input(data, expected_features):
    """Validate prediction input."""
    # Check required features
    missing = set(expected_features) - set(data.keys())
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Check types and ranges
    for feature, value in data.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"{feature} must be numeric")
        if value < 0 or value > 1000:
            raise ValueError(f"{feature} out of range")

    return True
```

### Prediction with Error Handling
```python
def predict_safe(model, input_data):
    """Make prediction with comprehensive error handling."""
    try:
        # Validate
        validate_input(input_data, expected_features)

        # Prepare
        input_df = pd.DataFrame([input_data])[expected_features]

        # Predict
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(max(probabilities))

        return {
            'success': True,
            'prediction': prediction,
            'confidence': confidence
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'prediction': None
        }
```

---

## 🔍 Monitoring

### Data Drift Detection
```python
def detect_drift(reference_data, new_data, threshold=0.1):
    """Detect distribution drift in features."""
    drift_detected = {}

    for col in reference_data.columns:
        # Compare distributions
        ref_mean, ref_std = reference_data[col].mean(), reference_data[col].std()
        new_mean, new_std = new_data[col].mean(), new_data[col].std()

        # Normalized difference
        drift_score = abs(new_mean - ref_mean) / (ref_std + 1e-6)

        drift_detected[col] = drift_score > threshold

    return drift_detected
```

### Performance Monitoring
```python
# Track these metrics in production
metrics = {
    'predictions_per_second': ...,
    'latency_p95_ms': ...,
    'error_rate': ...,
    'prediction_distribution': ...,
    'confidence_distribution': ...,
    'feature_drift_score': ...
}

# Alert if thresholds exceeded
if metrics['latency_p95_ms'] > 500:
    send_alert("High latency detected")
```

---

## 🧠 Key Principles

### Data
1. **Always validate** - Data quality issues = model failures
2. **Split first** - Prevent leakage by splitting before preprocessing
3. **Version everything** - Data, features, models

### Features
4. **Domain knowledge** - Best features come from understanding the problem
5. **Check for leakage** - "Will I have this at prediction time?"
6. **Simple > complex** - Ratios and sums often beat fancy features

### Models
7. **Start simple** - Baseline first, complex models later
8. **Cross-validate** - Single validation set can be lucky/unlucky
9. **Track experiments** - Log everything, compare systematically

### Evaluation
10. **Beyond accuracy** - Choose metrics based on business impact
11. **Analyze errors** - Understanding failures > overall scores
12. **Document limitations** - Model cards prevent misuse

### Production
13. **Fail gracefully** - Handle all edge cases
14. **Monitor everything** - Models degrade without monitoring
15. **Iterate** - ML is never "done"

---

## 🛠️ Tool Recommendations

### Data Processing
- **Small data (<1GB)**: pandas
- **Large data (>10GB)**: polars, DuckDB
- **Massive data (>100GB)**: Spark, Dask

### ML Frameworks
- **Tabular data**: scikit-learn, XGBoost, LightGBM
- **Deep learning**: PyTorch (growing), TensorFlow (declining)
- **Auto-ML**: H2O AutoML, AutoGluon

### MLOps
- **Experiment tracking**: MLflow, Weights & Biases
- **Model serving**: BentoML, FastAPI, TorchServe
- **Monitoring**: Evidently, WhyLabs, Arize
- **Orchestration**: Airflow, Prefect, Dagster

---

## 📚 Further Reading

### Books
- "Designing Machine Learning Systems" by Chip Huyen
- "Machine Learning Design Patterns" by Lakshmanan et al.
- "Feature Engineering for Machine Learning" by Zheng & Casari

### Online Resources
- scikit-learn documentation (best ML docs)
- Made With ML (end-to-end tutorials)
- Full Stack Deep Learning (production ML course)

### Communities
- r/MachineLearning (research)
- r/LearnMachineLearning (beginners)
- MLOps Community (Slack)

---

## 🎯 Quick Decision Trees

### "Which model should I use?"
```
Tabular data?
  → Try XGBoost/LightGBM first
  → Then Random Forest
  → Then Linear models

Text data?
  → Try pre-trained transformers (Hugging Face)
  → Then fine-tune

Images?
  → Try pre-trained CNNs (torchvision)
  → Then fine-tune

Time series?
  → Try ARIMA, Prophet
  → Then LSTM/Transformers
```

### "My model isn't working, what do I do?"
```
High training error?
  → Underfitting
  → Try: More features, complex model, less regularization

Low training, high validation error?
  → Overfitting
  → Try: More data, simpler model, more regularization, better features

Low error everywhere but fails in prod?
  → Data drift or leakage
  → Check: Input distributions, feature engineering, training process
```

---

**Remember**: ML is iterative. Start simple, measure everything, improve systematically!
