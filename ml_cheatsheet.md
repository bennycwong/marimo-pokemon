# ML Engineering Cheatsheet

Quick reference guide for professional ML engineering practices covering all 8 modules.

---

## 💼 Business Context (Module 0)

### When to Use ML vs. Simple Rules

```python
# ❌ DON'T use ML for:
- Simple rules work (e.g., "flag transactions > $10k")
- You have < 100 labeled examples
- You need 100% accuracy (medical devices, etc.)
- You can't explain why (regulated industries)

# ✅ DO use ML for:
- Complex patterns humans can't codify
- You have lots of labeled data (1000s+)
- Probabilistic decisions are OK
- Problem is worth the cost ($100k+ value)
```

### ROI Calculation Template

```python
# Current state costs
manual_labor_cost = 50000  # $/year
error_cost = 20000  # $/year from mistakes
current_total = manual_labor_cost + error_cost

# ML solution costs
development_cost = 100000  # One-time (your time + infra)
maintenance_cost = 20000  # $/year
ml_total_year1 = development_cost + maintenance_cost

# Expected benefits
automation_savings = 50000  # $/year (less manual work)
quality_improvement = 30000  # $/year (fewer errors)
new_revenue = 50000  # $/year (new capabilities)
benefits = automation_savings + quality_improvement + new_revenue

# ROI calculation
roi_year1 = (benefits - ml_total_year1) / ml_total_year1
roi_year2 = (benefits - maintenance_cost) / maintenance_cost

print(f"Year 1 ROI: {roi_year1:.1%}")  # May be negative!
print(f"Year 2 ROI: {roi_year2:.1%}")  # Should be positive
```

### Metric Translation (Technical → Business)

```python
# Map model metrics to business impact
metric_translations = {
    'accuracy': 'How often we get it right',
    'precision': 'Of our predictions, how many are correct',
    'recall': 'Of all real cases, how many we catch',
    'f1_score': 'Balance between precision and recall',
    'rmse': 'Average prediction error in dollars',
    'latency': 'How fast users get results'
}

# Example: Explain to stakeholder
"""
Our model has 90% precision and 80% recall.

In plain English:
- Precision (90%): When we say "buy", we're right 90% of the time
- Recall (80%): We catch 80% of good opportunities

Business impact:
- 10% false positives = wasted effort on bad leads
- 20% false negatives = missed revenue opportunities
- Trade-off: Be more conservative (↑precision) or aggressive (↑recall)?
"""
```

---

## 📊 Data Engineering (Module 1)

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

## 🔍 Production Monitoring (Module 6)

### Data Drift Detection (Statistical Tests)
```python
from scipy import stats

def detect_drift_ks_test(reference_data, new_data, feature, alpha=0.05):
    """Detect drift using Kolmogorov-Smirnov test."""
    statistic, p_value = stats.ks_2samp(
        reference_data[feature],
        new_data[feature]
    )

    drift_detected = p_value < alpha

    if drift_detected:
        print(f"⚠️ DRIFT in {feature}: p={p_value:.4f}")

    return drift_detected

# Run weekly on production data
for feature in numeric_features:
    detect_drift_ks_test(train_data, prod_data, feature)
```

### Production Debugging Runbook

```python
# Scenario: Accuracy drops from 87% to 65%
"""
1. Check data drift:
   - Run KS tests on all features
   - Compare feature distributions (train vs prod)
   - Look for new categories in categorical features

2. Check data quality:
   - Missing values increased?
   - Out-of-range values?
   - New/unexpected data types?

3. Check model serving:
   - Correct model version loaded?
   - Feature preprocessing same as training?
   - Any code changes recently?

4. Check external factors:
   - Market event affecting behavior?
   - New upstream data source?
   - Seasonality (holidays, etc.)?

5. Mitigation:
   - Rollback to previous model if safe
   - Retrain with recent data
   - Add new features to handle shift
"""
```

### Alert Thresholds

```python
# Define clear alert levels
ALERT_THRESHOLDS = {
    'latency_p95_ms': {
        'warning': 100,   # Log and monitor
        'critical': 500,  # Page on-call
        'action': 'Scale up instances or optimize inference'
    },
    'error_rate': {
        'warning': 0.01,  # 1%
        'critical': 0.05,  # 5%
        'action': 'Check logs, rollback if needed'
    },
    'mae_increase': {
        'warning': 0.20,  # 20% worse
        'critical': 0.50,  # 50% worse
        'action': 'Investigate drift, retrain model'
    },
    'drift_features': {
        'warning': 1,     # 1 feature drifted
        'critical': 3,    # 3+ features drifted
        'action': 'Retrain with recent data'
    }
}
```

### Monitoring Dashboard Metrics

```python
# Essential metrics to track
production_metrics = {
    # Model performance
    'predictions_per_hour': lambda: count_predictions(last_hour),
    'avg_confidence': lambda: np.mean(confidence_scores),
    'low_confidence_rate': lambda: (confidence_scores < 0.7).mean(),

    # Service health
    'latency_p50_ms': lambda: np.percentile(latencies, 50),
    'latency_p95_ms': lambda: np.percentile(latencies, 95),
    'latency_p99_ms': lambda: np.percentile(latencies, 99),
    'error_rate': lambda: errors / total_requests,

    # Data quality
    'missing_values_rate': lambda: data.isna().mean().mean(),
    'out_of_range_rate': lambda: check_ranges(data),

    # Drift detection
    'drift_score': lambda: compute_drift(train_data, prod_data),
}
```

---

## 🤝 Team Collaboration (Module 7)

### Git Workflows for ML

```bash
# What to commit
✅ Commit:
- Code (*.py, *.ipynb)
- Configs (*.yaml, *.json)
- Requirements (requirements.txt, pyproject.toml)
- Documentation (README.md, model cards)
- Small datasets (< 10MB)

❌ Don't commit:
- Trained models (*.pkl, *.h5) - use model registry
- Large datasets (*.csv > 10MB) - use data versioning (DVC)
- Credentials (.env, *.key)
- Cache (__pycache__/, .ipynb_checkpoints/)
- Experiment outputs (mlruns/, wandb/)

# .gitignore for ML
__pycache__/
*.pyc
.ipynb_checkpoints/
mlruns/
wandb/
*.pkl
*.h5
*.pth
data/raw/
data/processed/
.env
.venv/
models/
```

### ML Code Review Checklist

```markdown
## Data & Features
- [ ] No data leakage (fit on train only)
- [ ] Features available at prediction time
- [ ] Train/val/test splits correct
- [ ] Data validation in place

## Reproducibility
- [ ] Random seeds set
- [ ] Dependencies versioned
- [ ] Configs documented
- [ ] Can reproduce results

## Model Training
- [ ] Baseline comparison included
- [ ] Cross-validation used (not single split)
- [ ] Hyperparameters logged
- [ ] Model artifacts saved properly

## Evaluation
- [ ] Appropriate metrics chosen
- [ ] Error analysis performed
- [ ] Model card created
- [ ] Test set only used once

## Code Quality
- [ ] Type hints on functions
- [ ] Docstrings present
- [ ] No hardcoded paths
- [ ] Error handling in place

## Production Readiness
- [ ] Input validation
- [ ] Edge cases handled
- [ ] Monitoring plan
- [ ] Rollback strategy
```

### PR Description Template

```markdown
## Summary
[1-2 sentence overview of changes]

## Changes
- **Data Pipeline**: [What changed]
- **Features**: [New/modified features]
- **Model**: [Algorithm changes]
- **Evaluation**: [Metrics, performance]

## Results
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Accuracy | 85.2% | 87.1% | +1.9% |
| F1 | 0.82 | 0.85 | +0.03 |
| Latency | 45ms | 42ms | -3ms |

## Testing
- [ ] Unit tests pass
- [ ] Cross-validation scores match
- [ ] No data leakage verified
- [ ] Edge cases tested

## Deployment Plan
1. Deploy to staging
2. Run A/B test (10% traffic, 48 hours)
3. Monitor metrics
4. Full rollout if metrics improve

## Questions for Reviewers
1. [Specific question]
2. [Another question]
```

### Common ML Code Smells

```python
# 🚨 Code Smell #1: Fitting on entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # WRONG!
X_train, X_test = train_test_split(X_scaled)

# ✅ Fix: Split first, fit on train only
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🚨 Code Smell #2: Using test set for decisions
if test_accuracy > 0.90:
    use_this_model()  # WRONG! You peeked!

# ✅ Fix: Use validation set for all decisions
if val_accuracy > 0.90:
    final_test_accuracy = model.score(X_test, y_test)  # Only once!

# 🚨 Code Smell #3: Target variable in features
df['avg_price_by_category'] = df.groupby('category')['price'].transform('mean')
# WRONG! At prediction time, we don't know the price yet!

# ✅ Fix: Only use information available at prediction time
df['count_by_category'] = df.groupby('category')['id'].transform('count')

# 🚨 Code Smell #4: No error handling
def predict(data):
    return model.predict(data)  # What if data is malformed?

# ✅ Fix: Validate and handle errors
def predict(data):
    try:
        validated = validate_schema(data)
        return model.predict(validated)
    except ValidationError as e:
        return {'error': str(e), 'prediction': None}
```

---

## 🧠 Key Principles (All 8 Modules)

### Business (Module 0)
1. **Frame problems first** - Understand business value before coding
2. **Calculate ROI** - ML is expensive, justify the investment
3. **Know when not to use ML** - Simple rules often win

### Data (Module 1)
4. **Always validate** - Data quality issues = model failures
5. **Split first** - Prevent leakage by splitting before preprocessing
6. **Version everything** - Data, features, models

### Features (Module 2)
7. **Domain knowledge** - Best features come from understanding the problem
8. **Check for leakage** - "Will I have this at prediction time?"
9. **Simple > complex** - Ratios and sums often beat fancy features

### Models (Module 3)
10. **Start simple** - Baseline first, complex models later
11. **Cross-validate** - Single validation set can be lucky/unlucky
12. **Track experiments** - Log everything, compare systematically

### Evaluation (Module 4)
13. **Beyond accuracy** - Choose metrics based on business impact
14. **Analyze errors** - Understanding failures > overall scores
15. **Document limitations** - Model cards prevent misuse

### Deployment (Module 5)
16. **Validate inputs** - Never trust user data
17. **Fail gracefully** - Handle all edge cases
18. **Test edge cases** - Unusual inputs will happen

### Production (Module 6)
19. **Monitor everything** - Models degrade without monitoring
20. **Have runbooks** - When (not if) things break, have a plan
21. **Automate retraining** - Drift happens, be ready

### Collaboration (Module 7)
22. **Review code thoroughly** - Most ML bugs caught in review
23. **Document decisions** - Future you will thank current you
24. **Communicate clearly** - Stakeholders aren't ML experts

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
