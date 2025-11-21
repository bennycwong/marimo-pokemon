"""
Module 3: Model Training & Experimentation
===========================================

Professional ML Engineering Onboarding Project
Pokemon Card Type Classification

Learning Objectives:
- Think in experiments, not models
- Understand the model zoo (when to use what)
- Master cross-validation
- Track experiments systematically
- Tune hyperparameters efficiently

Duration: 3-4 hours
"""

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Module 3: Model Training & Experimentation

    **"Think in experiments, not models"** — Every successful ML team

    Welcome to the most exciting part: training models! But here's the key insight:
    **Success in ML comes from systematic experimentation, not finding the "best" algorithm.**

    ## What You'll Learn

    1. **Experimentation Mindset**: Formulate hypotheses and test systematically
    2. **The Model Zoo**: When to use linear models vs trees vs neural networks
    3. **Cross-Validation**: Get robust performance estimates
    4. **Hyperparameter Tuning**: Optimize models without overfitting
    5. **Experiment Tracking**: Never lose track of what works

    ## Industry Reality

    > "We ran 10,000+ experiments to improve our model by 2%"
    > — Google Brain team

    **Why systematic experimentation matters:**
    - No single "best" algorithm (it depends on your data)
    - Small improvements compound to big business value
    - Documentation prevents repeating mistakes
    - Reproducibility is critical for debugging

    Let's learn to experiment like a pro!
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from typing import Dict, List, Tuple, Any
    import warnings
    import time
    warnings.filterwarnings('ignore')

    # Visualization setup
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    return Any, Dict, Path, pd, plt, time


@app.cell
def _(Path, pd):
    # Load the data we prepared in Module 2
    DATA_PATH = Path("data/clean/pokemon_cards_clean_latest.csv")
    df = pd.read_csv(DATA_PATH)

    print(f"✅ Loaded {len(df)} Pokemon cards")
    print(f"✅ Target: {df['type'].nunique()} Pokemon types")
    return (df,)


@app.cell
def _(df, pd):
    # Feature engineering (from Module 2)
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering from Module 2."""
        df_feat = df.copy()

        # Engineered features
        df_feat['total_stats'] = (df_feat['hp'] + df_feat['attack'] + df_feat['defense'] +
                                   df_feat['sp_attack'] + df_feat['sp_defense'] + df_feat['speed'])
        df_feat['physical_bias'] = (df_feat['attack'] + df_feat['defense']) - (df_feat['sp_attack'] + df_feat['sp_defense'])
        df_feat['offensive_bias'] = (df_feat['attack'] + df_feat['sp_attack']) - (df_feat['defense'] + df_feat['sp_defense'])
        df_feat['stat_balance'] = df_feat[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].std(axis=1)
        df_feat['physical_ratio'] = df_feat['attack'] / (df_feat['defense'] + 1)
        df_feat['special_ratio'] = df_feat['sp_attack'] / (df_feat['sp_defense'] + 1)
        df_feat['hp_ratio'] = df_feat['hp'] / df_feat['total_stats']
        df_feat['physical_bulk'] = df_feat['hp'] * df_feat['defense']
        df_feat['special_bulk'] = df_feat['hp'] * df_feat['sp_defense']

        # Convert boolean to int (for sklearn compatibility)
        df_feat['legendary_int'] = df_feat['is_legendary'].astype(int)

        return df_feat

    df_eng = engineer_features(df)

    # Select features for modeling (use legendary_int instead of is_legendary)
    feature_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
                    'generation', 'legendary_int',
                    'total_stats', 'physical_bias', 'offensive_bias', 'stat_balance',
                    'physical_ratio', 'special_ratio', 'hp_ratio',
                    'physical_bulk', 'special_bulk']

    X = df_eng[feature_cols].copy()
    y = df_eng['type'].copy()

    print(f"✅ Features: {len(feature_cols)}")
    print(f"✅ Classes: {y.nunique()}")
    return X, y


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section 1: Data Splitting (The Foundation)

    **Critical question**: How do we know if our model is any good?

    Answer: We need to test it on data it hasn't seen during training!

    ### The Standard Split

    - **Training set** (70%): Model learns from this
    - **Validation set** (15%): Tune hyperparameters, compare models
    - **Test set** (15%): Final evaluation (touch once!)

    **Why 3 sets?**
    - Training: For learning
    - Validation: For model selection (prevents overfitting to test set!)
    - Test: Unbiased final performance estimate

    Let's split our data properly:
    """)
    return


@app.cell
def _(X, y):
    from sklearn.model_selection import train_test_split

    # Split: 70% train, 15% validation, 15% test
    # Use stratify to maintain class balance
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Dataset Split:")
    print(f"  Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Verify stratification worked
    print(f"\nClass balance check:")
    print(f"  Original:   {y.value_counts(normalize=True).head(3).to_dict()}")
    print(f"  Training:   {y_train.value_counts(normalize=True).head(3).to_dict()}")
    print(f"  Validation: {y_val.value_counts(normalize=True).head(3).to_dict()}")
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell
def _(X_test, X_train, X_val):
    # Preprocessing (fit on train, transform all)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("✅ Data preprocessed (scaled)")
    print(f"   Training shape: {X_train_scaled.shape}")
    return StandardScaler, X_test_scaled, X_train_scaled, X_val_scaled


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section 2: Baseline Model (Always Start Here!)

    **Rule #1 of ML**: Always start with a simple baseline.

    **Why?**
    - Know if your problem is even solvable
    - Understand the difficulty
    - Have something to compare against
    - Catch data issues early

    **Common baselines**:
    - **Dummy classifier**: Always predict most common class
    - **Simple model**: Logistic Regression or Decision Tree

    Let's establish our baseline:
    """)
    return


@app.cell
def _(X_train_scaled, X_val_scaled, y_train, y_val):
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score

    # Baseline 1: Most frequent class
    dummy_model = DummyClassifier(strategy='most_frequent')
    dummy_model.fit(X_train_scaled, y_train)
    dummy_pred = dummy_model.predict(X_train_scaled)
    dummy_val_pred = dummy_model.predict(X_val_scaled)

    dummy_train_acc = accuracy_score(y_train, dummy_pred)
    dummy_val_acc = accuracy_score(y_val, dummy_val_pred)

    print("Baseline 1: Dummy Classifier (Most Frequent)")
    print(f"  Training Accuracy:   {dummy_train_acc:.2%}")
    print(f"  Validation Accuracy: {dummy_val_acc:.2%}")
    print(f"  → This is our \"zero skill\" baseline\n")
    return accuracy_score, dummy_val_acc


@app.cell
def _(
    X_train_scaled,
    X_val_scaled,
    accuracy_score,
    dummy_val_acc,
    y_train,
    y_val,
):
    from sklearn.linear_model import LogisticRegression

    # Baseline 2: Logistic Regression
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train_scaled, y_train)

    baseline_train_pred = baseline_model.predict(X_train_scaled)
    baseline_val_pred = baseline_model.predict(X_val_scaled)

    baseline_train_acc = accuracy_score(y_train, baseline_train_pred)
    baseline_val_acc = accuracy_score(y_val, baseline_val_pred)

    print("Baseline 2: Logistic Regression")
    print(f"  Training Accuracy:   {baseline_train_acc:.2%}")
    print(f"  Validation Accuracy: {baseline_val_acc:.2%}")
    print(f"  → Actual ML baseline\n")

    improvement = baseline_val_acc - dummy_val_acc
    print(f"✅ Our features provide {improvement:.1%} improvement over random!")
    return LogisticRegression, baseline_train_acc, baseline_val_acc


@app.cell
def _(mo):
    mo.md("""
    ### 💡 Baseline Insights

    **Key observations**:
    - Dummy classifier: ~15-20% (just guessing most common type)
    - Logistic Regression: ~50-60% (learning real patterns!)

    **What this tells us**:
    - Problem is definitely solvable with ML
    - Features contain useful signal
    - But there's room for improvement (60% → 80%+?)

    This is realistic! Most ML projects start here.

    ---
    ## Section 3: The Model Zoo (Which Model When?)

    **Key insight**: No free lunch theorem says "no algorithm is best for all problems."

    So how do we choose? Let's train multiple model types and compare!

    ### Models We'll Try

    1. **Logistic Regression** (linear, fast, interpretable)
    2. **Decision Tree** (non-linear, interpretable, baseline for trees)
    3. **Random Forest** (ensemble of trees, robust, good default)
    4. **Gradient Boosting (XGBoost)** (often best for tabular data)

    Let's train all four and compare systematically:
    """)
    return


@app.cell
def _(
    LogisticRegression,
    X_train_scaled,
    X_val_scaled,
    accuracy_score,
    time,
    y_train,
    y_val,
):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    # Try to import XGBoost (optional)
    try:
        import xgboost as xgb
        xgb_available = True
    except Exception as e:
        xgb = None
        xgb_available = False
        print("⚠️  XGBoost not available (will skip XGBoost model)")
        print("   To install: brew install libomp (Mac)")

    # Dictionary to store results
    results = {}

    # Model 1: Logistic Regression (we already have this, but let's time it)
    start = time.time()
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_train_time = time.time() - start

    lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train_scaled))
    lr_val_acc = accuracy_score(y_val, lr_model.predict(X_val_scaled))

    results['Logistic Regression'] = {
        'model': lr_model,
        'train_acc': lr_train_acc,
        'val_acc': lr_val_acc,
        'train_time': lr_train_time
    }

    print("Model 1: Logistic Regression")
    print(f"  Train Acc: {lr_train_acc:.2%} | Val Acc: {lr_val_acc:.2%} | Time: {lr_train_time:.2f}s\n")
    return (
        DecisionTreeClassifier,
        RandomForestClassifier,
        lr_train_time,
        results,
        xgb,
        xgb_available,
    )


@app.cell
def _(
    DecisionTreeClassifier,
    X_train_scaled,
    X_val_scaled,
    accuracy_score,
    results,
    time,
    y_train,
    y_val,
):
    # Model 2: Decision Tree
    start_dt = time.time()
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    dt_train_time = time.time() - start_dt

    dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train_scaled))
    dt_val_acc = accuracy_score(y_val, dt_model.predict(X_val_scaled))

    results['Decision Tree'] = {
        'model': dt_model,
        'train_acc': dt_train_acc,
        'val_acc': dt_val_acc,
        'train_time': dt_train_time
    }

    print("Model 2: Decision Tree")
    print(f"  Train Acc: {dt_train_acc:.2%} | Val Acc: {dt_val_acc:.2%} | Time: {dt_train_time:.2f}s\n")
    return


@app.cell
def _(
    RandomForestClassifier,
    X_train_scaled,
    X_val_scaled,
    accuracy_score,
    results,
    time,
    y_train,
    y_val,
):
    # Model 3: Random Forest
    start_rf = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    rf_train_time = time.time() - start_rf

    rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train_scaled))
    rf_val_acc = accuracy_score(y_val, rf_model.predict(X_val_scaled))

    results['Random Forest'] = {
        'model': rf_model,
        'train_acc': rf_train_acc,
        'val_acc': rf_val_acc,
        'train_time': rf_train_time
    }

    print("Model 3: Random Forest")
    print(f"  Train Acc: {rf_train_acc:.2%} | Val Acc: {rf_val_acc:.2%} | Time: {rf_train_time:.2f}s\n")
    return rf_train_acc, rf_train_time, rf_val_acc


@app.cell
def _(
    X_train_scaled,
    X_val_scaled,
    accuracy_score,
    results,
    time,
    xgb,
    xgb_available,
    y_train,
    y_val,
):
    from sklearn.preprocessing import LabelEncoder

    # Model 4: XGBoost (if available)
    if xgb_available:
        # XGBoost needs integer labels
        le_xgb = LabelEncoder()
        y_train_encoded = le_xgb.fit_transform(y_train)
        y_val_encoded = le_xgb.transform(y_val)

        start_xgb = time.time()
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train_scaled, y_train_encoded)
        xgb_train_time = time.time() - start_xgb

        xgb_train_acc = accuracy_score(y_train_encoded, xgb_model.predict(X_train_scaled))
        xgb_val_acc = accuracy_score(y_val_encoded, xgb_model.predict(X_val_scaled))

        results['XGBoost'] = {
            'model': xgb_model,
            'train_acc': xgb_train_acc,
            'val_acc': xgb_val_acc,
            'train_time': xgb_train_time
        }

        print("Model 4: XGBoost")
        print(f"  Train Acc: {xgb_train_acc:.2%} | Val Acc: {xgb_val_acc:.2%} | Time: {xgb_train_time:.2f}s\n")
    else:
        # Skip XGBoost if not available
        le_xgb = None
        y_train_encoded = None
        y_val_encoded = None
        xgb_model = None
        xgb_train_acc = None
        xgb_val_acc = None
        xgb_train_time = None
        start_xgb = None
        print("Model 4: XGBoost - Skipped (not installed)\n")
    return


@app.cell
def _(pd, plt, results):
    # Compare all models
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df[['train_acc', 'val_acc', 'train_time']].round(4)

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df)
    print("=" * 70)

    # Visualize comparison
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy comparison
    _ax = _axes[0]
    comparison_df[['train_acc', 'val_acc']].plot(kind='bar', ax=_ax)
    _ax.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
    _ax.set_ylabel('Accuracy')
    _ax.set_xlabel('Model')
    _ax.legend(['Training', 'Validation'])
    _ax.set_ylim([0, 1])
    _ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Baseline')

    # Plot 2: Train time vs accuracy
    _ax2 = _axes[1]
    _ax2.scatter(comparison_df['train_time'], comparison_df['val_acc'], s=200, alpha=0.6)
    for idx, model_name in enumerate(comparison_df.index):
        _ax2.annotate(model_name, (comparison_df.iloc[idx]['train_time'], comparison_df.iloc[idx]['val_acc']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    _ax2.set_xlabel('Training Time (seconds)')
    _ax2.set_ylabel('Validation Accuracy')
    _ax2.set_title('Speed vs Accuracy Tradeoff', fontweight='bold', fontsize=14)
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return (comparison_df,)


@app.cell
def _(comparison_df, mo):
    best_model = comparison_df['val_acc'].idxmax()
    best_val_acc = comparison_df['val_acc'].max()

    mo.md(
        f"""
        ### 🏆 Model Comparison Results

        **Best model**: {best_model} ({best_val_acc:.2%} validation accuracy)

        **Key observations**:
        - Linear model (Logistic Regression): Fast but limited by linearity
        - Decision Tree: Can overfit easily (high train, lower val)
        - Random Forest: More robust, better generalization
        - XGBoost: Often the best for tabular data

        **Overfitting check**:
        - Gap between train and val accuracy indicates overfitting
        - Random Forest and XGBoost handle this better (ensemble methods)

        ### 🤔 Socratic Question

        **"Your validation accuracy is 90% but test accuracy is 70%. What happened?"**

        Think about:
        - Did you tune hyperparameters using validation set?
        - Did you peek at validation set too many times?
        - Is there data leakage?
        - Different distribution in test set?

        **Answer**: You overfit to the validation set through iterative tuning!
        This is why we have a separate test set we touch only once.

        ---
        ## Section 4: Cross-Validation (Robust Evaluation)

        **Problem**: What if our validation split got lucky/unlucky?

        **Solution**: **k-Fold Cross-Validation**

        Instead of one train/val split, we do k splits:
        - Split data into k folds
        - Train on k-1 folds, validate on 1 fold
        - Repeat k times
        - Average the results

        **Benefits**:
        - More robust performance estimate
        - Use all data for training (eventually)
        - Detect if performance varies by split

        Let's use 5-fold CV on our best model:
        """
    )
    return


@app.cell
def _(RandomForestClassifier, StandardScaler, X_train, y_train):
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline

    # Create pipeline (preprocessing + model)
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ])

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

    print("5-Fold Cross-Validation Results (Random Forest):")
    print(f"  Fold scores: {[f'{score:.2%}' for score in cv_scores]}")
    print(f"  Mean: {cv_scores.mean():.2%}")
    print(f"  Std:  {cv_scores.std():.2%}")
    print(f"\n✅ CV gives us a robust estimate: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    return Pipeline, cv, cv_scores


@app.cell
def _(cv_scores, plt):
    # Visualize CV scores
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.bar(range(1, len(cv_scores)+1), cv_scores, alpha=0.7, color='steelblue')
    _ax.axhline(cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.2%}')
    _ax.fill_between(range(1, len(cv_scores)+1),
                     cv_scores.mean() - cv_scores.std(),
                     cv_scores.mean() + cv_scores.std(),
                     alpha=0.2, color='red', label=f'±1 Std: {cv_scores.std():.2%}')
    _ax.set_xlabel('Fold')
    _ax.set_ylabel('Accuracy')
    _ax.set_title('Cross-Validation Scores by Fold', fontweight='bold', fontsize=14)
    _ax.legend()
    _ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ### 💡 Cross-Validation Insights

    **Why standard deviation matters**:
    - Low std: Model is stable across different data splits
    - High std: Model performance varies (might be overfitting or data issues)

    **When to use CV**:
    - ✅ Model selection (which algorithm?)
    - ✅ Feature selection (which features?)
    - ✅ Hyperparameter tuning (which settings?)
    - ❌ Final test evaluation (use held-out test set once!)

    ---
    ## Section 5: Hyperparameter Tuning

    **Hyperparameters** = Settings you choose before training

    Examples:
    - Random Forest: n_estimators, max_depth, min_samples_split
    - XGBoost: learning_rate, max_depth, n_estimators
    - Neural Networks: learning_rate, batch_size, hidden_layers

    **Challenge**: How do we find good hyperparameters?

    ### Approaches:
    1. **Manual tuning**: Try values one by one (slow!)
    2. **Grid Search**: Try all combinations (thorough but expensive)
    3. **Random Search**: Try random combinations (faster, often good enough)
    4. **Bayesian Optimization**: Smart search (advanced)

    Let's use Grid Search on Random Forest:
    """)
    return


@app.cell
def _(Pipeline, RandomForestClassifier, StandardScaler, X_train, cv, y_train):
    from sklearn.model_selection import GridSearchCV

    # Define parameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 15, 20, None],
        'classifier__min_samples_split': [2, 5, 10]
    }

    # Create pipeline
    rf_pipeline_tune = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # Grid search with CV
    print("Running Grid Search... (this may take 1-2 minutes)")
    grid_search = GridSearchCV(
        rf_pipeline_tune,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✅ Grid Search Complete!")
    print(f"   Best CV Score: {grid_search.best_score_:.2%}")
    print(f"   Best Parameters: {grid_search.best_params_}")
    return (grid_search,)


@app.cell
def _(X_val, accuracy_score, grid_search, rf_val_acc, y_val):
    # Evaluate best model on validation set
    best_rf_model = grid_search.best_estimator_
    val_pred_tuned = best_rf_model.predict(X_val)
    val_acc_tuned = accuracy_score(y_val, val_pred_tuned)

    print(f"\nTuned Model Performance:")
    print(f"  Validation Accuracy: {val_acc_tuned:.2%}")
    print(f"  Improvement from default: {(val_acc_tuned - rf_val_acc)*100:.1f} percentage points")
    return (val_acc_tuned,)


@app.cell
def _(mo):
    mo.md("""
    ### 🎯 Hyperparameter Tuning Insights

    **Key learnings**:
    - Tuning can improve performance by 2-5% (significant!)
    - But it's expensive (trying many combinations)
    - Diminishing returns after a point

    **Best practices**:
    1. Start with default hyperparameters
    2. Understand which hyperparameters matter most
    3. Use Random Search first (faster)
    4. Use Grid Search to refine
    5. Always validate on held-out data

    ### ⚠️ Common Pitfall: Hyperparameter Overfitting

    If you tune hyperparameters too aggressively on validation set,
    you overfit to the validation set!

    **Solution**: Use nested CV or keep a true test set untouched.

    ---
    ## Section 6: Experiment Tracking

    **Problem**: You've run 10 experiments. Which one was best? What settings did you use?

    **Solution**: Track everything!

    In production, use tools like:
    - **MLflow**: Open source experiment tracking
    - **Weights & Biases**: Cloud-based with rich visualizations
    - **Neptune**: Experiment management platform

    For this project, we'll use a simple approach:
    """)
    return


@app.cell
def _(
    Any,
    Dict,
    baseline_train_acc,
    baseline_val_acc,
    grid_search,
    lr_train_time,
    pd,
    rf_train_acc,
    rf_train_time,
    rf_val_acc,
    val_acc_tuned,
):
    # Simple experiment tracker
    class ExperimentTracker:
        """Simple experiment tracking system."""

        def __init__(self):
            self.experiments = []

        def log_experiment(
            self,
            name: str,
            model_type: str,
            hyperparameters: Dict[str, Any],
            train_acc: float,
            val_acc: float,
            train_time: float,
            notes: str = ""
        ):
            """Log an experiment."""
            experiment = {
                'name': name,
                'model_type': model_type,
                'hyperparameters': str(hyperparameters),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_time': train_time,
                'notes': notes
            }
            self.experiments.append(experiment)

        def get_summary(self) -> pd.DataFrame:
            """Get summary of all experiments."""
            return pd.DataFrame(self.experiments)

        def get_best(self, metric='val_acc') -> Dict:
            """Get best experiment by metric."""
            df = self.get_summary()
            best_idx = df[metric].idxmax()
            return df.loc[best_idx].to_dict()

    # Create tracker
    tracker = ExperimentTracker()

    # Log our experiments
    tracker.log_experiment(
        name="exp_001_baseline",
        model_type="Logistic Regression",
        hyperparameters={'max_iter': 1000},
        train_acc=baseline_train_acc,
        val_acc=baseline_val_acc,
        train_time=lr_train_time,
        notes="Baseline model"
    )

    tracker.log_experiment(
        name="exp_002_random_forest",
        model_type="Random Forest",
        hyperparameters={'n_estimators': 100, 'max_depth': 15},
        train_acc=rf_train_acc,
        val_acc=rf_val_acc,
        train_time=rf_train_time,
        notes="Default RF parameters"
    )

    tracker.log_experiment(
        name="exp_003_rf_tuned",
        model_type="Random Forest",
        hyperparameters=grid_search.best_params_,
        train_acc=grid_search.best_score_,
        val_acc=val_acc_tuned,
        train_time=rf_train_time,
        notes="Grid search tuned"
    )

    print("Experiment Summary:")
    print(tracker.get_summary()[['name', 'model_type', 'val_acc', 'train_time']].to_string(index=False))

    best_exp = tracker.get_best()
    print(f"\n🏆 Best Experiment: {best_exp['name']}")
    print(f"   Validation Accuracy: {best_exp['val_acc']:.2%}")
    return


@app.cell
def _(mo):
    mo.md("""
    ### 📊 Why Experiment Tracking Matters

    **In production**:
    - You'll run 100s or 1000s of experiments
    - Need to compare across weeks/months
    - Team members need to see what worked
    - Reproducibility is critical

    **What to track**:
    - ✅ Model type and hyperparameters
    - ✅ Training and validation metrics
    - ✅ Training time and resources
    - ✅ Feature set used
    - ✅ Data version
    - ✅ Code version (git commit)
    - ✅ Random seeds
    - ✅ Notes and observations

    ---
    ## Section 7: Key Takeaways & Socratic Questions

    ### ✅ What You Learned

    1. **Always start with a baseline** (know your starting point)
    2. **Try multiple model types** (no free lunch theorem)
    3. **Use cross-validation** (robust performance estimates)
    4. **Tune hyperparameters systematically** (grid/random search)
    5. **Track everything** (experiments, not just final models)

    ### 🤔 Socratic Questions

    1. **"Your validation accuracy is 90% but test accuracy is 70%. What happened?"**
       - Answer: Overfit to validation set through tuning

    2. **"When would you choose Logistic Regression over XGBoost, even if XGBoost is more accurate?"**
       - Think: Interpretability, speed, production constraints, model size

    3. **"You're tuning hyperparameters. Should you use test data to select the best model? Why not?"**
       - Answer: No! Test set must remain untouched until final evaluation

    4. **"Your model is 90% accurate but your PM is unhappy. What might you be missing?"**
       - Think: Class imbalance, cost-sensitive errors, business metrics vs ML metrics

    5. **"Training takes 8 hours. How do you experiment efficiently?"**
       - Think: Smaller data samples, simpler models first, distributed training, caching

    ---
    ## 🏢 Industry Context

    ### How Companies Do This at Scale

    **Netflix**:
    - 1000s of experiments tracked in MLflow
    - Automated hyperparameter tuning (Optuna)
    - A/B tests for final model selection
    - Retraining pipelines (Airflow)

    **Google**:
    - AutoML for hyperparameter search
    - Distributed training (multiple GPUs/TPUs)
    - Experiment management (internal tools)
    - Rigorous A/B testing

    **Small Startup**:
    - Manual tracking (spreadsheets initially)
    - Grid/random search for tuning
    - Simple retraining scripts
    - Focus on business metrics

    ### Common Pitfalls

    ⚠️ **Don't**: Train one model and call it done
    ✅ **Do**: Try multiple approaches systematically

    ⚠️ **Don't**: Tune hyperparameters on test set
    ✅ **Do**: Use validation set or CV for tuning

    ⚠️ **Don't**: Forget to track experiments
    ✅ **Do**: Log everything from day one

    ⚠️ **Don't**: Overfit through excessive tuning
    ✅ **Do**: Use nested CV or keep test set sacred

    ---
    ## 🎯 Module 3 Checkpoint

    You've completed Module 3 when you can:

    - [ ] Set up an experiment from scratch in 15 minutes
    - [ ] Explain bias-variance tradeoff with concrete example
    - [ ] Choose appropriate model for a new problem
    - [ ] Run cross-validation and interpret results
    - [ ] Tune hyperparameters without overfitting
    - [ ] Track experiments systematically

    **Next**: Module 4 - Model Evaluation & Validation

    In the next module, you'll learn how to:
    - Evaluate beyond accuracy
    - Analyze errors systematically
    - Create model cards
    - Communicate model quality to stakeholders
    """)
    return


if __name__ == "__main__":
    app.run()
