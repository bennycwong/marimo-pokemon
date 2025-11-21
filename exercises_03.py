"""
Module 3 Exercises: Model Training & Experimentation
====================================================

These exercises reinforce the concepts from Module 3.
Complete each exercise and check your solutions.

Time estimate: 2-3 hours
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
    # Module 3 Exercises

    ## Exercise 3.1: Implement Cross-Validation from Scratch (45 min)

    **Goal**: Understand cross-validation deeply by implementing it yourself.

    **Instructions**:
    1. Implement k-fold cross-validation manually (no sklearn)
    2. Split data into k folds
    3. For each fold, train and evaluate
    4. Return mean and std of scores
    5. Compare your results to sklearn's cross_val_score

    **Learning Objective**: Deep understanding > using libraries blindly
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Load and prepare data (same as Module 3)
    DATA_PATH = Path("data/clean/pokemon_cards_clean_latest.csv")
    df = pd.read_csv(DATA_PATH)

    feature_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
                    'generation', 'is_legendary']
    X = df[feature_cols]
    y = df['type']

    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    return StandardScaler, X, accuracy_score, np, train_test_split, y


@app.cell
def _(np):
    # TODO: Implement k-fold cross-validation from scratch
    def manual_k_fold_cv(X, y, model, k=5, random_state=42):
        """
        Implement k-fold cross-validation manually.

        Args:
            X: Features (DataFrame or array)
            y: Target (Series or array)
            model: Sklearn model instance
            k: Number of folds
            random_state: Random seed

        Returns:
            scores: List of accuracy scores for each fold
        """
        # TODO: Set random seed
        np.random.seed(random_state)

        # TODO: Shuffle indices
        indices = np.arange(len(X))
        # np.random.shuffle(indices)

        # TODO: Split indices into k folds
        # fold_size = len(X) // k
        # folds = []
        # for i in range(k):
        #     start_idx = i * fold_size
        #     end_idx = start_idx + fold_size if i < k-1 else len(X)
        #     folds.append(indices[start_idx:end_idx])

        # TODO: For each fold:
        #   1. Use fold as validation, rest as training
        #   2. Scale data (fit on train, transform val)
        #   3. Train model
        #   4. Evaluate on validation fold
        #   5. Store score

        scores = []
        # for i in range(k):
        #     # Get train/val indices
        #     val_indices = folds[i]
        #     train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
        #
        #     # Split data
        #     X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        #     y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
        #
        #     # Scale
        #     scaler = StandardScaler()
        #     X_train_scaled = scaler.fit_transform(X_train)
        #     X_val_scaled = scaler.transform(X_val)
        #
        #     # Train
        #     model.fit(X_train_scaled, y_train)
        #
        #     # Evaluate
        #     y_pred = model.predict(X_val_scaled)
        #     score = accuracy_score(y_val, y_pred)
        #     scores.append(score)

        return scores

    # TODO: Test your implementation
    # model = LogisticRegression(max_iter=1000, random_state=42)
    # manual_scores = manual_k_fold_cv(X, y, model, k=5)
    # print(f"Manual CV scores: {[f'{s:.2%}' for s in manual_scores]}")
    # print(f"Mean: {np.mean(manual_scores):.2%}, Std: {np.std(manual_scores):.2%}")

    # TODO: Compare to sklearn
    # from sklearn.model_selection import cross_val_score
    # from sklearn.pipeline import Pipeline
    # pipeline = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('model', LogisticRegression(max_iter=1000, random_state=42))
    # ])
    # sklearn_scores = cross_val_score(pipeline, X, y, cv=5)
    # print(f"\nSklearn CV scores: {[f'{s:.2%}' for s in sklearn_scores]}")
    # print(f"Mean: {sklearn_scores.mean():.2%}, Std: {sklearn_scores.std():.2%}")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Reflection Questions:
    1. How close are your results to sklearn's?
    2. What did you learn by implementing it yourself?
    3. Why is it important to shuffle data before splitting?
    4. What happens if you forget to scale the validation set?

    ---

    ## Exercise 3.2: Model Selection Competition (60 min)

    **Goal**: Train 4+ models and select the best one systematically.

    **Scenario**: You're tasked with building a Pokemon type classifier.
    You need to recommend ONE model for production.

    **Instructions**:
    1. Train at least 4 different model types
    2. Use cross-validation for each
    3. Track all experiments (parameters, scores, time)
    4. Consider multiple criteria:
       - Accuracy
       - Training time
       - Inference speed
       - Model size
       - Interpretability
    5. Write a recommendation memo to your "PM"

    **Learning Objective**: Model selection is multi-criteria decision making.
    """)
    return


@app.cell
def _(X, train_test_split, y):
    # Split data
    X_train_ex, X_test_ex, y_train_ex, y_test_ex = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TODO: Train multiple models and track results
    # Models to try:
    # 1. Logistic Regression
    # 2. Decision Tree
    # 3. Random Forest
    # 4. XGBoost
    # 5. (Bonus) SVM, KNN, Naive Bayes, etc.

    results_competition = {}

    # TODO: For each model:
    #   - Train with cross-validation
    #   - Measure training time
    #   - Measure inference time (predict 1000 samples)
    #   - Calculate model size (use sys.getsizeof or pickle)
    #   - Record all metrics

    # Example structure:
    # from sklearn.ensemble import RandomForestClassifier
    # import time
    # import pickle
    #
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train_ex)
    #
    # # Training time
    # start = time.time()
    # model.fit(X_train_scaled, y_train_ex)
    # train_time = time.time() - start
    #
    # # Inference time
    # X_test_scaled = scaler.transform(X_test_ex)
    # start = time.time()
    # for _ in range(1000):
    #     _ = model.predict(X_test_scaled[:1])
    # inference_time = (time.time() - start) / 1000
    #
    # # Model size
    # model_bytes = len(pickle.dumps(model))
    # model_size_mb = model_bytes / 1024 / 1024
    #
    # # Accuracy
    # y_pred = model.predict(X_test_scaled)
    # acc = accuracy_score(y_test_ex, y_pred)
    #
    # results_competition['Random Forest'] = {
    #     'accuracy': acc,
    #     'train_time': train_time,
    #     'inference_time_ms': inference_time * 1000,
    #     'model_size_mb': model_size_mb
    # }

    # TODO: Create a summary DataFrame and visualizations
    return X_test_ex, X_train_ex, y_test_ex, y_train_ex


@app.cell
def _(mo):
    mo.md("""
    ### TODO: Write Recommendation Memo

    **To**: Product Manager
    **From**: ML Engineer (You!)
    **Re**: Pokemon Type Classifier - Model Recommendation

    **Executive Summary**:
    (1-2 sentences on recommended model)

    **Models Evaluated**:
    (Table or bullet list of models and key metrics)

    **Recommendation**:
    Model: [Your choice]
    Reasoning:
    - Accuracy: [Why this is sufficient]
    - Speed: [Production requirements]
    - Maintainability: [Team considerations]
    - Trade-offs: [What we're giving up]

    **Next Steps**:
    - [What happens before production]
    - [What monitoring is needed]

    ---

    ## Exercise 3.3: Hyperparameter Tuning Challenge (60 min)

    **Goal**: Optimize a model within constraints.

    **Scenario**: Your model must meet these production requirements:
    - Validation accuracy > 70%
    - Inference time < 10ms per sample
    - Model size < 50MB
    - Training time < 5 minutes

    **Instructions**:
    1. Choose a model type
    2. Find hyperparameters that meet ALL constraints
    3. Use RandomizedSearchCV or GridSearchCV
    4. Document your tuning process
    5. Verify all constraints are met

    **Learning Objective**: Production constraints guide optimization.
    """)
    return


@app.cell
def _():
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler as _StandardScaler
    import time

    # TODO: Define your parameter space
    # Think about which parameters affect:
    # - Accuracy (n_estimators, max_depth, min_samples_split)
    # - Speed (n_estimators, max_depth)
    # - Size (n_estimators, max_depth)

    param_distributions = {
        # TODO: Fill in parameter ranges
        # 'classifier__n_estimators': [10, 50, 100, 200],
        # 'classifier__max_depth': [5, 10, 15, 20],
        # ...
    }

    # TODO: Create pipeline
    # pipeline = Pipeline([
    #     ('scaler', _StandardScaler()),
    #     ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    # ])

    # TODO: Run randomized search
    # print("Running Randomized Search...")
    # search = RandomizedSearchCV(
    #     pipeline,
    #     param_distributions,
    #     n_iter=20,  # Try 20 random combinations
    #     cv=3,
    #     scoring='accuracy',
    #     random_state=42,
    #     n_jobs=-1
    # )
    #
    # start = time.time()
    # search.fit(X_train_ex, y_train_ex)
    # total_train_time = time.time() - start
    #
    # print(f"✅ Search complete in {total_train_time:.1f}s")
    # print(f"   Best CV Score: {search.best_score_:.2%}")
    # print(f"   Best Params: {search.best_params_}")

    # TODO: Verify constraints
    # best_model = search.best_estimator_
    #
    # # Check 1: Accuracy > 70%
    # print(f"\nConstraint Check:")
    # print(f"  ✓ Accuracy: {search.best_score_:.2%} > 70%")
    #
    # # Check 2: Inference time < 10ms
    # X_test_scaled = best_model.named_steps['scaler'].transform(X_test_ex)
    # start = time.time()
    # for _ in range(100):
    #     _ = best_model.predict(X_test_scaled[:1])
    # avg_inference_time = (time.time() - start) / 100 * 1000
    # print(f"  {'✓' if avg_inference_time < 10 else '✗'} Inference: {avg_inference_time:.2f}ms < 10ms")
    #
    # # Check 3: Model size < 50MB
    # import pickle
    # model_size = len(pickle.dumps(best_model)) / 1024 / 1024
    # print(f"  {'✓' if model_size < 50 else '✗'} Size: {model_size:.2f}MB < 50MB")
    #
    # # Check 4: Training time < 5 minutes
    # print(f"  {'✓' if total_train_time < 300 else '✗'} Train time: {total_train_time:.1f}s < 300s")
    return (StandardScaler,)


@app.cell
def _(mo):
    mo.md("""
    ### Reflection Questions:
    1. Which constraints were hardest to meet?
    2. What trade-offs did you make?
    3. How would you handle conflicting constraints?
    4. What would you do if you couldn't meet all constraints?

    ---

    ## 🎯 Module 3 Checkpoint: "Debug the Failing Model"

    **Final Challenge** (60-90 min):

    You're given a trained model that performs poorly. Your job: diagnose and fix it.

    **Scenario**: A colleague trained a model and got these results:
    - Training accuracy: 99%
    - Validation accuracy: 45%
    - Test accuracy: 43%

    **Your Tasks**:
    1. **Diagnose**: What's wrong? (Overfitting? Underfitting? Data issues?)
    2. **Root Cause**: Why did this happen?
    3. **Fix**: Implement a solution
    4. **Verify**: Show improved results
    5. **Document**: Explain your debugging process

    This simulates a real production debugging scenario!
    """)
    return


@app.cell
def _(
    StandardScaler,
    X_test_ex,
    X_train_ex,
    accuracy_score,
    y_test_ex,
    y_train_ex,
):
    # Simulate the "broken" model
    from sklearn.tree import DecisionTreeClassifier

    # This model is intentionally overfit
    broken_scaler = StandardScaler()
    X_train_broken = broken_scaler.fit_transform(X_train_ex)
    X_test_broken = broken_scaler.transform(X_test_ex)

    broken_model = DecisionTreeClassifier(random_state=42)  # No max_depth = overfits!
    broken_model.fit(X_train_broken, y_train_ex)

    train_acc_broken = accuracy_score(y_train_ex, broken_model.predict(X_train_broken))
    test_acc_broken = accuracy_score(y_test_ex, broken_model.predict(X_test_broken))

    print("🚨 Broken Model Performance:")
    print(f"  Training Accuracy: {train_acc_broken:.2%}")
    print(f"  Test Accuracy: {test_acc_broken:.2%}")
    print(f"\n❌ This model has problems! Your task: Fix it.")
    return


@app.cell
def _(mo):
    mo.md("""
    ### TODO: Debug and Fix the Model

    **Step 1: Diagnosis** (15 min)
    - What symptoms do you see?
    - What's the likely problem?
    - How can you verify your hypothesis?

    **Step 2: Investigation** (15 min)
    - Look at the model (tree depth, parameters)
    - Check learning curves
    - Examine feature importance

    **Step 3: Fix** (30 min)
    - Try multiple solutions:
      - Regularization (max_depth, min_samples_split)
      - Different model type
      - Cross-validation
      - More data
      - Different features
    - Compare before/after

    **Step 4: Document** (15 min)
    - Write up your debugging process
    - What worked? What didn't?
    - How did you know when it was fixed?

    ---

    ## 📝 Self-Assessment

    Before moving to Module 4, rate yourself:

    - [ ] I can set up an experiment from scratch in 15 minutes
    - [ ] I understand bias-variance tradeoff with examples
    - [ ] I can implement cross-validation from scratch
    - [ ] I know when to use which model type
    - [ ] I can tune hyperparameters systematically
    - [ ] I can debug overfitting/underfitting
    - [ ] I track experiments consistently

    **If you checked all boxes**: You're ready for Module 4!

    ---

    ## 💡 Key Takeaways

    By now you should understand:

    1. **Systematic experimentation beats random trying**
    2. **Cross-validation gives robust estimates**
    3. **Hyperparameter tuning is constrained optimization**
    4. **Model selection involves trade-offs**
    5. **Debugging is a systematic process**

    **Next Module**: Model Evaluation & Validation

    Now that we can train models, let's learn to evaluate them properly!
    """)
    return


if __name__ == "__main__":
    app.run()
