"""
Module 7: Team Collaboration & ML Code Reviews
==============================================

Professional ML Engineering Onboarding Project

Learning Objectives:
- Use Git effectively for ML projects
- Conduct professional ML code reviews
- Work with existing ML codebases
- Collaborate effectively with team members
- Write clear documentation and PR descriptions

Duration: 2.5 hours

⚠️ CRITICAL: ML is a team sport. Technical skills alone aren't enough.
This module teaches you to work effectively with other engineers.
"""

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md("""
    # Module 7: Team Collaboration & ML Code Reviews

    **"The best code is code that your teammates can understand and maintain"**

    ## Why Collaboration Skills Matter

    **Reality Check**:
    - You'll spend 30% of your time reviewing others' code
    - You'll spend 20% of your time getting your code reviewed
    - You'll spend 15% of your time in meetings coordinating
    - **Only 35% of your time actually writing new code!**

    ## What You'll Learn

    1. **Git for ML**: Branch strategies, commit messages, .gitignore
    2. **Code Reviews**: What to look for in ML code
    3. **PR Best Practices**: Writing descriptions that get approved
    4. **Team Communication**: Async collaboration patterns
    5. **Reading Codebases**: Understanding existing ML systems

    ## Industry Reality

    > "I've seen brilliant ML engineers struggle on teams because they couldn't
    > communicate effectively or work with existing code."
    > — Engineering Manager at Google

    **This module teaches you the "soft skills" that make or break ML careers.**
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 1: Git Workflows for ML Projects

    **ML Projects Have Special Git Needs**

    ### What's Different About ML Code?

    | Regular Software | ML Software |
    |------------------|-------------|
    | Deterministic code | Experiments with randomness |
    | Small code changes | Small code + large data changes |
    | Version code only | Version code + data + models |
    | Clear "done" | Iterative improvements |
    | Fast tests | Slow training/evaluation |

    ### Branch Naming for ML

    **Good Branch Names** (descriptive + context):

    ```bash
    # Feature branches
    feature/add-xgboost-model
    feature/hp-tuning-random-forest
    experiment/try-ensemble-method

    # Bug fixes
    fix/data-leakage-in-features
    fix/preprocessing-null-handling

    # Experiments (short-lived)
    exp/test-new-features
    exp/different-train-split

    # Data changes
    data/update-training-set-2024Q1
    data/fix-schema-validation
    ```

    **Bad Branch Names**:
    - `test` (what test?)
    - `fix` (fix what?)
    - `model2` (what's different from model1?)
    - `johns-branch` (no context)

    ### What to Commit

    **✅ DO Commit**:
    - Code (Python, notebooks, config files)
    - Requirements (requirements.txt, pyproject.toml)
    - Documentation (README, model cards)
    - Test data (small samples for testing)
    - Configuration files
    - Scripts for data processing
    - Experiment tracking logs (MLflow artifacts as metadata)

    **❌ DON'T Commit**:
    - Large datasets (>100MB)
    - Trained models (>10MB)
    - Virtual environments (.venv/)
    - IDE configs (.vscode/, .idea/)
    - Temporary files (__pycache__/, *.pyc)
    - Secrets (.env, credentials.json)
    - Generated files (outputs/, logs/)

    ### The .gitignore for ML Projects

    ```bash
    # Python
    __pycache__/
    *.py[cod]
    *.egg-info/
    .venv/
    venv/

    # ML Specific
    *.pkl
    *.h5
    *.pth
    *.onnx
    models/
    checkpoints/

    # Data
    data/raw/
    data/processed/
    *.csv
    *.parquet
    *.tfrecord

    # Experiments
    mlruns/
    runs/
    logs/
    wandb/

    # Notebooks
    .ipynb_checkpoints/
    *-checkpoint.ipynb

    # Outputs
    outputs/
    figures/
    predictions/

    # IDEs
    .vscode/
    .idea/
    *.swp

    # OS
    .DS_Store
    Thumbs.db

    # Secrets
    .env
    secrets/
    *.key
    credentials.json
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Commit Messages for ML

    **The Formula**: `<type>: <short description>`

    **Types**:
    - `feat:` - New feature or model
    - `fix:` - Bug fix
    - `exp:` - Experiment results
    - `data:` - Data-related changes
    - `refactor:` - Code restructuring
    - `test:` - Add tests
    - `docs:` - Documentation
    - `perf:` - Performance improvement

    **Good Examples**:

    ```bash
    feat: add XGBoost model with hyperparameter tuning
    # Body: Achieved 89% accuracy, 2% improvement over RandomForest

    fix: prevent data leakage in feature engineering
    # Body: Moved scaling inside CV loop, accuracy dropped 87%->85% (more realistic)

    exp: test ensemble of 3 models
    # Body: RF + XGB + LogReg ensemble = 88% (vs 87% single model)
    # Not worth complexity for 1% gain

    data: update training set with 2024 Q1 data
    # Body: Added 5,000 new examples, rebalanced classes

    perf: optimize preprocessing from 5min to 30sec
    # Body: Vectorized operations, removed loop, used polars
    ```

    **Bad Examples**:

    ```bash
    update model  # What update?
    fix bug       # What bug?
    changes       # What changes?
    wip           # Work in progress - don't commit!
    test          # Test what?
    ```

    ### The Commit Message Template

    ```
    <type>: <short summary in 50 chars>

    <Longer explanation if needed:>
    - What changed?
    - Why did it change?
    - What's the impact?

    <Metrics if relevant:>
    - Accuracy: 87% → 89%
    - Latency: 100ms → 45ms
    - Training time: 2h → 30min

    <Breaking changes:>
    - Feature X renamed to Y
    - API endpoint changed
    ```

    **Example**:

    ```
    feat: add confidence thresholding for predictions

    Added ability to return "uncertain" for predictions with
    confidence <0.7. This reduces errors on ambiguous cases.

    Metrics:
    - High confidence (>0.7): 92% accurate
    - Low confidence (<0.7): 68% accurate
    - Now return "uncertain" for low confidence
    - Overall system accuracy: 87% → 91% (on confident predictions)

    Breaking change:
    - API now returns {"prediction": str, "confidence": float, "certain": bool}
    - Previous: {"prediction": str, "confidence": float}
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### 🎓 Socratic Question #1

    **Scenario**: You've been experimenting for 2 weeks. Your branch has:
    - 47 commits
    - Messages like "wip", "test", "update", "fix"
    - 3 different model approaches tried
    - Final model is good, ready to merge

    **Questions**:
    1. Should you merge this as-is?
    2. How would you clean up the commit history?
    3. What information should you preserve?
    4. What should you squash/remove?

    <details>
    <summary>💡 How to Clean Up</summary>

    **No, don't merge as-is!**

    **The Problem**:
    - 47 messy commits pollute main branch history
    - Future developers can't understand the changes
    - Hard to revert if needed
    - Shows lack of professionalism

    **Solution: Interactive Rebase**

    ```bash
    # Squash 47 commits into meaningful ones
    git rebase -i HEAD~47

    # This opens an editor. Change from:
    pick commit1 wip
    pick commit2 test
    pick commit3 update
    ... (44 more)

    # To:
    pick commit1 feat: explore RandomForest baseline
    squash commit2-15 (squash experiments into one)
    pick commit16 feat: implement XGBoost model
    squash commit17-30
    pick commit31 feat: add ensemble method (final)
    squash commit32-47
    ```

    **Result: 3 Clear Commits**:

    ```
    1. feat: explore RandomForest baseline (72% accuracy)
    2. feat: implement XGBoost model (87% accuracy)
    3. feat: add ensemble method - FINAL (89% accuracy)
    ```

    **What to Preserve**:
    - Final approach and reasoning
    - Key metrics (accuracy improvements)
    - Important decisions ("Why ensemble?")

    **What to Remove**:
    - "wip" commits
    - Failed experiments (unless instructive)
    - Debug commits
    - Temporary fixes

    **Better Workflow for Next Time**:
    - Commit meaningfully as you go
    - One commit per logical change
    - Squash locally before pushing
    - Don't push every experiment

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 2: ML Code Reviews

    **What to Look For in ML Code (Different from Regular Code)**

    ### The ML Code Review Checklist

    #### 1. Data Leakage (CRITICAL!)

    ```python
    # ❌ BAD: Data leakage
    # Scaling entire dataset before split
    X_scaled = scaler.fit_transform(X)  # Leakage!
    X_train, X_test = train_test_split(X_scaled, y)

    # ✅ GOOD: Fit on train only
    X_train, X_test = train_test_split(X, y)
    scaler.fit(X_train)  # Fit on train
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

    **Review Question**: "Is the preprocessing fitted on training data only?"

    #### 2. Reproducibility

    ```python
    # ❌ BAD: Not reproducible
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # ✅ GOOD: Reproducible
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    ```

    **Review Question**: "Are random seeds set?"

    #### 3. Train/Val/Test Splits

    ```python
    # ❌ BAD: Using test set for model selection
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    # Choosing based on test performance!
    if model1.score(X_test, y_test) > model2.score(X_test, y_test):
        best_model = model1

    # ✅ GOOD: Using validation set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # Select on validation
    if model1.score(X_val, y_val) > model2.score(X_val, y_val):
        best_model = model1

    # Evaluate on test (once!)
    final_score = best_model.score(X_test, y_test)
    ```

    **Review Question**: "Is the test set only used at the very end?"

    #### 4. Feature Engineering Logic

    ```python
    # ❌ BAD: Using information from the future
    # This leaks target information!
    df['avg_price_by_type'] = df.groupby('type')['price'].transform('mean')

    # ✅ GOOD: Only use features available at prediction time
    # Count is fine (doesn't use target)
    df['count_by_type'] = df.groupby('type')['id'].transform('count')
    ```

    **Review Question**: "Will these features be available at prediction time?"

    #### 5. Error Handling

    ```python
    # ❌ BAD: No input validation
    def predict(card_data):
        features = extract_features(card_data)
        return model.predict(features)[0]

    # ✅ GOOD: Validate inputs
    def predict(card_data):
        # Validate required fields
        required = ['hp', 'attack', 'defense']
        missing = [f for f in required if f not in card_data]
        if missing:
            raise ValueError(f"Missing fields: {missing}")

        # Validate ranges
        if card_data['hp'] < 0 or card_data['hp'] > 300:
            raise ValueError("HP out of valid range")

        features = extract_features(card_data)
        return model.predict(features)[0]
    ```

    **Review Question**: "What happens with invalid inputs?"

    #### 6. Hardcoded Values

    ```python
    # ❌ BAD: Magic numbers
    if confidence > 0.73:  # Why 0.73?
        return prediction

    # ✅ GOOD: Named constants with documentation
    CONFIDENCE_THRESHOLD = 0.73  # Chosen via ROC curve, balances precision/recall
    if confidence > CONFIDENCE_THRESHOLD:
        return prediction
    ```

    **Review Question**: "Are magic numbers explained?"

    #### 7. Model Versioning

    ```python
    # ❌ BAD: No version tracking
    joblib.dump(model, 'model.pkl')

    # ✅ GOOD: Version and metadata
    import joblib
    import json
    from datetime import datetime

    model_version = 'v2.1.0'
    model_path = f'models/pokemon_classifier_{model_version}.pkl'

    # Save model
    joblib.dump(model, model_path)

    # Save metadata
    metadata = {
        'version': model_version,
        'training_date': datetime.now().isoformat(),
        'accuracy': 0.87,
        'features': feature_names,
        'hyperparameters': model.get_params()
    }
    with open(model_path.replace('.pkl', '_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    ```

    **Review Question**: "Can we track which model version is deployed?"
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Common ML Code Smells

    | Code Smell | Why It's Bad | How to Fix |
    |------------|--------------|------------|
    | **Leaky preprocessing** | Uses test data to fit scalers | Fit on train only |
    | **No random seeds** | Results not reproducible | Set random_state everywhere |
    | **Test set peeking** | Overoptimistic metrics | Use validation for model selection |
    | **Magic numbers** | Hard to understand decisions | Named constants with docs |
    | **No error handling** | Crashes on bad input | Validate inputs |
    | **Hardcoded paths** | Breaks on other machines | Use Path, config files |
    | **No type hints** | Unclear what functions expect | Add type hints |
    | **No docstrings** | Functions undocumented | Add docstrings |
    | **Jupyter mess** | Can't run cells out of order | Convert to scripts |
    | **No tests** | Can't verify correctness | Add unit tests |

    ### The Code Review Template

    **Use this when reviewing ML code**:

    ```markdown
    ## Code Review Checklist

    ### Data & Features
    - [ ] No data leakage (preprocessing fitted on train only)
    - [ ] Features available at prediction time
    - [ ] No future information used
    - [ ] Train/val/test splits correct

    ### Reproducibility
    - [ ] Random seeds set
    - [ ] Dependencies versioned (requirements.txt)
    - [ ] Dataset version documented
    - [ ] Results reproducible

    ### Code Quality
    - [ ] Type hints added
    - [ ] Docstrings present
    - [ ] Error handling for edge cases
    - [ ] No hardcoded paths/values
    - [ ] Follows team style guide

    ### Testing
    - [ ] Unit tests added
    - [ ] Edge cases tested
    - [ ] Integration tests for pipeline
    - [ ] Model loading tested

    ### Performance
    - [ ] No obvious inefficiencies
    - [ ] Vectorized operations used
    - [ ] Memory usage reasonable
    - [ ] Training time acceptable

    ### Documentation
    - [ ] README updated
    - [ ] Model card created
    - [ ] API docs updated
    - [ ] Breaking changes noted
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### How to Give Good Code Review Feedback

    **The Formula**: Be kind, be specific, be constructive

    **❌ Bad Feedback**:
    - "This is wrong"
    - "Why did you do it this way?"
    - "This won't work"
    - "LGTM" (with no actual review)

    **✅ Good Feedback**:

    ```markdown
    ## Data Leakage Issue (CRITICAL)

    **Location**: `train.py:45-50`

    **Issue**: The scaler is fitted on the entire dataset before splitting:
    ```python
    X_scaled = scaler.fit_transform(X)  # Line 45
    X_train, X_test = train_test_split(X_scaled, y)  # Line 46
    ```

    **Why this matters**: This leaks test set statistics into training,
    resulting in overoptimistic metrics (test set was used during preprocessing).

    **Suggested fix**:
    ```python
    X_train, X_test = train_test_split(X, y)
    scaler.fit(X_train)  # Fit on train only
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

    **Impact**: This will likely reduce reported accuracy by 2-5%, but the
    new metric will be more realistic.

    **Reference**: See our data leakage guide in docs/best-practices.md
    ```

    **What Makes This Good**:
    - ✅ Clear severity (CRITICAL)
    - ✅ Specific location
    - ✅ Explanation of the problem
    - ✅ Suggested solution (with code!)
    - ✅ Impact assessment
    - ✅ Helpful reference

    ### The Feedback Spectrum

    **Blocking** (must fix before merge):
    - 🚨 Data leakage
    - 🚨 Test set peeking
    - 🚨 Security vulnerabilities
    - 🚨 Breaking changes without discussion

    **Strongly Suggest** (should fix):
    - ⚠️ No random seeds
    - ⚠️ No error handling
    - ⚠️ Magic numbers
    - ⚠️ Poor naming

    **Nit** (nice to have):
    - 💭 Code style inconsistencies
    - 💭 Could be more efficient
    - 💭 Alternative approach suggestion

    **Praise** (do more of this!):
    - ✨ "Nice catch on that edge case!"
    - ✨ "Great docstring - very clear"
    - ✨ "Clever optimization here"

    ### The Sandwich Method

    ```
    1. Start with something positive
    2. Give constructive feedback
    3. End with encouragement

    Example:
    "Great job implementing the XGBoost model! The hyperparameter
    tuning looks thorough.

    I noticed the preprocessing might have data leakage (see comment
    on line 45). This is a super common issue - I made the same mistake
    last month! [Suggested fix]

    Once that's addressed, this will be ready to merge. Looking forward
    to the improved accuracy!"
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### 🎓 Socratic Question #2

    **Scenario**: You're reviewing this code:

    ```python
    def train_model(df):
        # Prepare features
        df['type_encoded'] = df['type'].astype('category').cat.codes
        X = df[['hp', 'attack', 'defense', 'type_encoded']]
        y = df['is_legendary']

        # Train model
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)

        return model
    ```

    **Questions**:
    1. What's wrong with this code? (List at least 3 issues)
    2. How would you provide feedback? (Write actual review comments)
    3. What would you ask the author?
    4. Is this blocking or just a suggestion?

    <details>
    <summary>💡 Issues & Feedback</summary>

    **Issues Found**:

    1. **BLOCKING: Feature leakage**
       - `type_encoded` uses the target variable (legendary status often
         correlates with type)
       - Should this feature even exist?

    2. **BLOCKING: No train/test split**
       - Training on entire dataset
       - No way to evaluate generalization
       - Metrics will be overly optimistic

    3. **HIGH: Not reproducible**
       - No `random_state` set
       - Results will vary each run
       - Can't debug issues

    4. **MEDIUM: Encoding not saved**
       - Category encoding happens inline
       - At prediction time, won't have same encoding
       - New categories will break

    5. **LOW: No validation**
       - What if df is empty?
       - What if columns are missing?
       - No error messages

    **Good Review Feedback**:

    ```markdown
    ## Review Comments

    ### 🚨 CRITICAL: Possible feature leakage
    **Line 3**: `type_encoded` might leak information about legendary status.

    **Question**: Is Pokemon type correlated with legendary status? If yes,
    this feature gives the model an unfair advantage. We need to verify this
    is a legitimate feature that would be available at prediction time.

    **Action**: Can you share a crosstab of type vs legendary status?
    ```python
    pd.crosstab(df['type'], df['is_legendary'])
    ```

    ---

    ### 🚨 BLOCKING: No train/test split
    **Line 10**: Training on entire dataset without holdout.

    **Issue**: We can't evaluate how well this generalizes. The model might
    be memorizing the training data.

    **Suggested fix**:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test accuracy: {score:.2%}")
    ```

    ---

    ### ⚠️ HIGH: Not reproducible
    **Line 9**: Missing `random_state` parameter.

    **Impact**: Results will change every run, making debugging impossible.

    **Fix**:
    ```python
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    ```

    ---

    ### ⚠️ MEDIUM: Encoding won't work at prediction time
    **Line 3**: Category encoding happens inline.

    **Issue**: At prediction time, if we see a new type, this will fail.
    Also, the encoding might be different.

    **Better approach**: Use sklearn's OrdinalEncoder or OneHotEncoder,
    fit it on training data, save it with the model:
    ```python
    from sklearn.preprocessing import OrdinalEncoder

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(df[['type']])
    df['type_encoded'] = encoder.transform(df[['type']])

    # Save encoder with model
    joblib.dump((model, encoder), 'model_and_encoder.pkl')
    ```

    ---

    ### 💭 NIT: Add input validation
    **Suggestion**: Add checks for empty dataframes, missing columns, etc.

    Not blocking, but would make the code more robust.

    ---

    ## Overall Assessment

    The core approach looks good! Once we address the train/test split and
    feature leakage concerns, this should be solid. The encoding issue can
    be fixed as a follow-up if needed.

    **Recommendation**: Needs changes before merge (train/test split is critical)
    ```

    **What to Ask**:
    - "What accuracy are you getting?"
    - "Have you checked for class imbalance?"
    - "Is there a baseline to compare against?"

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 3: Writing PR Descriptions

    **Your PR description is the first thing reviewers see. Make it good!**

    ### The PR Description Template

    ```markdown
    ## Summary
    [One sentence: What does this PR do?]

    ## Motivation
    [Why are we making this change?]

    ## Changes
    - [Bullet point of changes]
    - [Be specific]

    ## Metrics
    **Before**:
    - Accuracy: [X%]
    - Latency: [Xms]

    **After**:
    - Accuracy: [Y%]
    - Latency: [Yms]

    ## Testing
    - [ ] Unit tests added/updated
    - [ ] Integration tests pass
    - [ ] Manually tested with [dataset/scenario]
    - [ ] Model loads correctly
    - [ ] No data leakage verified

    ## Deployment Notes
    [Any special considerations for deployment?]
    - Breaking changes: [Yes/No, describe]
    - Database migrations: [Yes/No]
    - Config changes: [Yes/No]
    - Rollback plan: [How to rollback if needed]

    ## Screenshots/Metrics
    [If applicable, show before/after visualizations]

    ## Checklist
    - [ ] Code follows style guide
    - [ ] Tests added
    - [ ] Documentation updated
    - [ ] Reviewed by [person]
    ```

    ### Real Example

    ```markdown
    ## Summary
    Add XGBoost model with hyperparameter tuning, improving accuracy by 5%.

    ## Motivation
    Our current Random Forest model achieves 82% accuracy. We've hit a plateau
    with feature engineering. XGBoost should capture more complex patterns.

    ## Changes
    - Added `models/xgboost_model.py` with XGBoost implementation
    - Hyperparameter tuning using RandomizedSearchCV (see `train_xgboost.py`)
    - Updated inference API to support XGBoost
    - Added model comparison notebook (`notebooks/model_comparison.ipynb`)

    ## Metrics
    **Before** (Random Forest):
    - Accuracy: 82%
    - Precision: 0.83
    - Recall: 0.81
    - Training time: 5 minutes
    - Inference latency: 45ms

    **After** (XGBoost):
    - Accuracy: 87% ⬆ +5%
    - Precision: 0.88
    - Recall: 0.86
    - Training time: 12 minutes
    - Inference latency: 52ms ⬆ +7ms

    **Tradeoff**: Slightly slower inference (52ms vs 45ms), but well within our
    <100ms SLA. Acceptable given 5% accuracy improvement.

    ## Testing
    - [x] Unit tests for XGBoost wrapper
    - [x] Integration test: model loads and predicts
    - [x] Cross-validation: 5-fold CV shows 87% ± 2%
    - [x] Manually tested on 100 held-out cards: 88% accuracy
    - [x] Verified no data leakage (preprocessing in CV loop)
    - [x] Reproducible: random_state=42, same results across runs

    ## Deployment Notes
    **Breaking Changes**: None. API remains the same.

    **Rollback Plan**: If issues arise, revert to RF model by changing
    `MODEL_TYPE=random_forest` in config. Rollback takes ~5 minutes.

    **Monitoring**: Watch for:
    - Latency (should stay <100ms)
    - Memory usage (XGBoost uses more RAM)
    - Accuracy on production data

    ## Screenshots
    ### Confusion Matrix Comparison
    [Insert image: before/after confusion matrices]

    ### Feature Importance
    [Insert image: XGBoost feature importance]

    ## Checklist
    - [x] Code follows PEP 8
    - [x] Type hints added
    - [x] Docstrings added
    - [x] Tests added (coverage: 87%)
    - [x] README updated
    - [x] Model card created (`models/xgboost_v1_card.md`)

    ---

    **Ready for review!** @ml-team
    ```

    **Why This is Good**:
    - ✅ Clear what changed
    - ✅ Explains why
    - ✅ Shows metrics (with tradeoffs)
    - ✅ Testing is thorough
    - ✅ Deployment considerations covered
    - ✅ Easy to review
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 4: Working with Existing ML Codebases

    **Reading code is harder than writing code**

    ### How to Onboard to an ML Codebase

    **Day 1-3: Understand the System**

    1. **Read the README**
       - What does the system do?
       - How to set up locally?
       - How to run tests?

    2. **Run the System Locally**
       ```bash
       # Follow setup instructions exactly
       python -m venv .venv
       source .venv/bin/activate
       pip install -r requirements.txt

       # Run tests
       pytest

       # Run training pipeline
       python train.py

       # Check if model loads
       python -c "import joblib; model = joblib.load('model.pkl'); print('OK')"
       ```

    3. **Trace a Request End-to-End**
       - API endpoint → preprocessing → model → response
       - Print statements everywhere to understand flow

    **Week 1: Make Small Changes**

    4. **Fix a Typo** (easy confidence builder)
    5. **Add a Test** (learn testing patterns)
    6. **Update Documentation** (learn the system by documenting)
    7. **Fix a Small Bug** (something from backlog)

    **Week 2+: Larger Contributions**

    8. **Add a Feature** (follow existing patterns)
    9. **Refactor Something** (improve what you understand)
    10. **Review Others' PRs** (learn by reviewing)

    ### Reading ML Code Checklist

    **When encountering new ML code, ask**:

    - **Data**: Where does it come from? How is it loaded?
    - **Features**: What features are used? How are they engineered?
    - **Model**: What algorithm? What hyperparameters?
    - **Training**: How is the model trained? What's the split?
    - **Evaluation**: What metrics? How is it validated?
    - **Deployment**: How is it served? What's the API?
    - **Monitoring**: What's tracked? Any alerts?

    ### Common Patterns in ML Codebases

    **Pattern 1: Pipeline Pattern**
    ```python
    # Usually: data → features → model → predictions
    def train_pipeline():
        data = load_data()
        features = engineer_features(data)
        model = train_model(features)
        evaluate_model(model, features)
        save_model(model)
    ```

    **Pattern 2: Config Pattern**
    ```python
    # Separate code from configuration
    config = {
        'model_type': 'xgboost',
        'hyperparameters': {'n_estimators': 100, 'max_depth': 5},
        'features': ['hp', 'attack', 'defense'],
        'random_state': 42
    }
    ```

    **Pattern 3: Experiment Tracking Pattern**
    ```python
    # Log experiments for comparison
    import mlflow

    with mlflow.start_run():
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_metric("accuracy", 0.87)
        mlflow.sklearn.log_model(model, "model")
    ```

    ### How to Safely Modify Existing ML Code

    **The Golden Rules**:

    1. **Don't change what you don't understand**
       - If you're not sure why something exists, ask first
       - There might be a good reason for "weird" code

    2. **Add tests before changing**
       - Write test for current behavior
       - Make your change
       - Verify test still passes (or update it consciously)

    3. **Change one thing at a time**
       - Don't refactor + add features + fix bugs in one PR
       - Makes review easier, rollback safer

    4. **Measure before and after**
       - Run benchmark before your change
       - Run again after
       - Did metrics change? By how much?

    5. **Ask for help early**
       - Don't spend 3 days stuck
       - Ask after 1 hour if truly stuck
       - "I'm trying to do X, I've tried Y, but Z happens. Any ideas?"
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 5: Team Communication Best Practices

    **Async Collaboration in ML Teams**

    ### Slack/Discord Etiquette

    **Good Messages**:

    ```
    📊 Experiment Results - XGBoost vs Random Forest

    TL;DR: XGBoost wins, +5% accuracy, acceptable latency increase

    Metrics:
    - RF:  82% acc, 45ms latency
    - XGB: 87% acc, 52ms latency

    Details: [link to notebook]
    Next: Planning to deploy to staging for A/B test

    Feedback welcome! @ml-team
    ```

    **What makes this good**:
    - ✅ Clear subject
    - ✅ TL;DR first
    - ✅ Key numbers
    - ✅ Links to details
    - ✅ Next steps
    - ✅ Call to action

    **Bad Messages**:

    ```
    "hey can someone look at this"
    [no context, no link, no urgency]

    "model not working"
    [what model? what error? what did you try?]

    "thoughts?"
    [on what? too vague]
    ```

    ### Writing Good Documentation

    **README.md Template**:

    ```markdown
    # Pokemon Type Classifier

    Predicts Pokemon type based on stats. 87% accuracy on test set.

    ## Quick Start

    ```bash
    # Install
    pip install -r requirements.txt

    # Train model
    python train.py

    # Run predictions
    python predict.py --input card.json
    ```

    ## Project Structure

    ```
    ├── data/              # Data files (gitignored)
    ├── models/            # Saved models
    ├── notebooks/         # Exploration notebooks
    ├── src/               # Source code
    │   ├── features.py    # Feature engineering
    │   ├── model.py       # Model definition
    │   └── train.py       # Training script
    ├── tests/             # Unit tests
    └── README.md
    ```

    ## Model Details

    - **Algorithm**: XGBoost
    - **Features**: HP, Attack, Defense, Sp.Atk, Sp.Def, Speed, + engineered
    - **Performance**: 87% accuracy, 88% precision, 86% recall
    - **Latency**: 52ms P95

    See [model card](models/xgboost_v1_card.md) for details.

    ## Development

    ```bash
    # Run tests
    pytest

    # Format code
    black src/

    # Type check
    mypy src/
    ```

    ## Deployment

    See [deployment guide](docs/deployment.md)

    ## Questions?

    - Slack: #ml-team
    - Docs: [link]
    - Owner: @ml-engineer
    ```

    ### Asking for Help

    **The XY Problem**: You ask about your attempted solution (Y) instead of
    your actual problem (X).

    **❌ Bad**:
    ```
    "How do I convert a pandas DataFrame to a numpy array but keep the column names?"
    ```
    (This is the Y - your attempted solution)

    **✅ Good**:
    ```
    "I'm trying to pass features to XGBoost, but it's losing the feature names.
    I need the feature names for feature importance. How should I handle this?

    Current code:
    [code snippet]

    Error:
    [error message]

    What I've tried:
    - Converting to numpy (loses names)
    - DataFrame.values (same issue)
    ```
    (This explains the X - your actual problem)

    ### Meeting Etiquette

    **Before the Meeting**:
    - [ ] Read the agenda
    - [ ] Prepare materials (if presenting)
    - [ ] Review relevant docs/code

    **During the Meeting**:
    - [ ] Be on time
    - [ ] Mute when not speaking
    - [ ] Take notes
    - [ ] Ask questions (don't sit silently confused)
    - [ ] Contribute (you're there for a reason)

    **After the Meeting**:
    - [ ] Follow up on action items
    - [ ] Share notes (if you took them)
    - [ ] Implement decisions
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Module 7 Summary

    ### Key Takeaways

    **Git for ML**:
    - ✅ Use descriptive branch names (feature/add-xgboost)
    - ✅ Write clear commit messages (<type>: <description>)
    - ✅ Don't commit large files (models, data)
    - ✅ Clean up history before merging (squash experiments)

    **Code Reviews**:
    - ✅ Check for data leakage (CRITICAL!)
    - ✅ Verify reproducibility (random seeds)
    - ✅ Look for train/test issues
    - ✅ Give constructive feedback (be kind + specific)
    - ✅ Use the review checklist

    **PR Descriptions**:
    - ✅ Clear summary (what changed)
    - ✅ Motivation (why)
    - ✅ Metrics (before/after)
    - ✅ Testing (what you tested)
    - ✅ Deployment notes

    **Team Collaboration**:
    - ✅ Read existing code carefully
    - ✅ Make small changes first
    - ✅ Communicate clearly (TL;DR first)
    - ✅ Ask for help early
    - ✅ Document your work

    ### Self-Assessment

    You're collaboration-ready when you can:

    - ✅ Use Git professionally for ML projects
    - ✅ Spot data leakage in code reviews
    - ✅ Write clear PR descriptions
    - ✅ Give constructive code review feedback
    - ✅ Onboard to a new codebase in 1 week
    - ✅ Communicate effectively with teammates

    **If you can do all of these, you'll thrive on an ML team!**

    Many brilliant engineers struggle with teamwork. You now have the
    collaboration skills that make the difference between good and great.

    ---

    **Next**: Module 8 - Capstone Project

    Time to put everything together! Build an end-to-end ML system
    (Pokemon card price prediction) demonstrating all your skills.
    """)
    return


if __name__ == "__main__":
    app.run()
