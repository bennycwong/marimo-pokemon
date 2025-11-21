"""
Module 7 Exercises: Team Collaboration & Code Reviews
=====================================================

These exercises practice real teamwork scenarios.
You'll review code, write PRs, and practice collaboration skills.

Time estimate: 2-3 hours
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
    # Module 7 Exercises

    Complete these exercises to master ML team collaboration.

    ## Exercise 7.1: Code Review Practice (60 min)

    **Goal**: Practice reviewing ML code and providing constructive feedback.

    **Instructions**: Review 3 pull requests below. For each:
    1. Identify issues (data leakage, reproducibility, etc.)
    2. Categorize severity (blocking/high/medium/low)
    3. Write constructive feedback
    4. Suggest specific fixes

    **Learning Objective**: Most ML bugs are caught in code review, not testing!
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Pull Request #1: Add Feature Engineering

    **Author**: @junior-ml-engineer
    **Title**: "feat: add new features for better accuracy"

    **Description**:
    ```
    Added some new features. Accuracy improved!
    ```

    **Code Changes**:

    ```python
    # file: src/features.py

    def engineer_features(df):
        '''Add new features to improve model performance.'''

        # Feature 1: Stats total
        df['total_stats'] = df['hp'] + df['attack'] + df['defense']

        # Feature 2: Power ratio
        df['power_ratio'] = df['attack'] / df['defense']

        # Feature 3: Average price by type (NEW!)
        df['avg_price_by_type'] = df.groupby('type')['price'].transform('mean')

        # Feature 4: Rarity score (NEW!)
        # Rare cards are more likely to be legendary
        df['rarity_score'] = df['is_legendary'].astype(int) * 10

        # Feature 5: Normalize HP
        df['hp_normalized'] = (df['hp'] - df['hp'].mean()) / df['hp'].std()

        return df
    ```

    **Testing**: "Manually tested, works fine"

    ---

    **TODO: Your Code Review**

    ```markdown
    ## Issues Found

    ### Issue 1:
    **Severity**: [BLOCKING / HIGH / MEDIUM / LOW]
    **Location**: [Line number or feature name]
    **Problem**: [What's wrong?]
    **Why it matters**: [Impact]
    **Suggested fix**: [Specific solution]

    ### Issue 2:
    [Repeat for each issue]

    ## Overall Assessment
    [Ready to merge / Needs changes / Needs major rework]

    ## Positive Feedback
    [What did they do well?]

    ## Questions for Author
    1. [Question]
    2. [Question]
    ```

    <details>
    <summary>💡 Review Guide</summary>

    **Issues in This Code**:

    1. **🚨 BLOCKING: Data Leakage**
       - **avg_price_by_type**: Uses target variable (price) to create feature!
       - At prediction time, we won't know the price yet
       - This will make accuracy artificially high

    2. **🚨 BLOCKING: Feature Leakage**
       - **rarity_score**: Uses `is_legendary` directly
       - If we're predicting type, using legendary status as a feature leaks information
       - If legendary status is correlated with type, this is cheating

    3. **⚠️ HIGH: Division by Zero**
       - **power_ratio**: What if defense = 0?
       - Will crash with ZeroDivisionError

    4. **⚠️ HIGH: Look-ahead Bias**
       - **hp_normalized**: Uses mean/std of entire dataset
       - Should fit scaler on train set only

    5. **💭 MEDIUM: Poor PR Description**
       - No metrics shown
       - No explanation of features
       - No testing details

    **Good Review**:

    ```markdown
    Thanks for working on feature engineering! I see some interesting ideas here.

    ## 🚨 BLOCKING Issues

    ### Issue 1: Data Leakage in `avg_price_by_type`
    **Line**: 14

    **Problem**: This feature uses the target variable (price) to create a feature.
    At prediction time, we won't have the price yet!

    **Impact**: This will make your accuracy artificially high. When deployed,
    the model won't work because this feature isn't available.

    **Fix**: Remove this feature, or use something that's available at prediction
    time (like `count_by_type`):
    ```python
    df['count_by_type'] = df.groupby('type').transform('count')
    ```

    ---

    ### Issue 2: Feature Leakage in `rarity_score`
    **Line**: 18

    **Problem**: Using `is_legendary` as a feature when predicting type is
    problematic if legendary status correlates with type.

    **Questions**:
    - What are we predicting? Type or legendary status?
    - Is legendary status available at prediction time?
    - Have you checked if it's correlated with type?

    **Action**: Please clarify the use case and check correlation.

    ---

    ## ⚠️ HIGH Priority

    ### Issue 3: Division by Zero
    **Line**: 11

    **Problem**: If defense = 0, this will crash.

    **Fix**:
    ```python
    df['power_ratio'] = df['attack'] / (df['defense'] + 1)  # +1 prevents div by 0
    ```

    ---

    ### Issue 4: Normalization Leakage
    **Line**: 21

    **Problem**: Using mean/std of entire dataset. Should fit on train only.

    **Better approach**: Use sklearn.preprocessing.StandardScaler in your
    training pipeline, fit on train set only.

    ---

    ## 💭 Suggestions

    ### Improve PR Description
    Could you add:
    - What accuracy improvement did you see?
    - Why did you choose these features?
    - What testing did you do?

    ---

    ## ✨ What You Did Well

    - `total_stats` is a good feature!
    - `power_ratio` is creative (just need to handle div by 0)
    - Good variety of feature types

    ---

    ## Recommendation

    **Needs changes before merge** (blocking issues with data leakage).

    Once fixed, I think the valid features (total_stats, power_ratio) will
    be good additions!
    ```

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Pull Request #2: Improve Model Performance

    **Author**: @mid-level-ml-engineer
    **Title**: "perf: switch to XGBoost for better accuracy"

    **Description**:
    ```
    Replaced RandomForest with XGBoost

    **Metrics**:
    - Before: 82% accuracy
    - After: 91% accuracy (+9%!)

    **Testing**: Looks good on test set
    ```

    **Code Changes**:

    ```python
    # file: src/train.py

    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    def train_model():
        # Load data
        df = load_pokemon_data()

        # Features
        X = df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]
        y = df['type']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Try multiple models
        models = {
            'rf': RandomForestClassifier(n_estimators=100),
            'xgb': xgb.XGBClassifier(n_estimators=100, max_depth=6)
        }

        best_model = None
        best_score = 0

        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"{name}: {score:.2%}")

            if score > best_score:
                best_score = score
                best_model = model

        print(f"Best model: {best_score:.2%}")
        return best_model

    # Result:
    # rf: 82.00%
    # xgb: 91.00%  ← MUCH BETTER!
    # Best model: 91.00%
    ```

    ---

    **TODO: Your Code Review**

    ```markdown
    ## Issues Found

    ### Issue 1:
    [Your analysis]

    ### Issue 2:
    [Your analysis]

    ## Overall Assessment
    [Your verdict]
    ```

    <details>
    <summary>💡 Review Guide</summary>

    **Issues in This Code**:

    1. **🚨 CRITICAL: Test Set Selection Bias**
       - Choosing best model based on TEST set performance!
       - Should use validation set
       - Test set should only be used once at the very end
       - This 91% is overly optimistic

    2. **⚠️ HIGH: Not Reproducible**
       - No random_state in train_test_split
       - No random_state in models
       - Results will vary every run

    3. **⚠️ MEDIUM: Single Split**
       - Should use cross-validation
       - Single split could be lucky
       - 91% might not be real performance

    4. **💭 LOW: No stratification**
       - Should stratify on target (y)
       - Ensures balanced classes in train/test

    **The Big Problem**: This looks like a 9% improvement, but it's likely much less
    once we fix the methodology!

    **Good Review**:

    ```markdown
    Great initiative on exploring XGBoost! However, I'm concerned about the
    evaluation methodology. The 91% number is likely overoptimistic.

    ## 🚨 CRITICAL: Test Set Peeking

    **Line**: 30 - Model selection based on test set

    **Problem**: You're selecting the best model by evaluating on the test set.
    This is a form of "test set peeking" - we've effectively trained on the test
    set by using it to make decisions.

    **Why this matters**: The test set should be a "held-out future" that we
    only look at once. By selecting the model that performs best on it, we've
    optimized for that specific test set. The real performance is likely lower.

    **Impact**: Your 91% accuracy is probably overestimated by 2-5%. The true
    performance might be closer to 86-89%.

    **Fix**: Use a validation set for model selection:

    ```python
    # Split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Train on train, select on validation
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)  # Use validation!
        print(f"{name} val: {val_score:.2%}")

        if val_score > best_score:
            best_score = val_score
            best_model = model

    # Finally, evaluate on test (once!)
    test_score = best_model.score(X_test, y_test)
    print(f"Final test score: {test_score:.2%}")
    ```

    ---

    ## ⚠️ HIGH: Not Reproducible

    **Lines**: 15, 18, 19

    **Problem**: No random seeds set.

    **Fix**:
    ```python
    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # Add this!
    )

    # Models
    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
    'xgb': xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
    ```

    ---

    ## 💭 Suggestion: Use Cross-Validation

    Single train/test split can be misleading. Consider:

    ```python
    from sklearn.model_selection import cross_val_score

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{name}: {scores.mean():.2%} ± {scores.std():.2%}")
    ```

    This gives you a more robust estimate.

    ---

    ## Recommendation

    **Needs changes** - The methodology needs fixing before we can trust the metrics.

    Once fixed, I expect the performance gain to be smaller (maybe 84-87%
    instead of 91%), but if XGBoost is still better, we should definitely use it!

    Could you re-run with the validation set approach and share the updated metrics?
    ```

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Pull Request #3: Add Input Validation

    **Author**: @senior-ml-engineer
    **Title**: "feat: add comprehensive input validation for production"

    **Description**:
    ```
    Adds input validation to prevent bad requests from crashing the API.

    **Changes**:
    - Validate required fields
    - Check data types
    - Verify value ranges
    - Return clear error messages

    **Testing**:
    - Unit tests added
    - Tested with 50 malformed requests
    - All handled gracefully
    ```

    **Code Changes**:

    ```python
    # file: src/api.py

    from typing import Dict, Any
    from pydantic import BaseModel, Field, validator

    class PokemonCardInput(BaseModel):
        '''Schema for Pokemon card input validation.'''

        hp: int = Field(..., ge=1, le=300, description="Hit points")
        attack: int = Field(..., ge=1, le=300, description="Attack stat")
        defense: int = Field(..., ge=1, le=300, description="Defense stat")
        sp_attack: int = Field(..., ge=1, le=300, description="Special attack")
        sp_defense: int = Field(..., ge=1, le=300, description="Special defense")
        speed: int = Field(..., ge=1, le=300, description="Speed")

        @validator('*')
        def check_not_negative(cls, v):
            if isinstance(v, (int, float)) and v < 0:
                raise ValueError("Stats cannot be negative")
            return v

        class Config:
            schema_extra = {
                "example": {
                    "hp": 90,
                    "attack": 85,
                    "defense": 75,
                    "sp_attack": 110,
                    "sp_defense": 95,
                    "speed": 80
                }
            }

    def predict(card_data: Dict[str, Any]) -> Dict[str, Any]:
        '''Make prediction with input validation.'''
        try:
            # Validate input
            validated_input = PokemonCardInput(**card_data)

            # Extract features
            features = [[
                validated_input.hp,
                validated_input.attack,
                validated_input.defense,
                validated_input.sp_attack,
                validated_input.sp_defense,
                validated_input.speed
            ]]

            # Predict
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            confidence = float(max(proba))

            return {
                "success": True,
                "prediction": prediction,
                "confidence": confidence,
                "certain": confidence > 0.7
            }

        except ValueError as e:
            return {
                "success": False,
                "error": "Invalid input",
                "details": str(e)
            }
        except Exception as e:
            # Log error
            logger.error(f"Prediction error: {e}")
            return {
                "success": False,
                "error": "Internal error",
                "details": "Please contact support"
            }
    ```

    ---

    **TODO: Your Code Review**

    ```markdown
    ## Review

    ### What I Like:
    [Positive feedback]

    ### Suggestions:
    [Any improvements?]

    ### Questions:
    [Anything unclear?]

    ## Verdict:
    [Approve / Request changes]
    ```

    <details>
    <summary>💡 Review Guide</summary>

    **This is GOOD code!**

    Things done well:
    - ✅ Uses Pydantic for validation (industry standard)
    - ✅ Clear error messages
    - ✅ Type hints throughout
    - ✅ Handles exceptions gracefully
    - ✅ Returns structured responses
    - ✅ Example provided in schema
    - ✅ Logs errors
    - ✅ Tests added

    **Minor Suggestions**:

    1. The `@validator('*')` is redundant since `Field(ge=1)` already prevents negatives
    2. Could add a custom error message for 400 vs 500 errors
    3. Might want to validate feature order matches model training

    **Good Review**:

    ```markdown
    Excellent work on input validation! This is exactly what we need for production.

    ## ✨ What I Really Like

    1. **Pydantic usage**: Perfect choice for API validation
    2. **Clear error messages**: Users will know what's wrong
    3. **Comprehensive**: Covers all the edge cases
    4. **Testing**: Great that you tested with malformed inputs
    5. **Documentation**: The example in `schema_extra` is helpful

    ## 💭 Minor Suggestions

    ### 1. Redundant Validator

    **Line 16**: The `@validator('*')` check for negatives is redundant since
    `Field(ge=1)` already enforces this.

    **Suggestion**: Remove the validator, or use it for additional business
    logic checks:

    ```python
    @validator('speed')
    def speed_realistic(cls, v, values):
        # Check that speed isn't impossibly high for low HP
        if 'hp' in values and v > 200 and values['hp'] < 50:
            raise ValueError("Speed too high for low HP Pokemon")
        return v
    ```

    ### 2. Error Response Codes

    **Line 65**: Could differentiate between client errors (400) and server errors (500):

    ```python
    except ValueError as e:
        # 400 - Client error
        return {"success": False, "error_code": 400, "error": str(e)}
    except Exception as e:
        # 500 - Server error
        logger.error(f"Prediction error: {e}")
        return {"success": False, "error_code": 500, "error": "Internal error"}
    ```

    This helps with monitoring (track 400s vs 500s separately).

    ### 3. Feature Order Verification

    Optional: Add a check that features are in the right order:

    ```python
    EXPECTED_FEATURES = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

    def validate_feature_order(features):
        assert features == EXPECTED_FEATURES, "Feature order mismatch"
    ```

    ## Questions

    1. **Performance**: Pydantic validation adds ~1-2ms. Is this acceptable for
       our latency SLA? (I assume yes, but good to confirm)

    2. **Edge cases**: What about Pokemon with 0 stats (e.g., Shedinja has 1 HP)?
       Should we allow 0, or is `ge=1` intentional?

    3. **API versioning**: Should the response schema include a version field
       for future compatibility?

    ## Recommendation

    **✅ APPROVE** - Ready to merge!

    The suggestions are optional nice-to-haves. This code significantly improves
    our API robustness. Great work!

    Once merged, let's monitor error rates to see what edge cases users hit.
    ```

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Exercise 7.2: Write a PR Description (30 min)

    **Goal**: Practice writing clear PR descriptions.

    **Scenario**: You've just finished implementing the Pokemon type classifier
    using the XGBoost model (Module 3). Now you need to create a PR.

    **Your Changes**:
    - Implemented XGBoost model
    - Accuracy: 87% (vs 82% baseline RandomForest)
    - Hyperparameter tuned using RandomizedSearchCV
    - Added unit tests
    - Updated model card

    **TODO: Write the PR description**
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ```markdown
    ## TODO: Complete Your PR Description

    ## Summary
    [One sentence - what does this PR do?]

    ## Motivation
    [Why are we making this change?]
    [What problem does it solve?]

    ## Changes
    - [Change 1]
    - [Change 2]
    - [Change 3]

    ## Metrics

    **Before** (RandomForest baseline):
    - Accuracy: [X%]
    - Precision: [X.XX]
    - Recall: [X.XX]
    - F1: [X.XX]
    - Training time: [X minutes]
    - Inference latency: [Xms]

    **After** (XGBoost):
    - Accuracy: [Y%] ⬆/⬇ [+X%]
    - Precision: [Y.YY]
    - Recall: [Y.YY]
    - F1: [Y.YY]
    - Training time: [Y minutes]
    - Inference latency: [Yms]

    **Analysis**: [Explain tradeoffs]

    ## Testing
    - [ ] Unit tests added/updated
    - [ ] Integration tests pass
    - [ ] Cross-validation performed ([X]-fold)
    - [ ] Manually tested on [X] examples
    - [ ] Checked for data leakage
    - [ ] Verified reproducibility

    ## Hyperparameters
    [List the final hyperparameters you chose and why]

    ## Deployment Notes

    **Breaking Changes**: [Yes/No, explain]

    **Backward Compatibility**: [Yes/No]

    **Rollback Plan**: [How to rollback if issues]

    **Monitoring**: [What to watch for]

    ## Screenshots/Visualizations
    [If applicable: confusion matrix, feature importance, learning curves]

    ## Checklist
    - [ ] Code follows style guide
    - [ ] Type hints added
    - [ ] Docstrings added
    - [ ] Tests added (coverage: [X%])
    - [ ] Documentation updated
    - [ ] Model card updated

    ## Questions for Reviewers
    [Any specific feedback you want?]

    ---

    **Ready for review!** @ml-team
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Exercise 7.3: Clean Up Git History (20 min)

    **Goal**: Practice organizing commits before merging.

    **Scenario**: You've been experimenting for a week. Your branch has this history:

    ```
    * wip
    * fix typo
    * update
    * trying different hyperparameters
    * this doesn't work
    * reverting
    * maybe this works?
    * fix
    * add xgboost
    * clean up
    * test
    * final version
    ```

    **TODO: Plan how to clean this up**

    ```markdown
    ## Goal: 3-4 Meaningful Commits

    ### Commit 1:
    **Message**: [Write a good commit message]
    **Includes**: [Which of the 12 commits should be squashed into this?]
    **Why**: [Reasoning]

    ### Commit 2:
    **Message**: [Write a good commit message]
    **Includes**: [Which commits?]
    **Why**: [Reasoning]

    ### Commit 3:
    **Message**: [Write a good commit message]
    **Includes**: [Which commits?]
    **Why**: [Reasoning]

    ## Git Commands to Execute

    ```bash
    # TODO: Write the git commands you would use
    # git rebase -i HEAD~12
    # ...
    ```

    ## Why This Matters
    [Explain why clean history is important]
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Exercise 7.4: Onboarding to a Codebase (30 min)

    **Goal**: Practice systematically understanding new code.

    **Scenario**: You just joined a team. Their ML system has this structure:

    ```
    pokemon-classifier/
    ├── README.md
    ├── requirements.txt
    ├── config/
    │   └── model_config.yaml
    ├── data/
    │   └── processor.py
    ├── models/
    │   ├── base_model.py
    │   ├── random_forest.py
    │   └── xgboost_model.py
    ├── api/
    │   ├── app.py
    │   └── schemas.py
    ├── tests/
    │   └── test_model.py
    └── train.py
    ```

    **TODO: Plan your onboarding**

    ```markdown
    ## Day 1: Orientation

    ### Step 1: Read Documentation (30 min)
    [ ] Read README.md
    [ ] Understand: What does this system do?
    [ ] Understand: How is it deployed?
    [ ] Understand: What's the tech stack?

    **Questions to ask the team**:
    1. [Your question]
    2. [Your question]

    ### Step 2: Setup Locally (1 hour)
    [ ] Follow README setup instructions
    [ ] Run tests: `pytest`
    [ ] Train model: `python train.py`
    [ ] Start API: `python api/app.py`
    [ ] Make a test prediction

    **If stuck, ask**: [Who to ask? What to ask?]

    ### Step 3: Trace a Prediction (1 hour)
    [ ] Add print statements to follow request flow
    [ ] API endpoint → data processing → model → response

    **Document the flow**:
    ```
    1. Request comes in: [which file, which function?]
    2. Data validated: [where?]
    3. Features extracted: [where?]
    4. Model predicts: [where?]
    5. Response formatted: [where?]
    ```

    ## Day 2-3: Small Contributions

    ### Easy Wins
    [ ] Fix a typo in documentation
    [ ] Add a docstring to an undocumented function
    [ ] Add a test case
    [ ] Improve error message

    **Goal**: Build confidence with small, safe changes

    ## Week 1: Understanding the Code

    ### Questions to Answer
    1. **Data**: Where does training data come from?
    2. **Features**: What features are used?
    3. **Model**: What algorithm? What hyperparameters?
    4. **Training**: How often is model retrained?
    5. **Deployment**: How is it deployed?
    6. **Monitoring**: What's monitored?

    ### Create Your Onboarding Doc
    [Write notes as you learn the system]
    [This will help the next person who joins!]

    ## Week 2: Larger Contribution

    ### Pick a Starter Task
    [ ] Find a "good first issue" in backlog
    [ ] Ask for task recommendations
    [ ] Estimate how long it will take
    [ ] Implement it
    [ ] Get code reviewed
    [ ] Celebrate your first real contribution!

    ## Onboarding Success Checklist

    After 2 weeks, you should be able to:
    - [ ] Run the system locally
    - [ ] Explain what it does (to a non-engineer)
    - [ ] Trace a request end-to-end
    - [ ] Make small code changes confidently
    - [ ] Know who to ask for help
    - [ ] Have shipped at least one PR
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Bonus Exercise 7.5: Team Communication Practice (Optional - 20 min)

    **Goal**: Practice writing clear technical communications.

    **Scenarios**: Write the message you would send for each situation.

    ### Scenario A: Sharing Experiment Results

    You tested 3 models: RandomForest (82%), XGBoost (87%), LightGBM (88%).
    XGBoost is fastest. LightGBM is slightly better but takes 3x longer to train.

    **TODO: Write a Slack message sharing your results**

    ```
    [Your message here]
    ```

    ---

    ### Scenario B: Asking for Help

    You've been stuck for 2 hours. Your XGBoost model is giving you this error:
    "ValueError: feature names mismatch". You've tried googling and checking
    the feature order.

    **TODO: Write a message asking for help**

    ```
    [Your message here]
    ```

    ---

    ### Scenario C: Proposing a Change

    You found that the preprocessing pipeline has a performance bottleneck.
    It takes 5 minutes to process 10k cards. You think you can optimize it
    to 30 seconds using vectorization.

    **TODO: Write a proposal message**

    ```
    [Your message here]
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Self-Assessment

    After completing these exercises, you should be able to:

    - ✅ Review ML code and spot common issues (data leakage, etc.)
    - ✅ Give constructive, specific feedback
    - ✅ Write clear PR descriptions with metrics
    - ✅ Clean up git history before merging
    - ✅ Onboard to a new codebase systematically
    - ✅ Communicate effectively with teammates

    **If you can do all of these, you're ready to collaborate on an ML team!**

    Many technically skilled engineers struggle with teamwork and communication.
    You now have the collaboration skills that make you a valuable team member.

    ---

    **Next**: Module 8 - Capstone Project

    Put everything together! Build an end-to-end Pokemon card price predictor
    demonstrating all your technical and collaboration skills.
    """)
    return


if __name__ == "__main__":
    app.run()
