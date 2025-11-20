"""
Module 2 Exercises: EDA & Feature Engineering
=============================================

These exercises reinforce the concepts from Module 2.
Complete each exercise and check your solutions.

Time estimate: 2-3 hours
"""

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        # Module 2 Exercises

        Complete these exercises to solidify your understanding of EDA and feature engineering.

        ## Exercise 2.1: Feature Engineering Challenge (45 min)

        **Goal**: Engineer 5 new features and justify each one.

        **Scenario**: You've been given the Pokemon dataset and told that the current model
        achieves 65% accuracy. Your task is to engineer features that could improve performance.

        **Instructions**:
        1. Load the cleaned Pokemon dataset
        2. Conduct quick EDA to understand the data
        3. Engineer 5 NEW features (not the ones from the main notebook)
        4. For each feature, write a justification:
           - What pattern does it capture?
           - Why might it help classification?
           - What domain knowledge does it encode?
        5. Test if your features correlate with the target

        **Learning Objective**: Practice creative feature engineering with domain knowledge.
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from sklearn.preprocessing import LabelEncoder

    # Load the clean data
    DATA_PATH = Path("data/clean/pokemon_cards_clean_latest.csv")
    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df)} Pokemon cards")
    print(f"Columns: {df.columns.tolist()}")
    return (
        DATA_PATH,
        LabelEncoder,
        Path,
        df,
        np,
        pd,
        plt,
        sns,
    )


@app.cell
def __(df, np, pd):
    # TODO: Engineer 5 new features

    def engineer_creative_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer 5 new creative features.

        TODO: Implement your features here.

        Ideas to consider:
        - Combat effectiveness metrics
        - Type-specific patterns (e.g., glass cannons vs tanks)
        - Rarity-stat relationships
        - Generation trends
        - Price-value metrics
        - Stat variance/consistency
        - Defensive vs offensive archetypes

        Returns:
            DataFrame with new features added
        """
        df_new = df.copy()

        # TODO: Feature 1
        # Justification:
        # df_new['feature_1'] = ...

        # TODO: Feature 2
        # Justification:
        # df_new['feature_2'] = ...

        # TODO: Feature 3
        # Justification:
        # df_new['feature_3'] = ...

        # TODO: Feature 4
        # Justification:
        # df_new['feature_4'] = ...

        # TODO: Feature 5
        # Justification:
        # df_new['feature_5'] = ...

        return df_new

    # TODO: Test your function
    # df_engineered = engineer_creative_features(df)
    # print(f"Added {len(df_engineered.columns) - len(df.columns)} new features")
    return (engineer_creative_features,)


@app.cell
def __(mo):
    mo.md(
        """
        ### Feature Justification Template

        For each feature you create, answer these questions:

        **Feature 1: [Name]**
        - What: (What does this feature compute?)
        - Why: (Why might it help classify Pokemon types?)
        - Domain knowledge: (What Pokemon knowledge informed this?)
        - Expected correlation: (Which types should have high/low values?)

        **Feature 2: [Name]**
        - ...

        (Continue for all 5 features)

        ---

        ## Exercise 2.2: Spot the Leakage (30 min)

        **Goal**: Identify data leakage in provided code examples.

        **Instructions**:
        Below are 5 code snippets. Some have data leakage, some don't.
        For each snippet:
        1. Identify if there's leakage
        2. Explain WHY it's leakage (or why it's not)
        3. If leakage exists, fix it

        **Learning Objective**: Develop intuition for spotting leakage.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### Snippet 1: Feature Scaling

        ```python
        from sklearn.preprocessing import StandardScaler

        # Fit scaler on all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Then split
        X_train, X_test = train_test_split(X_scaled, test_size=0.2)
        ```

        **TODO: Answer these questions:**
        - Is there leakage? (Yes/No)
        - Why or why not?
        - If yes, how would you fix it?
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### Snippet 2: Price Normalization

        ```python
        # Normalize price by the maximum price in the dataset
        df['price_normalized'] = df['price_usd'] / df['price_usd'].max()
        ```

        **TODO: Answer these questions:**
        - Is there leakage? (Yes/No)
        - Why or why not?
        - When would this be problematic?
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### Snippet 3: Type-Based Features

        ```python
        # Create a feature: average HP of this Pokemon's type
        type_mean_hp = df.groupby('type')['hp'].transform('mean')
        df['type_avg_hp'] = type_mean_hp
        ```

        **TODO: Answer these questions:**
        - Is there leakage? (Yes/No)
        - Why or why not?
        - What's the key problem here?
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### Snippet 4: Historical Price Feature

        ```python
        # Add price from last year (for time series data)
        df['price_last_year'] = df.groupby('card_id')['price_usd'].shift(12)
        ```

        **TODO: Answer these questions:**
        - Is there leakage? (Yes/No)
        - Why or why not?
        - What assumption makes this safe or unsafe?
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### Snippet 5: Encoding Categorical Variables

        ```python
        # One-hot encode rarity
        X_train, X_test = train_test_split(X, y, test_size=0.2)

        # Fit encoder on training data only
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoder.fit(X_train[['rarity']])

        # Transform both sets
        X_train_encoded = encoder.transform(X_train[['rarity']])
        X_test_encoded = encoder.transform(X_test[['rarity']])
        ```

        **TODO: Answer these questions:**
        - Is there leakage? (Yes/No)
        - Why or why not?
        - Is this the correct approach?

        ---

        ## Exercise 2.3: Feature Improvement Competition (60 min)

        **Goal**: Beat the baseline model using ONLY feature engineering.

        **The Challenge**:
        - You're given a baseline model with 65% accuracy
        - You can ONLY change features (not the model)
        - Goal: Achieve >75% accuracy

        **Rules**:
        - Same train/test split (provided)
        - Same model (Logistic Regression with default params)
        - Only feature engineering allowed
        - No looking at test set labels!

        **Learning Objective**: Understand that features > models.
        """
    )
    return


@app.cell
def __(df):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    # Fixed train/test split (DO NOT CHANGE)
    X = df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]
    y = df['type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline model
    def train_baseline(X_train, y_train):
        """Train baseline model (DO NOT CHANGE)."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        return model, scaler

    baseline_model, baseline_scaler = train_baseline(X_train, y_train)

    # Baseline performance
    X_train_scaled = baseline_scaler.transform(X_train)
    X_test_scaled = baseline_scaler.transform(X_test)

    train_acc = accuracy_score(y_train, baseline_model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, baseline_model.predict(X_test_scaled))

    print(f"Baseline Performance:")
    print(f"  Training Accuracy: {train_acc:.2%}")
    print(f"  Test Accuracy: {test_acc:.2%}")
    print(f"\n🎯 Your goal: Beat {test_acc:.2%} using better features!")
    return (
        LogisticRegression,
        StandardScaler,
        X,
        X_test,
        X_test_scaled,
        X_train,
        X_train_scaled,
        accuracy_score,
        baseline_model,
        baseline_scaler,
        test_acc,
        train_acc,
        train_baseline,
        train_test_split,
        y,
        y_test,
        y_train,
    )


@app.cell
def __(
    LogisticRegression,
    StandardScaler,
    X_test,
    X_train,
    accuracy_score,
    df,
    pd,
    test_acc,
    y_test,
    y_train,
):
    # TODO: Engineer better features
    def create_improved_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create improved features for the model.

        You have access to ALL columns in df:
        - hp, attack, defense, sp_attack, sp_defense, speed
        - generation, is_legendary, rarity, price_usd

        Rules:
        - DO NOT use 'type' (that's the target!)
        - DO NOT use 'card_id' or 'name'
        - Engineer features that capture patterns

        Returns:
            DataFrame with improved features
        """
        df_improved = df.copy()

        # TODO: Add your engineered features here
        # Examples (but create your own!):
        # df_improved['total_stats'] = df['hp'] + df['attack'] + ...
        # df_improved['attack_defense_ratio'] = df['attack'] / (df['defense'] + 1)
        # ... add more features

        return df_improved

    # TODO: Test your improved features
    # df_improved = create_improved_features(df)

    # # Extract same indices for train/test
    # X_train_improved = df_improved.loc[X_train.index]
    # X_test_improved = df_improved.loc[X_test.index]

    # # Drop non-numeric and target columns
    # drop_cols = ['type', 'card_id', 'name', 'rarity', 'speed_tier', 'gen_group']
    # drop_cols = [col for col in drop_cols if col in X_train_improved.columns]
    # X_train_improved = X_train_improved.drop(columns=drop_cols, errors='ignore')
    # X_test_improved = X_test_improved.drop(columns=drop_cols, errors='ignore')

    # # Train model with improved features (same model!)
    # scaler_improved = StandardScaler()
    # X_train_improved_scaled = scaler_improved.fit_transform(X_train_improved)
    # X_test_improved_scaled = scaler_improved.transform(X_test_improved)

    # model_improved = LogisticRegression(max_iter=1000, random_state=42)
    # model_improved.fit(X_train_improved_scaled, y_train)

    # # Evaluate
    # train_acc_improved = accuracy_score(y_train, model_improved.predict(X_train_improved_scaled))
    # test_acc_improved = accuracy_score(y_test, model_improved.predict(X_test_improved_scaled))

    # print(f"\n✨ Your Improved Model:")
    # print(f"  Training Accuracy: {train_acc_improved:.2%}")
    # print(f"  Test Accuracy: {test_acc_improved:.2%}")
    # print(f"\n📈 Improvement: {(test_acc_improved - test_acc)*100:.1f} percentage points")

    # if test_acc_improved > 0.75:
    #     print("🎉 SUCCESS! You beat 75% accuracy with feature engineering!")
    # elif test_acc_improved > test_acc:
    #     print("👍 Good progress! Try engineering more features to hit 75%.")
    # else:
    #     print("🤔 Hmm, try different features. Think about what distinguishes Pokemon types.")
    return (create_improved_features,)


@app.cell
def __(mo):
    mo.md(
        """
        ### 🤔 Reflection Questions:

        1. **Which features helped most?** Why do you think that is?
        2. **Did any features hurt performance?** Why might that be?
        3. **How did you know what features to try?** (Random vs. hypothesis-driven)
        4. **What's the limit of feature engineering?** Can features alone get to 90%+ accuracy?

        ---

        ## 🎯 Module 2 Checkpoint: "Feature Engineering Competition"

        **Final Challenge** (60-90 min):

        You're joining a Kaggle-style competition. The problem:
        - **Task**: Predict Pokemon type from stats
        - **Metric**: Accuracy on hidden test set
        - **Constraint**: You can only submit 3 times
        - **Your advantage**: Feature engineering skills!

        **Your Deliverables**:
        1. **EDA Report** (5-10 key findings that inform features)
        2. **Feature Engineering Code** (well-documented)
        3. **Feature Importance Analysis** (which features matter?)
        4. **Submission** (predictions on "test set")
        5. **Post-mortem** (what worked, what didn't, what would you try next)

        This simulates a real ML competition / work scenario!
        """
    )
    return


@app.cell
def __(df, train_test_split):
    # Create a "competition" setup
    # We'll hold out a "private test set" that you don't see labels for

    X_comp = df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
                  'generation', 'is_legendary']]
    y_comp = df['type']

    # Split: 60% train, 20% public test (you can see labels), 20% private test (hidden)
    X_train_comp, X_temp, y_train_comp, y_temp = train_test_split(
        X_comp, y_comp, test_size=0.4, random_state=42, stratify=y_comp
    )

    X_public_test, X_private_test, y_public_test, y_private_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("Competition Dataset Split:")
    print(f"  Training: {len(X_train_comp)} samples (you can train on this)")
    print(f"  Public Test: {len(X_public_test)} samples (you can see labels, use for validation)")
    print(f"  Private Test: {len(X_private_test)} samples (labels hidden, use for final submission)")
    return (
        X_comp,
        X_private_test,
        X_public_test,
        X_temp,
        X_train_comp,
        y_comp,
        y_private_test,
        y_public_test,
        y_temp,
        y_train_comp,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ### TODO: Complete the Competition

        **Step 1: EDA (15 min)**
        - Analyze training data
        - Find patterns
        - Document insights

        **Step 2: Feature Engineering (30 min)**
        - Create features based on EDA
        - Test on public test set
        - Iterate

        **Step 3: Final Submission (15 min)**
        - Train on train + public test (optional, but realistic!)
        - Generate predictions for private test
        - Submit (reveal private test accuracy)

        **Step 4: Post-Mortem (15 min)**
        - What worked?
        - What didn't?
        - What would you try with more time?

        ---

        ## 📝 Self-Assessment

        Before moving to Module 3, rate yourself:

        - [ ] I can generate 10+ feature ideas in 5 minutes
        - [ ] I can spot data leakage in code without hints
        - [ ] I understand why features often beat models
        - [ ] I can explain feature engineering decisions to stakeholders
        - [ ] I can build a preprocessing Pipeline from scratch
        - [ ] I know when to use domain knowledge vs. automated feature selection

        **If you checked all boxes**: You're ready for Module 3!

        **If not**: Review the sections you struggled with and try the exercises again.

        ---

        ## 💡 Additional Challenges (Optional)

        If you want more practice:

        1. **Challenge 1**: Engineer features for a regression task (predict price_usd)
        2. **Challenge 2**: Handle categorical features with high cardinality (>100 categories)
        3. **Challenge 3**: Create interaction features (feature crosses)
        4. **Challenge 4**: Automated feature engineering using polynomial features or sklearn's PolynomialFeatures
        5. **Challenge 5**: Build a custom sklearn transformer for your features

        These will prepare you for advanced feature engineering!

        ---

        ## 🎓 Key Takeaways

        By now, you should deeply understand:

        1. **Good features beat complex models** (you just proved it!)
        2. **Domain knowledge is invaluable** (Pokemon knowledge → better features)
        3. **Data leakage is subtle** (always ask "will I have this at prediction time?")
        4. **EDA drives feature engineering** (understand before you engineer)
        5. **Iteration is key** (try, evaluate, improve)

        **Next Module**: Now that we have great features, let's build great models!
        """
    )
    return


if __name__ == "__main__":
    app.run()
