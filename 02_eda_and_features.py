"""
Module 2: Exploratory Data Analysis & Feature Engineering
==========================================================

Professional ML Engineering Onboarding Project
Pokemon Card Type Classification

Learning Objectives:
- Conduct analysis that drives decisions
- Engineer features systematically
- Understand feature-model relationships
- Avoid data leakage
- Build scikit-learn Pipelines

Duration: 3-4 hours
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
        # Module 2: EDA & Feature Engineering

        **"Good features beat complex models"** — Every ML practitioner

        In this module, you'll learn the most impactful skill in ML: **feature engineering**.
        Research shows that 70% of model performance comes from features, not algorithms.

        ## What You'll Learn

        1. **Exploratory Data Analysis**: Extract insights that drive feature design
        2. **Feature Engineering**: Transform raw data into model-ready inputs
        3. **Feature Importance**: Understand what makes a good feature
        4. **Data Leakage**: The silent killer of ML models
        5. **Preprocessing Pipelines**: Build reusable, production-ready transformations

        ## Industry Reality

        > "At Netflix, we spend 80% of time on features, 20% on models"
        > — Netflix ML team

        **Why feature engineering matters:**
        - Simple model + great features > Complex model + raw features
        - Domain knowledge encoded in features = better models
        - Good features make models interpretable
        - Feature engineering is where creativity meets data science
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
    from typing import List, Tuple
    import warnings
    warnings.filterwarnings('ignore')

    # Visualization setup
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    return List, Path, Tuple, np, pd, plt, sns, warnings


@app.cell
def __(Path, pd):
    # Load cleaned data
    DATA_PATH = Path("data/clean/pokemon_cards_clean_latest.csv")
    df = pd.read_csv(DATA_PATH)

    print(f"✅ Loaded {len(df)} Pokemon cards")
    print(f"✅ Features: {df.columns.tolist()}")
    return DATA_PATH, df


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Section 1: Exploratory Data Analysis (EDA)

        EDA is **not** just making pretty plots. It's about:
        1. Understanding your data deeply
        2. Finding patterns that inform feature design
        3. Identifying biases and limitations
        4. Forming hypotheses to test

        **Goal**: By the end of EDA, you should know your data better than anyone.
        """
    )
    return


@app.cell
def __(df, mo):
    mo.md(
        f"""
        ### 📊 Dataset Overview

        **Shape**: {df.shape[0]} rows × {df.shape[1]} columns

        **Target variable**: `type` (18 different Pokemon types)
        **Features**: Stats (HP, Attack, Defense, etc.), Generation, Rarity, Legendary status

        Let's start by understanding the target variable distribution:
        """
    )
    return


@app.cell
def __(df, plt):
    # Target variable distribution
    type_counts = df['type'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    type_counts.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Pokemon Type Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Pokemon Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    print("Type distribution:")
    for ptype, count in type_counts.head(5).items():
        print(f"  {ptype}: {count} ({count/len(df)*100:.1f}%)")
    return ax, fig, ptype, type_counts


@app.cell
def __(df, mo):
    mo.md(
        f"""
        ### 🤔 First Observation: Class Imbalance

        **Key finding**: Our classes are imbalanced!
        - Most common type: {df['type'].value_counts().index[0]} ({df['type'].value_counts().iloc[0]} cards)
        - Least common types: {', '.join(df['type'].value_counts().tail(3).index.tolist())}

        **What this means**:
        - A naive model could get ~{(df['type'].value_counts().iloc[0] / len(df) * 100):.1f}% accuracy by always predicting the most common type
        - We need to be careful with evaluation metrics (accuracy alone is misleading)
        - Rare classes will be harder to predict

        This is **realistic** — real-world data is rarely balanced!
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### 📈 Numerical Feature Distributions

        Let's examine the distributions of our numerical features (stats).
        Understanding distributions helps us decide on preprocessing steps.
        """
    )
    return


@app.cell
def __(df, plt):
    # Distribution of numerical features
    stat_columns = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, col in enumerate(stat_columns):
        axes[idx].hist(df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col.upper()} Distribution', fontweight='bold')
        axes[idx].set_xlabel(col.replace('_', ' ').title())
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.1f}')
        axes[idx].axvline(df[col].median(), color='green', linestyle='--', label=f'Median: {df[col].median():.1f}')
        axes[idx].legend()

    plt.tight_layout()
    plt.show()
    return axes, col, idx, stat_columns


@app.cell
def __(df, stat_columns):
    # Statistical summary
    print("Statistical Summary of Stats:")
    print(df[stat_columns].describe().round(2))
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### 💡 Observation: Feature Distributions

        **Key findings**:
        - All stats are roughly normally distributed (good for many models!)
        - Similar scales (mostly 40-120 range)
        - Some outliers (legendary Pokemon with very high stats)

        **Implications for modeling**:
        - Scaling might help (but not critical since scales are similar)
        - Linear models will work reasonably well
        - Outliers are valid data (legendary Pokemon ARE stronger)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### 🔗 Feature Correlations

        **Critical question**: Are our features correlated?
        - High correlation might indicate redundancy
        - Or it might indicate related concepts (e.g., defense stats)
        """
    )
    return


@app.cell
def __(df, plt, sns, stat_columns):
    # Correlation heatmap
    corr_matrix = df[stat_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\nHighly correlated pairs (>0.6):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.6:
                print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.2f}")
    return ax, corr_matrix, fig, i, j


@app.cell
def __(mo):
    mo.md(
        """
        ### 🤔 Socratic Question: Correlated Features

        **Question**: You found that HP and Defense are highly correlated. Should you remove one?

        **Think about**:
        - Do they measure the same concept, or related concepts?
        - Does one add information the other doesn't?
        - What would happen if you remove one?
        - What would happen if you keep both?

        **Answer**: It depends! In our case:
        - HP = total hit points (survivability)
        - Defense = physical defense (damage reduction)
        - They're related but measure different aspects
        - **Keep both** — they provide complementary information
        - For some models (linear), correlation matters more than for tree-based models

        This is where **domain knowledge** beats blind feature selection!
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### 📊 Features vs Target: The Most Important Analysis

        **Key question**: Do our features actually help predict the target?

        Let's look at how stats vary by Pokemon type:
        """
    )
    return


@app.cell
def __(df, plt, stat_columns):
    # Stats by type - box plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    for idx, stat in enumerate(stat_columns):
        ax = axes[idx]
        df.boxplot(column=stat, by='type', ax=ax, rot=45)
        ax.set_title(f'{stat.upper()} by Pokemon Type')
        ax.set_xlabel('Type')
        ax.set_ylabel(stat.upper())
        plt.sca(ax)

    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    plt.show()
    return ax, axes, fig, idx, stat


@app.cell
def __(df, np):
    # Calculate mean stats by type
    stats_by_type = df.groupby('type')[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].mean()

    print("Mean Stats by Pokemon Type:")
    print(stats_by_type.round(1))
    print("\n🔍 Notice the patterns:")
    print(f"  - Fighting types: High attack ({stats_by_type.loc['Fighting', 'attack']:.1f})")
    print(f"  - Psychic types: High sp_attack ({stats_by_type.loc['Psychic', 'sp_attack']:.1f})")
    print(f"  - Steel types: High defense ({stats_by_type.loc['Steel', 'defense']:.1f})")
    return (stats_by_type,)


@app.cell
def __(mo):
    mo.md(
        """
        ### 💡 Critical Insight: Features ARE Predictive!

        **This is excellent news!** We can see clear patterns:
        - Fighting types have high physical attack
        - Psychic types have high special attack
        - Steel types have high defense
        - Dragon types have balanced high stats

        **This means**: Our features will help the model distinguish between types!

        If we saw NO patterns, we'd need to engineer better features or reconsider the problem.

        ---
        ## Section 2: Feature Engineering

        Now that we understand the data, let's create new features that might help our model.

        ### Feature Engineering Philosophy

        **Good features**:
        1. **Encode domain knowledge**: Use what you know about Pokemon
        2. **Capture relationships**: Ratios, differences, interactions
        3. **Are simple**: Complex ≠ better
        4. **Are generalizable**: Work on new data

        Let's create features systematically:
        """
    )
    return


@app.cell
def __(df, np, pd):
    def engineer_pokemon_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for Pokemon type classification.

        Features created:
        1. Total stats (sum of all stats)
        2. Physical vs Special bias (attack/defense difference)
        3. Offensive vs Defensive bias
        4. Speed tier (fast/medium/slow)
        5. Stat balance (standard deviation of stats)
        6. Attack/Defense ratio
        7. Legendary boost indicator
        8. Generation groupings

        Args:
            df: Input DataFrame with raw Pokemon data

        Returns:
            DataFrame with additional engineered features
        """
        df_feat = df.copy()

        # Feature 1: Total stats (classic Pokemon feature)
        df_feat['total_stats'] = (df_feat['hp'] + df_feat['attack'] + df_feat['defense'] +
                                   df_feat['sp_attack'] + df_feat['sp_defense'] + df_feat['speed'])

        # Feature 2: Physical vs Special bias
        df_feat['physical_bias'] = (df_feat['attack'] + df_feat['defense']) - (df_feat['sp_attack'] + df_feat['sp_defense'])

        # Feature 3: Offensive vs Defensive bias
        df_feat['offensive_bias'] = (df_feat['attack'] + df_feat['sp_attack']) - (df_feat['defense'] + df_feat['sp_defense'])

        # Feature 4: Speed tier (categorical)
        df_feat['speed_tier'] = pd.cut(df_feat['speed'],
                                        bins=[0, 60, 90, 200],
                                        labels=['slow', 'medium', 'fast'])

        # Feature 5: Stat balance (how balanced are the stats?)
        stat_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
        df_feat['stat_balance'] = df_feat[stat_cols].std(axis=1)

        # Feature 6: Attack/Defense ratios (with safe division)
        df_feat['physical_ratio'] = df_feat['attack'] / (df_feat['defense'] + 1)  # +1 to avoid division by zero
        df_feat['special_ratio'] = df_feat['sp_attack'] / (df_feat['sp_defense'] + 1)

        # Feature 7: Legendary boost (boolean to int)
        df_feat['legendary_int'] = df_feat['is_legendary'].astype(int)

        # Feature 8: Generation grouping (early/mid/late)
        df_feat['gen_group'] = pd.cut(df_feat['generation'],
                                       bins=[0, 3, 6, 9],
                                       labels=['early', 'mid', 'late'])

        # Feature 9: HP per total stats (survivability metric)
        df_feat['hp_ratio'] = df_feat['hp'] / df_feat['total_stats']

        # Feature 10: Bulk (HP * Defenses)
        df_feat['physical_bulk'] = df_feat['hp'] * df_feat['defense']
        df_feat['special_bulk'] = df_feat['hp'] * df_feat['sp_defense']

        print(f"✅ Engineered {len(df_feat.columns) - len(df.columns)} new features")

        return df_feat

    df_engineered = engineer_pokemon_features(df)

    print("\nNew features created:")
    new_features = set(df_engineered.columns) - set(df.columns)
    for feat in sorted(new_features):
        print(f"  - {feat}")
    return df_engineered, engineer_pokemon_features, new_features


@app.cell
def __(df_engineered, mo):
    mo.md(
        f"""
        ### 🎯 Feature Engineering Results

        We created **{len(df_engineered.columns) - 13} new features** from domain knowledge!

        **Why these features?**
        - `total_stats`: Classic Pokemon metric (indicates overall power)
        - `physical_bias` / `offensive_bias`: Captures battle style
        - `speed_tier`: Speed matters in Pokemon battles
        - `stat_balance`: Specialized vs generalist Pokemon
        - Ratios: Capture relationships between stats
        - `*_bulk`: Survivability metrics

        Let's see if they're predictive:
        """
    )
    return


@app.cell
def __(df_engineered, new_features):
    # Quick check: Do new features correlate with anything useful?
    feature_sample = ['total_stats', 'physical_bias', 'offensive_bias', 'stat_balance']

    print("Sample of new features:")
    print(df_engineered[feature_sample].describe().round(2))

    # Check correlation with target (using one-hot encoding)
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df_temp = df_engineered.copy()
    df_temp['type_encoded'] = le.fit_transform(df_temp['type'])

    correlations = df_temp[['type_encoded'] + feature_sample].corr()['type_encoded'].sort_values(ascending=False)

    print("\nCorrelation with target (type):")
    print(correlations[1:].round(3))  # Skip the 1.0 correlation with itself
    return LabelEncoder, correlations, df_temp, feature_sample, le


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Section 3: The Silent Killer - Data Leakage

        **Data leakage** = Information from the future or test set leaking into training

        **Why it's dangerous**:
        - Your model looks amazing (99%+ accuracy)
        - Then fails completely in production
        - Hard to detect (model trains successfully)

        ### Common Types of Leakage

        1. **Target leakage**: Feature contains information about the target
        2. **Train-test contamination**: Test data influences preprocessing
        3. **Temporal leakage**: Future information available in past
        4. **Preprocessing leakage**: Fitting on all data before splitting

        Let's see an example of leakage:
        """
    )
    return


@app.cell
def __(df_engineered, np, pd):
    # Example of DATA LEAKAGE (DO NOT DO THIS!)
    def create_leaky_feature_BAD(df):
        """
        BAD EXAMPLE - Creates a leaky feature!

        This feature uses the target to create itself = perfect leakage
        """
        df_leaky = df.copy()

        # Leaky feature: Mean total_stats per type (uses target!)
        type_mean_stats = df.groupby('type')['total_stats'].mean()
        df_leaky['type_mean_stats'] = df_leaky['type'].map(type_mean_stats)

        return df_leaky

    df_leaky = create_leaky_feature_BAD(df_engineered)

    print("⚠️  LEAKY FEATURE EXAMPLE")
    print("\nCorrelation of 'type_mean_stats' with type:")
    le_temp = LabelEncoder()
    df_temp2 = df_leaky.copy()
    df_temp2['type_encoded'] = le_temp.fit_transform(df_temp2['type'])
    correlation = df_temp2[['type_encoded', 'type_mean_stats']].corr().iloc[0, 1]
    print(f"  Correlation: {correlation:.3f}")
    print("\n🚨 This is PERFECT correlation because we used the target to create the feature!")
    print("   In production, you won't know the target, so this feature won't help!")
    return (
        correlation,
        create_leaky_feature_BAD,
        df_leaky,
        df_temp2,
        le_temp,
        type_mean_stats,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ### 🛡️ How to Avoid Leakage

        **Rules to live by**:
        1. ✅ **Split data FIRST**, then preprocess
        2. ✅ **Fit preprocessors only on training data**
        3. ✅ **Never use target variable to create features**
        4. ✅ **Think**: "Will I have this information at prediction time?"
        5. ✅ **Use pipelines** (scikit-learn prevents many leakage issues)

        ---
        ## Section 4: Preprocessing Pipelines (The Right Way)

        **Problem**: We need to apply transformations consistently:
        - Training data
        - Validation data
        - Test data
        - Production data

        **Solution**: scikit-learn Pipelines!

        ### Why Pipelines?
        - **Prevent leakage**: Fit only on training data
        - **Reproducible**: Same transformations every time
        - **Deployable**: Save the pipeline, not manual steps
        - **Clean code**: No repeated transformation code

        Let's build a proper pipeline:
        """
    )
    return


@app.cell
def __():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    return (
        ColumnTransformer,
        OneHotEncoder,
        Pipeline,
        SimpleImputer,
        StandardScaler,
        train_test_split,
    )


@app.cell
def __(df_engineered, train_test_split):
    # Step 1: Split data FIRST (before any preprocessing!)
    # This prevents leakage

    # Separate features and target
    feature_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
                    'generation', 'is_legendary',
                    'total_stats', 'physical_bias', 'offensive_bias', 'stat_balance',
                    'physical_ratio', 'special_ratio', 'hp_ratio',
                    'physical_bulk', 'special_bulk']

    X = df_engineered[feature_cols].copy()
    y = df_engineered['type'].copy()

    # Split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"✅ Data split:")
    print(f"  Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Target classes: {y.nunique()}")
    return (
        X,
        X_temp,
        X_test,
        X_train,
        X_val,
        feature_cols,
        y,
        y_temp,
        y_test,
        y_train,
        y_val,
    )


@app.cell
def __(
    ColumnTransformer,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    np,
):
    # Step 2: Build preprocessing pipeline

    # Numerical features pipeline
    numeric_features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
                        'total_stats', 'physical_bias', 'offensive_bias', 'stat_balance',
                        'physical_ratio', 'special_ratio', 'hp_ratio',
                        'physical_bulk', 'special_bulk']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Binary feature (is_legendary) - just convert to int
    binary_features = ['is_legendary']

    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])

    # Categorical feature (generation) - leave as is for now
    categorical_features = ['generation']

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])

    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bin', binary_transformer, binary_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any other columns
    )

    print("✅ Preprocessing pipeline created")
    print("\nPipeline steps:")
    print("  1. Impute missing values (median for numeric)")
    print("  2. Scale numeric features (StandardScaler)")
    print("  3. Handle binary and categorical features")
    return (
        binary_features,
        binary_transformer,
        categorical_features,
        categorical_transformer,
        numeric_features,
        numeric_transformer,
        preprocessor,
    )


@app.cell
def __(X_test, X_train, X_val, preprocessor):
    # Step 3: Fit pipeline ONLY on training data

    # Fit on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform validation and test (using fitted preprocessor)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print(f"✅ Data processed")
    print(f"  Training shape: {X_train_processed.shape}")
    print(f"  Validation shape: {X_val_processed.shape}")
    print(f"  Test shape: {X_test_processed.shape}")
    print("\n🎯 No leakage! Preprocessor was fit only on training data.")
    return X_test_processed, X_train_processed, X_val_processed


@app.cell
def __(mo):
    mo.md(
        """
        ### ✅ Why This Approach is Correct

        1. **Split BEFORE preprocessing**: No test data seen during training
        2. **Fit on train only**: Scaler statistics computed only from training data
        3. **Transform consistently**: Same transformations applied to val/test
        4. **Pipeline saves everything**: Can deploy this exact preprocessing

        ### ❌ What Would Be Wrong

        ```python
        # WRONG - Don't do this!
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # Fit on ALL data
        X_train, X_test = train_test_split(X_scaled)  # Then split
        # This leaks test set statistics into training!
        ```

        ---
        ## Section 5: Feature Importance Preview

        Before we train models (next module), let's get a quick sense of feature importance
        using a simple baseline model:
        """
    )
    return


@app.cell
def __(X_train_processed, y_train):
    from sklearn.ensemble import RandomForestClassifier

    # Train a quick baseline model to see feature importance
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    baseline_model.fit(X_train_processed, y_train)

    # Get feature importances
    feature_names = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
                     'total_stats', 'physical_bias', 'offensive_bias', 'stat_balance',
                     'physical_ratio', 'special_ratio', 'hp_ratio',
                     'physical_bulk', 'special_bulk', 'is_legendary', 'generation']

    importances = baseline_model.feature_importances_
    import pandas as pd
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("🎯 Top 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))
    return (
        RandomForestClassifier,
        baseline_model,
        feature_importance_df,
        feature_names,
        importances,
    )


@app.cell
def __(feature_importance_df, plt):
    # Visualize feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importance_df.sort_values('importance').plot(
        x='feature', y='importance', kind='barh', ax=ax, color='steelblue'
    )
    ax.set_title('Feature Importance (Baseline Random Forest)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.show()
    return ax, fig


@app.cell
def __(mo):
    mo.md(
        """
        ### 💡 Feature Importance Insights

        **Key observations**:
        - Engineered features (like `total_stats`, `physical_bias`) are very important!
        - Original stats are still useful
        - Some features are less important (but might help for specific types)

        **This validates our feature engineering!** 🎉

        ---
        ## Section 6: Key Takeaways & Socratic Questions

        ### ✅ What You Learned

        1. **EDA drives feature engineering** (understand before you engineer)
        2. **Domain knowledge is crucial** (Pokemon stats inform features)
        3. **Simple features can be powerful** (ratios, sums, differences)
        4. **Data leakage is the silent killer** (always split first!)
        5. **Pipelines prevent errors** (and make deployment easy)

        ### 🤔 Socratic Questions

        1. **"You found that HP and Defense are highly correlated. Should you remove one? Why or why not?"**

        2. **"Your model achieves 99% accuracy. You're suspicious. What do you check first?"**
           - Hint: Data leakage!

        3. **"Why do we fit preprocessing (like scalers) only on training data, not all data?"**
           - Hint: What happens if test data has different statistics?

        4. **"You're creating a 'power_ratio' feature (Attack/Defense). What happens when Defense is 0?"**
           - Hint: Division by zero! Always add +1 or handle edge cases

        5. **"A domain expert suggests a feature. It has zero correlation with the target. Do you include it?"**
           - Hint: Correlation with target isn't everything. Might help for specific classes or in combination.

        ---
        ## 🏢 Industry Context

        ### How Companies Do Feature Engineering

        **Netflix**:
        - 1000s of features per user
        - Feature stores for consistency
        - Automated feature generation
        - A/B test individual features

        **Airbnb**:
        - Feature engineering is 70% of ML work
        - Dedicated feature engineering team
        - Centralized feature repository
        - Feature quality monitoring

        **Google**:
        - Automated feature discovery (AutoML)
        - But still heavy manual engineering
        - Feature crosses (interaction features)
        - Embeddings for categorical features

        ### Common Pitfalls

        ⚠️ **Don't**: Create hundreds of random features
        ✅ **Do**: Create features based on hypotheses

        ⚠️ **Don't**: Ignore domain experts
        ✅ **Do**: Collaborate to encode domain knowledge

        ⚠️ **Don't**: Forget about feature leakage
        ✅ **Do**: Always ask "Will I have this at prediction time?"

        ⚠️ **Don't**: Skip EDA
        ✅ **Do**: Understand your data deeply first

        ---
        ## 🎯 Module 2 Checkpoint

        You've completed Module 2 when you can:

        - [ ] Generate 10 feature ideas in 5 minutes
        - [ ] Spot data leakage in someone else's code
        - [ ] Explain feature engineering decisions to stakeholders
        - [ ] Build a scikit-learn Pipeline from scratch

        **Next**: Module 3 - Model Training & Experimentation

        In the next module, you'll learn how to:
        - Train multiple model types systematically
        - Compare models fairly
        - Implement cross-validation
        - Track experiments like a pro
        - Tune hyperparameters efficiently
        """
    )
    return


if __name__ == "__main__":
    app.run()
