"""
Module 4: Model Evaluation & Validation
========================================

Professional ML Engineering Onboarding Project
Pokemon Card Type Classification

Learning Objectives:
- Evaluate beyond accuracy
- Understand model failure modes
- Communicate model quality
- Create model cards
- Set confidence thresholds

Duration: 2-3 hours
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
    # Module 4: Model Evaluation & Validation

    **"All models are wrong, but some are useful"** — George Box

    You've trained models. Now the critical question: **Are they good enough for production?**

    ## What You'll Learn

    1. **Metrics Beyond Accuracy**: Precision, recall, F1, ROC-AUC
    2. **Error Analysis**: Understand where and why models fail
    3. **Confusion Matrices**: Detailed performance breakdown
    4. **Model Cards**: Document limitations and intended use
    5. **Production Readiness**: Is this model deployable?

    ## Why This Matters

    > "95% accuracy sounds great until you realize it's a cancer detector"
    > — Every ML practitioner

    **Real examples of metric failures**:
    - Amazon hiring AI (high accuracy, biased against women)
    - Credit scoring (high AUC, discriminatory patterns)
    - Medical diagnosis (high accuracy on common cases, fails on rare diseases)

    Proper evaluation prevents disasters!
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore')

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    return Path, np, pd, plt, sns


@app.cell
def _(Path, pd):
    # Load data and train a model (from Module 3)
    DATA_PATH = Path("data/clean/pokemon_cards_clean_latest.csv")
    df = pd.read_csv(DATA_PATH)

    # Feature engineering
    df['total_stats'] = df['hp'] + df['attack'] + df['defense'] + df['sp_attack'] + df['sp_defense'] + df['speed']
    df['physical_bias'] = (df['attack'] + df['defense']) - (df['sp_attack'] + df['sp_defense'])

    feature_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
                    'total_stats', 'physical_bias', 'generation', 'is_legendary']
    X = df[feature_cols]
    y = df['type']

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    print(f"✅ Model trained on {len(X_train)} samples")
    print(f"✅ Test set: {len(X_test)} samples")
    return X_test, y_pred, y_pred_proba, y_test


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section 1: Beyond Accuracy

    **Question**: Your model is 95% accurate on spam detection. Is that good?

    **Answer**: It depends! If only 2% of emails are spam:
    - A dummy model (always predict "not spam") achieves 98% accuracy
    - Your 95% model is actually WORSE!

    **This is why we need better metrics.**

    ### Key Metrics for Multi-Class Classification

    1. **Accuracy**: Correct predictions / Total predictions (misleading with imbalance!)
    2. **Precision**: Of predicted positives, how many are correct?
    3. **Recall**: Of actual positives, how many did we find?
    4. **F1-Score**: Harmonic mean of precision and recall
    5. **Per-Class Metrics**: Performance on each individual class

    Let's compute all of these:
    """)
    return


@app.cell
def _(y_pred, y_test):
    from sklearn.metrics import classification_report, accuracy_score

    # Overall accuracy
    overall_acc = accuracy_score(y_test, y_pred)

    print(f"Overall Accuracy: {overall_acc:.2%}\n")

    # Detailed classification report
    print("Classification Report (Per-Class Metrics):")
    print("=" * 70)
    report = classification_report(y_test, y_pred)
    print(report)
    return (overall_acc,)


@app.cell
def _(mo):
    mo.md("""
    ### 💡 Understanding the Classification Report

    **For each class (Pokemon type)**:
    - **Precision**: When we predict this type, how often are we correct?
      - Example: Precision=0.80 for Fire → 80% of predicted Fire types are actually Fire
    - **Recall**: Of all actual instances of this type, how many did we find?
      - Example: Recall=0.70 for Fire → We found 70% of all Fire types
    - **F1-Score**: Balance between precision and recall
      - High F1 = Good at both finding and correctly identifying

    **Support**: Number of actual instances of each class in test set

    **Why this matters**:
    - Some types are easy to classify (high precision & recall)
    - Some types are confused with others (low recall)
    - Rare types might have low support (less confident in metrics)

    ---
    ## Section 2: Confusion Matrix (The Most Useful Visualization)

    **Confusion Matrix** shows which classes are confused with each other.

    This is GOLD for understanding model failures!
    """)
    return


@app.cell
def _(np, plt, sns, y_pred, y_test):
    from sklearn.metrics import confusion_matrix

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = sorted(y_test.unique())

    # Visualize
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix - Pokemon Type Classification', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Find most confused pairs
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(cm_normalized, 0)  # Ignore correct predictions

    most_confused_idx = np.unravel_index(cm_normalized.argmax(), cm_normalized.shape)
    most_confused_true = class_names[most_confused_idx[0]]
    most_confused_pred = class_names[most_confused_idx[1]]
    confusion_rate = cm_normalized[most_confused_idx]

    print(f"\n🔍 Most Confused Pair:")
    print(f"   {most_confused_true} → {most_confused_pred}: {confusion_rate:.1%} of {most_confused_true} predicted as {most_confused_pred}")
    return


@app.cell
def _(mo):
    mo.md("""
    ### 💡 Insights from Confusion Matrix

    **Diagonal elements**: Correct predictions (darker = better)
    **Off-diagonal elements**: Misclassifications (should be light)

    **What to look for**:
    - Which types are most often confused?
    - Are there systematic patterns? (e.g., physical types confused together)
    - Do rare types have more errors? (less training data)

    **Actions you can take**:
    - Engineer features to distinguish confused pairs
    - Collect more data for underperforming classes
    - Consider hierarchical classification
    - Adjust class weights

    ---
    ## Section 3: Error Analysis (The Most Important Step)

    **Key insight**: Understanding WHERE your model fails is more valuable than overall accuracy!

    Let's analyze errors systematically:
    """)
    return


@app.cell
def _(X_test, y_pred, y_test):
    # Create error analysis dataframe
    error_df = X_test.copy()
    error_df['true_type'] = y_test.values
    error_df['pred_type'] = y_pred
    error_df['is_correct'] = (y_test.values == y_pred)

    # Analyze errors
    errors = error_df[~error_df['is_correct']]

    print(f"Total Errors: {len(errors)} out of {len(error_df)} ({len(errors)/len(error_df)*100:.1f}%)\n")

    # Error breakdown by true class
    print("Errors by True Class:")
    error_by_class = errors.groupby('true_type').size().sort_values(ascending=False)
    print(error_by_class.head(10))

    # Most common error types
    print("\nMost Common Misclassifications:")
    error_pairs = errors.groupby(['true_type', 'pred_type']).size().sort_values(ascending=False)
    print(error_pairs.head(10))
    return error_df, errors


@app.cell
def _(error_df, errors, pd):
    # Statistical analysis of errors
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS: Statistical Patterns")
    print("=" * 70)

    # Do errors have different stat patterns?
    print("\nAverage Stats - Errors vs Correct:")
    stat_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'total_stats']

    correct_stats = error_df[error_df['is_correct']][stat_cols].mean()
    error_stats = errors[stat_cols].mean()

    comparison = pd.DataFrame({
        'Correct': correct_stats,
        'Errors': error_stats,
        'Difference': error_stats - correct_stats
    })
    print(comparison.round(1))

    print("\n💡 Interpretation:")
    print("   - Errors tend to occur on Pokemon with [analyze the differences]")
    print("   - This suggests [hypothesis about why errors occur]")
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🔍 Error Analysis Best Practices

    **Always ask**:
    1. **Which classes have most errors?** (focus improvement efforts)
    2. **Are errors random or systematic?** (pattern = fixable)
    3. **Do error cases share characteristics?** (edge cases, outliers)
    4. **What features would help?** (informed feature engineering)

    **In production**:
    - Monitor error rates per class over time
    - Investigate sudden changes in error patterns
    - Collect more data for high-error classes
    - Create ensemble models for confused pairs

    ---
    ## Section 4: Confidence and Calibration

    **Key insight**: A model that predicts "90% confident" should be right 90% of the time!

    Most models are poorly calibrated (overconfident or underconfident).

    Let's examine prediction confidence:
    """)
    return


@app.cell
def _(np, y_pred, y_pred_proba, y_test):
    # Get confidence (max probability) for each prediction
    confidences = np.max(y_pred_proba, axis=1)
    is_correct = (y_pred == y_test.values)

    # Bin by confidence
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(confidences, bins)

    print("Confidence vs Actual Accuracy:")
    print("=" * 50)
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = is_correct[mask].mean()
            bin_conf = confidences[mask].mean()
            print(f"  Confidence {bins[i-1]:.1f}-{bins[i]:.1f}: Actual Accuracy={bin_acc:.2%} (n={mask.sum()})")
    return


@app.cell
def _(mo):
    mo.md("""
    ### 💡 Confidence Insights

    **Well-calibrated model**: Confidence matches actual accuracy
    - Predicts 80% confidence → Actually right 80% of time

    **Overconfident model**: Confidence > actual accuracy
    - Predicts 90% confidence → Actually right 70% of time
    - Dangerous in production!

    **Underconfident model**: Confidence < actual accuracy
    - Predicts 60% confidence → Actually right 80% of time
    - Leaves value on the table

    **In production**: Use confidence thresholds
    - High confidence → Auto-accept
    - Medium confidence → Human review
    - Low confidence → Reject or escalate

    ---
    ## Section 5: Model Card (Production Documentation)

    **Model Card** = Standardized documentation for ML models

    Created by Google researchers to promote transparency and accountability.

    Let's create one for our Pokemon classifier:
    """)
    return


@app.cell
def _(mo, overall_acc):
    mo.md(f"""
    # Model Card: Pokemon Type Classifier

    ## Model Details
    - **Model Type**: Random Forest Classifier
    - **Version**: 1.0
    - **Date**: 2025-01-19
    - **Developers**: ML Engineering Team
    - **License**: Internal Use Only

    ## Intended Use
    - **Primary Use**: Classify Pokemon cards by type based on stats
    - **Intended Users**: Internal card management system
    - **Out-of-Scope Uses**: NOT for competitive gaming predictions, NOT for market pricing

    ## Training Data
    - **Dataset**: Pokemon Card Database (800 cards)
    - **Data Collection**: Generated from historical card data
    - **Preprocessing**: Standardization, feature engineering
    - **Data Splits**: 70% train, 15% val, 15% test

    ## Performance Metrics
    - **Overall Accuracy**: {overall_acc:.2%}
    - **Best Performing Classes**: Fire, Water, Grass (>75% F1)
    - **Worst Performing Classes**: Poison, Ground, Ice (<50% F1)
    - **Known Limitations**: Struggles with rare types (low support)

    ## Ethical Considerations
    - **Bias**: Model performs better on common types (data imbalance)
    - **Fairness**: No discriminatory impact (classification only)
    - **Privacy**: No PII in training data

    ## Caveats and Recommendations
    - ⚠️ **Do not use** on Pokemon from generations not in training data
    - ⚠️ **Do not use** if input stats are outside training range
    - ⚠️ **Do use** confidence thresholds (recommend >70% for auto-accept)
    - ✅ **Monitor** prediction distribution drift
    - ✅ **Retrain** quarterly with new card releases

    ## Model Maintenance
    - **Monitoring**: Track per-class accuracy weekly
    - **Retraining**: Quarterly or when accuracy drops >5%
    - **Owner**: ML Engineering Team
    - **Contact**: ml-team@company.com
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 📋 Why Model Cards Matter

    **Benefits**:
    - Transparency about capabilities and limitations
    - Prevents misuse (out-of-scope applications)
    - Documents ethical considerations
    - Facilitates audits and compliance
    - Helps users understand when to trust predictions

    **In practice**:
    - Required by many organizations
    - Part of model governance
    - Updated with each model version
    - Reviewed by stakeholders before deployment

    ---
    ## Section 6: Production Readiness Checklist

    Before deploying to production, answer these questions:
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### ✅ Production Readiness Checklist

    **Performance**:
    - [ ] Metrics meet business requirements?
    - [ ] Performance validated on held-out test set?
    - [ ] Error analysis completed?
    - [ ] Edge cases identified and handled?

    **Robustness**:
    - [ ] Model handles missing features gracefully?
    - [ ] Model handles out-of-range inputs?
    - [ ] Confidence calibration verified?
    - [ ] Failure modes documented?

    **Documentation**:
    - [ ] Model card created?
    - [ ] Training process documented?
    - [ ] Feature definitions documented?
    - [ ] Runbook for operations team?

    **Monitoring**:
    - [ ] Metrics to track in production defined?
    - [ ] Alerting thresholds set?
    - [ ] Data drift detection configured?
    - [ ] Retraining strategy documented?

    **Ethics & Compliance**:
    - [ ] Bias analysis completed?
    - [ ] Privacy requirements met?
    - [ ] Regulatory compliance verified?
    - [ ] Stakeholder sign-off obtained?

    ---
    ## Key Takeaways

    ### ✅ What You Learned

    1. **Accuracy is often the wrong metric**
    2. **Confusion matrices reveal actionable insights**
    3. **Error analysis drives improvement**
    4. **Confidence calibration matters in production**
    5. **Model cards prevent misuse**

    ### 🤔 Socratic Questions

    1. **"Your model is 95% accurate on spam detection. Is that good?"**
       - Depends on base rate! Could be worse than dummy classifier.

    2. **"Would you rather have high precision or high recall for cancer detection?"**
       - High recall! Missing cancer (false negative) is worse than false alarm.

    3. **"How do you explain 'the model is 80% confident' to a non-technical user?"**
       - "Based on similar examples, this prediction is correct 8 out of 10 times."

    ---
    ## 🎯 Module 4 Checkpoint

    You've completed Module 4 when you can:

    - [ ] Choose the right metric for any business problem
    - [ ] Interpret confusion matrices and find patterns
    - [ ] Conduct thorough error analysis
    - [ ] Create professional model cards
    - [ ] Assess production readiness

    **Next**: Module 5 - Deployment & Inference

    Now that we know our model is good, let's ship it to production!
    """)
    return


if __name__ == "__main__":
    app.run()
