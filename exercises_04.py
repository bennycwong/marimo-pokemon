"""
Module 4 Exercises: Model Evaluation & Validation
=================================================

These exercises reinforce the concepts from Module 4.
Complete each exercise and check your solutions.

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
    mo.md(
        """
        # Module 4 Exercises

        Complete these exercises to master model evaluation and validation.

        ## Exercise 4.1: Metric Selection Challenge (30 min)

        **Goal**: Choose the right metric for different business scenarios.

        **Scenario**: You're consulting for 3 different companies. Each has different business constraints and costs.

        For each scenario below:
        1. Identify the primary metric to optimize
        2. Explain WHY that metric matters most
        3. Identify the cost of false positives vs false negatives
        4. Set an acceptable threshold

        **Learning Objective**: Understand that the "best" metric depends on business context, not just technical performance.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Scenario A: Email Spam Filter

        **Context**:
        - Users receive ~100 emails/day
        - False positive (good email marked spam) = user misses important email
        - False negative (spam gets through) = minor annoyance
        - User feedback: "I hate when real emails go to spam!"

        **Your Task**:
        1. Should you optimize for Precision or Recall? Why?
        2. What's more costly: false positive or false negative?
        3. If you had to choose: 95% precision + 70% recall OR 80% precision + 95% recall?
        4. What F-beta score should you use (F1, F0.5, F2)?

        **TODO: Write your analysis below**
        ```
        Primary metric: [YOUR ANSWER]
        Reasoning: [YOUR REASONING]
        Cost analysis: [YOUR COST ANALYSIS]
        Threshold recommendation: [YOUR THRESHOLD]
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Scenario B: Cancer Screening Test

        **Context**:
        - Screening test for early cancer detection
        - False positive = unnecessary follow-up tests (expensive, stressful, but safe)
        - False negative = missed cancer (potentially fatal)
        - Follow-up tests cost $2,000 but are very accurate

        **Your Task**:
        1. Should you optimize for Precision or Recall? Why?
        2. What's more costly: false positive or false negative?
        3. If you had to choose: 60% precision + 99% recall OR 95% precision + 85% recall?
        4. What confidence threshold would you set (0.1, 0.5, 0.9)?

        **TODO: Write your analysis below**
        ```
        Primary metric: [YOUR ANSWER]
        Reasoning: [YOUR REASONING]
        Cost analysis: [YOUR COST ANALYSIS]
        Threshold recommendation: [YOUR THRESHOLD]
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Scenario C: Credit Card Fraud Detection

        **Context**:
        - Transactions processed in real-time
        - False positive = legitimate transaction declined (customer angry, lost sale)
        - False negative = fraudulent transaction approved (company loses money)
        - Average fraud: $500, Average legitimate transaction: $150
        - Fraud rate: 0.5% of all transactions

        **Your Task**:
        1. Should you optimize for Precision or Recall? Why?
        2. Calculate: Cost of false positive vs cost of false negative
        3. What's the business impact of 90% recall vs 99% recall?
        4. Would you optimize for F1 or something else?

        **TODO: Write your analysis below**
        ```
        Primary metric: [YOUR ANSWER]
        Reasoning: [YOUR REASONING]
        Cost analysis: [YOUR COST ANALYSIS]
        Business impact: [YOUR IMPACT ANALYSIS]
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---

        ## Exercise 4.2: Error Analysis Deep Dive (45 min)

        **Goal**: Systematically analyze model errors to find improvement opportunities.

        **Instructions**:
        1. Load the Pokemon dataset and train a model
        2. Find all misclassified examples
        3. Categorize errors into groups
        4. Identify patterns
        5. Recommend 3 specific improvements

        **Learning Objective**: Error analysis reveals more than overall metrics.
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load data
    DATA_PATH = Path("data/clean/pokemon_cards_clean_latest.csv")
    df = pd.read_csv(DATA_PATH)

    # Feature engineering
    df['total_stats'] = df['hp'] + df['attack'] + df['defense'] + df['sp_attack'] + df['sp_defense'] + df['speed']
    df['physical_bias'] = (df['attack'] + df['defense']) - (df['sp_attack'] + df['sp_defense'])

    feature_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
                    'total_stats', 'physical_bias', 'generation', 'is_legendary']
    X = df[feature_cols]
    y = df['type']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    print(f"✅ Model trained: {len(X_test)} test samples")
    print(f"✅ Overall accuracy: {(y_pred == y_test).mean():.2%}")
    return (
        DATA_PATH,
        X,
        X_test,
        X_test_scaled,
        X_train,
        X_train_scaled,
        df,
        feature_cols,
        model,
        np,
        pd,
        plt,
        scaler,
        sns,
        y,
        y_pred,
        y_test,
        y_train,
    )


@app.cell
def __(X_test, df, np, pd, y_pred, y_test):
    # TODO: Create a DataFrame with all misclassified examples
    # Include: actual type, predicted type, confidence, and all features

    errors_df = None  # TODO: Create this DataFrame

    # Hint: You'll need to combine X_test, y_test, y_pred, and prediction probabilities
    # errors_df = pd.DataFrame({
    #     'actual_type': ...,
    #     'predicted_type': ...,
    #     'confidence': ...,
    #     ... features ...
    # })

    # TODO: Answer these questions:
    # 1. What percentage of errors are on which types?
    # 2. Which type pairs are most commonly confused?
    # 3. Do errors have common feature patterns (e.g., all low HP)?
    # 4. Are errors correlated with low confidence scores?

    print("TODO: Analyze misclassified examples")
    return (errors_df,)


@app.cell
def __(mo):
    mo.md(
        """
        ### TODO: Complete Your Error Analysis

        After analyzing the errors, answer:

        1. **Most Confused Type Pairs**: Which types does the model confuse most often?

        2. **Feature Patterns**: Do misclassified Pokemon share characteristics?
           - Low total stats?
           - Similar attack/defense/speed patterns?
           - Specific generations?

        3. **Confidence Analysis**: Are errors concentrated in low-confidence predictions?

        4. **Recommendations**: Based on your analysis, suggest 3 specific improvements:
           ```
           Improvement 1: [e.g., "Collect more training data for Fairy type"]
           Reasoning: [Why this will help]

           Improvement 2: [e.g., "Add type-specific feature: elemental_affinity"]
           Reasoning: [Why this will help]

           Improvement 3: [e.g., "Use ensemble with type-specific models"]
           Reasoning: [Why this will help]
           ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---

        ## Exercise 4.3: Confidence Threshold Tuning (30 min)

        **Goal**: Find the optimal confidence threshold for a business scenario.

        **Scenario**: You're deploying the Pokemon type classifier to a card pricing API.
        - **If confident (>threshold)**: Auto-price the card
        - **If uncertain (<threshold)**: Send to human review
        - **Business constraints**:
          - Human review costs $0.50 per card
          - Wrong type = wrong price = angry customer + refund cost ~$5
          - You process 1000 cards/day

        **Your Task**: Find the confidence threshold that minimizes total cost.

        **Learning Objective**: Optimal thresholds balance automation vs accuracy vs cost.
        """
    )
    return


@app.cell
def __(X_test_scaled, model, np, pd, y_test):
    # Get prediction probabilities
    y_proba = model.predict_proba(X_test_scaled)

    # Get confidence (max probability)
    confidence_scores = y_proba.max(axis=1)

    # Get predictions
    y_pred_proba = model.classes_[y_proba.argmax(axis=1)]

    # Create DataFrame for analysis
    threshold_df = pd.DataFrame({
        'true_label': y_test.values,
        'predicted_label': y_pred_proba,
        'confidence': confidence_scores,
        'correct': y_test.values == y_pred_proba
    })

    print(f"✅ Confidence scores computed")
    print(f"Mean confidence: {confidence_scores.mean():.2%}")
    print(f"Min confidence: {confidence_scores.min():.2%}")
    print(f"Max confidence: {confidence_scores.max():.2%}")
    return confidence_scores, threshold_df, y_pred_proba, y_proba


@app.cell
def __(np, pd, threshold_df):
    # TODO: Complete this cost analysis function

    def analyze_threshold_cost(threshold: float, df: pd.DataFrame) -> dict:
        """
        Analyze the cost of using a confidence threshold.

        Args:
            threshold: Confidence threshold (0.0 to 1.0)
            df: DataFrame with predictions and confidence scores

        Returns:
            Dictionary with cost analysis
        """
        # Filter predictions above threshold
        auto_predictions = df[df['confidence'] >= threshold]
        manual_review = df[df['confidence'] < threshold]

        # Calculate metrics
        n_auto = len(auto_predictions)
        n_manual = len(manual_review)

        # TODO: Calculate these metrics
        # - How many auto predictions are correct?
        # - How many auto predictions are wrong?
        # - What's the total cost?

        n_correct = 0  # TODO: auto_predictions['correct'].sum()
        n_errors = 0   # TODO: n_auto - n_correct

        # Costs
        manual_review_cost = 0  # TODO: n_manual * 0.50
        error_cost = 0          # TODO: n_errors * 5.00
        total_cost = 0          # TODO: manual_review_cost + error_cost

        return {
            'threshold': threshold,
            'n_auto': n_auto,
            'n_manual': n_manual,
            'n_correct': n_correct,
            'n_errors': n_errors,
            'auto_accuracy': n_correct / n_auto if n_auto > 0 else 0,
            'manual_review_cost': manual_review_cost,
            'error_cost': error_cost,
            'total_cost': total_cost,
            'cost_per_card': total_cost / len(df)
        }

    # TODO: Test different thresholds
    thresholds = np.arange(0.3, 1.0, 0.05)
    results = []  # TODO: [analyze_threshold_cost(t, threshold_df) for t in thresholds]

    # TODO: Create a DataFrame and find optimal threshold
    # results_df = pd.DataFrame(results)
    # optimal_threshold = results_df.loc[results_df['total_cost'].idxmin()]

    print("TODO: Complete the cost analysis and find optimal threshold")
    return (analyze_threshold_cost,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### TODO: Threshold Analysis Questions

        After completing the cost analysis, answer:

        1. **Optimal Threshold**: What confidence threshold minimizes total cost?

        2. **Cost Breakdown**: At the optimal threshold:
           - How many cards go to auto-pricing?
           - How many go to human review?
           - What's the total daily cost?
           - Cost per card?

        3. **Tradeoffs**: What happens if you:
           - Increase threshold to 0.95 (very conservative)?
           - Decrease threshold to 0.50 (very aggressive)?

        4. **Business Recommendation**: Would you recommend deploying this system?
           ```
           Recommendation: [Deploy / Don't Deploy / Deploy with modifications]
           Reasoning: [Your reasoning]
           Risk assessment: [What could go wrong?]
           Success metrics: [How would you measure success in production?]
           ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---

        ## Exercise 4.4: Model Card Creation (30 min)

        **Goal**: Document your model professionally for stakeholders.

        **Instructions**: Create a complete model card following industry standards.

        **Learning Objective**: Documentation prevents misuse and builds trust.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Pokemon Type Classifier Model Card

        ### Model Details

        **TODO: Complete this section**

        - **Model Type**: [e.g., Random Forest Classifier]
        - **Version**: [e.g., v1.0.0]
        - **Date Trained**: [YYYY-MM-DD]
        - **Framework**: [e.g., scikit-learn 1.4.0]
        - **Input**: [Describe input features]
        - **Output**: [Describe output format]

        ---

        ### Intended Use

        **TODO: Describe the intended use case**

        **Primary Use Case**:
        - [What is this model designed for?]

        **Intended Users**:
        - [Who should use this model?]

        **Out of Scope**:
        - [What should this model NOT be used for?]

        ---

        ### Training Data

        **TODO: Document training data**

        - **Dataset**: [Name and source]
        - **Size**: [Number of samples]
        - **Features**: [List key features]
        - **Target**: [What are you predicting?]
        - **Date Range**: [When was data collected?]
        - **Known Issues**: [Any data quality concerns?]

        ---

        ### Performance Metrics

        **TODO: Document performance**

        **Overall Performance**:
        - Accuracy: [X.XX%]
        - Macro F1: [X.XX]
        - Weighted F1: [X.XX]

        **Per-Class Performance**:
        | Type      | Precision | Recall | F1-Score | Support |
        |-----------|-----------|--------|----------|---------|
        | Fire      | TODO      | TODO   | TODO     | TODO    |
        | Water     | TODO      | TODO   | TODO     | TODO    |
        | ...       | ...       | ...    | ...      | ...     |

        **Error Analysis**:
        - Most common errors: [Describe]
        - Edge cases: [Describe]

        ---

        ### Limitations

        **TODO: Be honest about limitations**

        **Known Limitations**:
        1. [Limitation 1, e.g., "Lower accuracy on Fairy type due to limited training data"]
        2. [Limitation 2, e.g., "Does not account for dual-type Pokemon"]
        3. [Limitation 3, e.g., "Trained only on Generation 1-8 Pokemon"]

        **Failure Modes**:
        1. [When does this model fail?]
        2. [What inputs cause problems?]

        ---

        ### Ethical Considerations

        **TODO: Consider ethical implications**

        **Bias Assessment**:
        - [Are some types over/under-represented?]
        - [Does the model favor certain generations?]

        **Fairness**:
        - [Does the model treat all types fairly?]

        **Potential Misuse**:
        - [How could this model be misused?]
        - [What safeguards are needed?]

        ---

        ### Deployment Recommendations

        **TODO: Provide deployment guidance**

        **Confidence Threshold**: [Based on Exercise 4.3]

        **Monitoring Plan**:
        - [What metrics should be tracked?]
        - [What alerts should be set up?]
        - [How often should the model be retrained?]

        **Rollback Plan**:
        - [What triggers a rollback?]
        - [What's the fallback system?]

        ---

        ### Contact Information

        **Model Owner**: [Your name/team]
        **Last Updated**: [Date]
        **Questions**: [How to reach you]
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---

        ## Exercise 4.5: Model Review Board (30 min)

        **Goal**: Present your model to stakeholders for approval.

        **Scenario**: You're presenting to a review board that will decide whether to deploy your model.

        **The Board Includes**:
        - VP of Product (non-technical, cares about business impact)
        - Engineering Manager (technical, cares about reliability)
        - Data Science Lead (technical, cares about methodology)
        - Risk Manager (cares about what could go wrong)

        **Your Task**: Prepare a 5-slide presentation

        **Learning Objective**: Communication is as important as technical skill.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### TODO: Prepare Your Presentation

        Create an outline for 5 slides:

        **Slide 1: Executive Summary**
        ```
        - One sentence: What does this model do?
        - Key metric: How well does it work?
        - Recommendation: Should we deploy it?
        - Expected impact: What business value?
        ```

        **Slide 2: Performance Highlights**
        ```
        - Overall accuracy: [X%]
        - Strengths: [What does it do well?]
        - Comparison: [How does this compare to baseline/current system?]
        - Visual: [What chart would you show?]
        ```

        **Slide 3: Known Limitations**
        ```
        - Limitation 1: [Be specific]
        - Limitation 2: [Be specific]
        - Mitigation: [How are you handling these?]
        - When it fails: [Specific scenarios]
        ```

        **Slide 4: Risk Assessment**
        ```
        - High risk scenario: [What's the worst that could happen?]
        - Medium risk scenario: [What's likely to happen?]
        - Mitigation strategies: [How are you reducing risk?]
        - Monitoring plan: [How will you detect problems?]
        ```

        **Slide 5: Recommendation & Next Steps**
        ```
        - Deploy: [Yes/No/Conditional]
        - Timeline: [When can we deploy?]
        - Success metrics: [How will we measure success?]
        - Rollback plan: [What if it goes wrong?]
        ```

        ---

        ### Practice: Answer Tough Questions

        **From VP of Product**:
        "85% accuracy sounds good, but what does that mean for our customers?"

        **Your answer**:
        [TODO: Write a non-technical explanation]

        **From Engineering Manager**:
        "What happens if the model crashes in production?"

        **Your answer**:
        [TODO: Explain your error handling and fallbacks]

        **From Data Science Lead**:
        "Did you check for data leakage? How do you know your validation set is representative?"

        **Your answer**:
        [TODO: Explain your validation methodology]

        **From Risk Manager**:
        "What's the worst case scenario if this model is wrong?"

        **Your answer**:
        [TODO: Honest risk assessment]
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---

        ## Bonus Exercise 4.6: Compare Two Models (Optional - 30 min)

        **Scenario**: You trained two models:
        - Model A: 88% accuracy, 50ms inference time
        - Model B: 92% accuracy, 300ms inference time

        **Your Task**: Write a recommendation memo explaining which to deploy and why.

        **Consider**:
        - Performance vs latency tradeoff
        - Cost implications
        - User experience impact
        - Scale (1000 requests/sec)

        **Learning Objective**: Engineering decisions involve tradeoffs, not just "better accuracy."
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---

        ## Self-Assessment

        After completing these exercises, you should be able to:

        - ✅ Choose the right metric for any business problem
        - ✅ Conduct thorough error analysis
        - ✅ Set optimal confidence thresholds
        - ✅ Write professional model cards
        - ✅ Present models to stakeholders confidently
        - ✅ Make deployment recommendations with supporting evidence

        **If you can do all of these, you're ready for production ML evaluation!**
        """
    )
    return


if __name__ == "__main__":
    app.run()
