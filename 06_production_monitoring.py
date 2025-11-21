"""
Module 6: Production ML & Monitoring
====================================

Professional ML Engineering Onboarding Project

Learning Objectives:
- Set up comprehensive monitoring for ML systems
- Detect and respond to data drift
- Debug production ML incidents systematically
- Plan model maintenance and retraining
- Write incident reports and postmortems

Duration: 3 hours

⚠️ CRITICAL MODULE: Production is where 80% of ML problems occur.
This module teaches you to survive and thrive in production.
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
    # Module 6: Production ML & Monitoring

    **"In production, everything that can go wrong, will go wrong"** — Murphy's Law for ML

    ## Why This Module is Critical

    **Sobering Statistics**:
    - 90% of ML systems degrade within 6 months without monitoring
    - Average time to detect production issues: 2-4 weeks (too long!)
    - #1 cause of ML failures in production: Data drift
    - #2 cause: Infrastructure issues
    - #3 cause: Silent model degradation

    ## What You'll Learn

    1. **Monitoring**: What to track and when to alert
    2. **Data Drift**: Detect when your model becomes obsolete
    3. **Production Debugging**: Systematic incident response
    4. **Model Maintenance**: Keep your model healthy long-term
    5. **Incident Management**: Postmortems and continuous improvement

    ## Industry Reality

    > "Our model was 90% accurate for 3 months, then dropped to 65% overnight.
    > We didn't notice for 2 weeks. Cost us $500k in bad decisions."
    > — Real incident from a Fortune 500 company

    **This module teaches you to catch problems in minutes, not weeks.**
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 1: What to Monitor in Production

    **The Golden Rule**: If you can't measure it, you can't improve it.

    ### Four Categories of Metrics

    | Category | What to Track | Why It Matters |
    |----------|---------------|----------------|
    | **Infrastructure** | Latency, throughput, errors, CPU/GPU | User experience, cost |
    | **ML Performance** | Accuracy, confidence, predictions | Model quality |
    | **Data Quality** | Input distribution, missing values | Early warning of drift |
    | **Business Impact** | Revenue, conversions, user satisfaction | Actual value delivered |

    ### Infrastructure Metrics

    **Latency** (Response Time):
    - **P50 (median)**: Half of requests faster
    - **P95**: 95% of requests faster (target SLA)
    - **P99**: 99% of requests faster (worst case)
    - **Max**: Slowest request (outliers)

    **Example SLA**:
    - P95 latency < 100ms
    - P99 latency < 500ms
    - Max latency < 2000ms

    **Throughput**:
    - Requests per second (RPS)
    - Predictions per minute
    - Daily/weekly volumes

    **Error Rates**:
    - 4xx errors (client errors - bad input)
    - 5xx errors (server errors - your fault)
    - Target: <0.1% error rate
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### ML Performance Metrics

    **Track These Daily**:

    1. **Model Accuracy** (if you have labels)
       - Daily/weekly accuracy
       - Per-class accuracy
       - Compared to baseline

    2. **Confidence Distribution**
       - Average confidence score
       - % predictions with confidence >0.9 (high confidence)
       - % predictions with confidence <0.5 (uncertain)
       - **Red flag**: Confidence drops significantly

    3. **Prediction Distribution**
       - % of each class predicted
       - **Red flag**: All predictions become one class
       - **Red flag**: Distribution shifts dramatically

    4. **Error Patterns** (if you catch errors)
       - Most common misclassifications
       - Errors by feature values
       - Time-based error trends

    ### Data Quality Metrics

    **Monitor Your Inputs**:

    1. **Feature Distributions**
       - Mean, std dev of numeric features
       - Category frequencies
       - Missing value rates

    2. **Drift Scores**
       - Statistical distance from training data
       - Per-feature drift detection
       - Overall data drift score

    3. **Data Anomalies**
       - Out-of-range values
       - New categories not seen in training
       - Unexpected null rates

    ### Business Metrics

    **Connect ML to Value**:

    | ML System | Business Metric | Target |
    |-----------|----------------|--------|
    | Pokemon type classifier | Cards priced correctly (within $5) | >90% |
    | Recommendation | Click-through rate | >8% |
    | Fraud detection | False positive rate | <5% |
    | Churn prediction | Prevented churns | +500/month |

    **Always track**: How much value is your ML delivering?
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Dashboard Design Best Practices

    **Good Dashboard Structure**:

    ```
    ┌─────────────────────────────────────┐
    │ ML System Health Overview           │
    │ ● Status: ✅ Healthy                │
    │ ● Accuracy: 87% (target: 85%)      │
    │ ● Latency P95: 45ms (target: 100ms)│
    │ ● Error rate: 0.03% (target: 0.1%) │
    └─────────────────────────────────────┘

    ┌─────────────────┬──────────────────┐
    │ Infrastructure  │ ML Performance   │
    │ - Latency graph │ - Accuracy trend │
    │ - Error rates   │ - Confidence dist│
    │ - Traffic vol   │ - Prediction dist│
    └─────────────────┴──────────────────┘

    ┌─────────────────┬──────────────────┐
    │ Data Quality    │ Business Impact  │
    │ - Drift scores  │ - $ value        │
    │ - Feature dist  │ - User feedback  │
    │ - Anomalies     │ - Conversions    │
    └─────────────────┴──────────────────┘
    ```

    **Key Principles**:
    - ✅ Most important metrics at the top
    - ✅ Red/yellow/green status indicators
    - ✅ Trend lines (not just current values)
    - ✅ Comparison to targets/baselines
    - ✅ Time range selector (last hour/day/week)
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### 🎓 Socratic Question #1

    **Scenario**: Your Pokemon type classifier dashboard shows:
    - Accuracy: 87% (stable for 3 months)
    - Latency P95: 85ms (good)
    - Average confidence: 0.78 (down from 0.85 last month)
    - Error rate: 0.05% (good)
    - Prediction distribution: Fire type increased from 15% to 35%

    **Questions**:
    1. Is there a problem? If yes, which metrics are concerning?
    2. What could cause confidence to drop while accuracy stays stable?
    3. Why did Fire type predictions double?
    4. What would you investigate first?
    5. Should you page the on-call engineer at 2am for this?

    <details>
    <summary>💡 Analysis</summary>

    **Yes, there's a problem!**

    **Red Flags**:
    1. ⚠️ **Confidence dropped** from 0.85 to 0.78 (9% drop)
       - Model is less certain about predictions
       - Could indicate data drift (seeing unfamiliar patterns)

    2. 🚨 **Prediction distribution shifted** dramatically
       - Fire type: 15% → 35% (2.3x increase!)
       - Either: Data changed (lots of Fire cards suddenly)
       - Or: Model is broken (predicting Fire for everything)

    **Likely Cause**: **Data drift**
    - Training data had 15% Fire types
    - Production data now has more Fire types (or model thinks so)
    - Confidence drops because inputs look different from training

    **Investigation Steps**:
    1. Check input data distribution (are Fire cards actually increasing?)
    2. Sample recent predictions - are they correct?
    3. Compare feature distributions to training data
    4. Check if there was a product change (new card set released?)

    **Should you page someone at 2am?**
    - **No**, this is not an emergency (system still works)
    - **But**: Create a ticket for next business day
    - **And**: Set up an alert for next time (auto-detect this pattern)

    **Good alert would be**:
    - "Prediction distribution shifted >20% from baseline"
    - "Confidence dropped >5% in 24 hours"

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 2: Setting Up Alerts

    **The Art of Alerting**: Alert on problems, not noise.

    ### Alert Severity Levels

    | Severity | When | Response Time | Example |
    |----------|------|---------------|---------|
    | **🚨 P0 Critical** | System down | Immediate (page on-call) | API returning 500s |
    | **⚠️ P1 High** | Degraded performance | Within 1 hour | Accuracy dropped 20% |
    | **🟡 P2 Medium** | Early warning | Within 1 day | Drift detected |
    | **ℹ️ P3 Low** | FYI | When convenient | Traffic increased 2x |

    ### Good Alert Criteria

    **DO Alert When**:
    - ✅ Error rate >0.5% for 5+ minutes
    - ✅ P95 latency >200ms for 10+ minutes
    - ✅ Accuracy drops >10% in 24 hours
    - ✅ Prediction distribution changes >30%
    - ✅ Confidence drops >10% in 1 week
    - ✅ Infrastructure: CPU >80% for 15 minutes

    **DON'T Alert When**:
    - ❌ Single slow request (use P95, not max)
    - ❌ Tiny accuracy changes (87.2% → 87.1%)
    - ❌ Traffic spikes (unless breaking system)
    - ❌ Noisy metrics that fluctuate naturally

    ### Preventing Alert Fatigue

    **The Problem**: Too many alerts → People ignore them → Miss real issues

    **Solutions**:
    1. **Aggregate**: "5 errors" not "1 error, 1 error, 1 error, 1 error, 1 error"
    2. **Thresholds**: Use historical baselines, not arbitrary numbers
    3. **Windows**: Require sustained issues (e.g., "latency high for 10 min")
    4. **Snooze**: Allow on-call to snooze known issues
    5. **Review**: Monthly review of alerts - disable the noisy ones

    ### SLAs and SLOs for ML

    **SLA** (Service Level Agreement): Promise to customers
    - "99.9% uptime" = 43 minutes downtime per month
    - "P95 latency <100ms"
    - "Accuracy >85%"

    **SLO** (Service Level Objective): Internal target (stricter than SLA)
    - If SLA is 99.9%, SLO might be 99.95%
    - Gives buffer before violating customer promise

    **Example for Pokemon Classifier**:
    - **SLO**: 99.95% uptime, P95 latency <80ms, accuracy >87%
    - **SLA**: 99.9% uptime, P95 latency <100ms, accuracy >85%
    - Alert when approaching SLA limits (hit SLO)

    ### On-Call Best Practices

    **Rotation**:
    - Weekly rotations (not daily - too disruptive)
    - Primary + secondary on-call
    - Handoff document with current issues

    **Runbooks**:
    - Every alert links to a runbook
    - Runbook explains: What this means, how to investigate, how to fix
    - We'll create these in Section 3!

    **Escalation**:
    - P0: Page immediately
    - P1: Page if no response in 30 min
    - P2/P3: Don't page, create ticket
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 3: Data Drift Detection

    **The Silent Killer of ML Models**

    ### What is Data Drift?

    **Definition**: When production data differs from training data

    **Why It Matters**:
    - Model learned patterns from training data
    - If production data is different, patterns don't apply
    - Accuracy degrades silently

    **Real Example**:
    - Trained fraud model on 2019 data
    - COVID-19 in 2020 changed shopping patterns dramatically
    - Model accuracy dropped from 92% to 68%
    - Took 6 weeks to notice!

    ### Types of Drift

    **1. Feature Drift** (Input distribution changes)
    - Training: Average price $10
    - Production: Average price $50 (inflation!)
    - Model still works, but confidence is wrong

    **2. Concept Drift** (Relationship changes)
    - Training: High price = Rare card
    - Production: High price = Popular card (market shift)
    - Model fundamentally wrong now

    **3. Label Drift** (What we're predicting changes)
    - Training: "Spam" meant Nigerian prince emails
    - Production: "Spam" means crypto scams
    - Model misses new types of spam
    """)
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from scipy import stats

    sns.set_style("whitegrid")
    return Path, np, pd, plt, sns, stats


@app.cell
def __(Path, pd):
    # Load Pokemon data (simulating training data)
    DATA_PATH = Path("data/clean/pokemon_cards_clean_latest.csv")
    df_train = pd.read_csv(DATA_PATH)

    # Simulate "production" data with drift
    df_production = df_train.copy()

    # Introduce drift: Power creep! (newer cards have higher stats)
    np.random.seed(42)
    drift_indices = np.random.choice(df_production.index, size=200, replace=False)
    df_production.loc[drift_indices, 'attack'] = df_production.loc[drift_indices, 'attack'] * 1.5
    df_production.loc[drift_indices, 'hp'] = df_production.loc[drift_indices, 'hp'] * 1.3

    print(f"Training data: {len(df_train)} cards")
    print(f"Production data (with drift): {len(df_production)} cards")
    return df_production, df_train, drift_indices


@app.cell
def __(mo):
    mo.md("""
    ### Detecting Drift: Statistical Tests

    **Method 1: Kolmogorov-Smirnov (KS) Test**
    - Compares two distributions
    - Returns p-value (0 = different, 1 = same)
    - Works for continuous features

    **Method 2: Chi-Square Test**
    - For categorical features
    - Compares category frequencies

    **Method 3: Population Stability Index (PSI)**
    - Industry standard
    - PSI <0.1 = No drift
    - PSI 0.1-0.2 = Moderate drift
    - PSI >0.2 = Significant drift
    """)
    return


@app.cell
def __(df_production, df_train, pd, stats):
    def detect_drift_ks_test(train_data: pd.Series, prod_data: pd.Series, threshold: float = 0.05) -> dict:
        """
        Detect drift using Kolmogorov-Smirnov test.

        Args:
            train_data: Training feature values
            prod_data: Production feature values
            threshold: P-value threshold (default 0.05)

        Returns:
            Dict with drift detection results
        """
        # Perform KS test
        statistic, p_value = stats.ks_2samp(train_data, prod_data)

        drift_detected = p_value < threshold

        return {
            'ks_statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected,
            'severity': 'High' if p_value < 0.001 else ('Medium' if p_value < 0.05 else 'Low')
        }

    # Test for drift in key features
    features_to_monitor = ['hp', 'attack', 'defense', 'speed']

    drift_results = {}
    for feature in features_to_monitor:
        result = detect_drift_ks_test(df_train[feature], df_production[feature])
        drift_results[feature] = result

        status = "🚨 DRIFT" if result['drift_detected'] else "✅ OK"
        print(f"{feature:15} {status:10} p-value: {result['p_value']:.4f}")

    return detect_drift_ks_test, drift_results, features_to_monitor


@app.cell
def __(df_production, df_train, features_to_monitor, mo, plt):
    # Visualize drift
    _fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, _feature in enumerate(features_to_monitor):
        ax = axes[idx]

        # Plot distributions
        ax.hist(df_train[_feature], bins=30, alpha=0.5, label='Training', color='blue')
        ax.hist(df_production[_feature], bins=30, alpha=0.5, label='Production', color='red')

        ax.set_xlabel(_feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{_feature.title()} Distribution')
        ax.legend()

    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return ax, axes


@app.cell
def __(mo):
    mo.md("""
    ### When to Retrain Your Model

    **Retraining Triggers**:

    | Trigger | When | Example |
    |---------|------|---------|
    | **Schedule** | Every X weeks/months | Every quarter, regardless |
    | **Drift threshold** | Drift score >0.2 | Multiple features drifting |
    | **Accuracy drop** | Accuracy <target | Below 85% for 3 days |
    | **Business event** | Major change | New product launch |
    | **Manual** | On demand | After fixing data bug |

    **Retraining Strategy**:

    ```python
    # Pseudo-code for retraining logic
    if drift_score > 0.2 OR accuracy < 0.85 for 3 days:
        trigger_retraining()

    if days_since_last_training > 90:
        trigger_scheduled_retraining()
    ```

    **A/B Testing New Models**:
    - Don't replace old model immediately!
    - Run new model on 5% of traffic (shadow mode)
    - Compare performance for 1 week
    - If better: Gradual rollout (5% → 25% → 50% → 100%)
    - If worse: Rollback
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### 🎓 Socratic Question #2

    **Scenario**: You detect drift in 3 features:
    - `hp`: p-value = 0.001 (HIGH drift)
    - `attack`: p-value = 0.03 (MEDIUM drift)
    - `generation`: p-value = 0.8 (NO drift)

    Model accuracy is still 87% (target: 85%).

    **Questions**:
    1. Should you retrain immediately?
    2. What might explain drift but stable accuracy?
    3. How would you decide if retraining is worth it?
    4. What data would you retrain on?

    <details>
    <summary>💡 Decision Framework</summary>

    **Should you retrain immediately? Maybe not!**

    **Analysis**:
    - ✅ Accuracy still meets target (87% > 85%)
    - ⚠️ Drift detected (features look different)
    - 🤔 Model still performs well despite drift

    **Possible Explanations**:
    1. **Robust model**: Random Forest handles distribution shift well
    2. **Irrelevant features**: Maybe `hp` isn't that important for classification
    3. **Correlated features**: `attack` drifted but correlated features didn't
    4. **Leading indicator**: Accuracy fine NOW, but will degrade soon

    **Decision Framework**:

    **Don't retrain if**:
    - Accuracy still good
    - Drift is minor (<0.2 PSI)
    - Business stakeholders happy

    **DO retrain if**:
    - Accuracy dropping (even if above threshold)
    - High confidence predictions becoming low confidence
    - Drift is severe (>0.3 PSI) even if accuracy OK
    - Better safe than sorry for critical systems

    **Recommendation for this scenario**:
    "Monitor closely, prepare retraining, but don't deploy yet:
    1. Set up daily accuracy tracking
    2. Prepare retraining pipeline (collect recent data)
    3. Train new model in parallel (don't deploy)
    4. Compare old vs new model performance
    5. Deploy new model only if meaningfully better"

    **What data to retrain on**:
    - **Option A**: Last 3-6 months (recent data only)
    - **Option B**: All historical + recent (larger dataset)
    - **Best practice**: Combine both, weight recent data higher
    - **Critical**: Validate on CURRENT production data, not old test set

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 4: Production Debugging Runbook ⭐ MOST IMPORTANT

    **When Things Go Wrong (And They Will)**

    ### The Debugging Mindset

    **Don't Panic. Be Systematic.**

    1. **Gather symptoms** (what's actually broken?)
    2. **Form hypotheses** (what could cause this?)
    3. **Test hypotheses** (check evidence)
    4. **Fix root cause** (not just symptoms)
    5. **Document** (so others learn)

    ### Incident Response Flowchart

    ```
    🚨 Alert Received
          ↓
    [Is system down?]
          ↓ Yes → IMMEDIATE: Rollback to last known good version
          ↓ No
    [Impact >10% users?]
          ↓ Yes → HIGH PRIORITY: Investigate now
          ↓ No → MEDIUM: Investigate within 24h
          ↓
    [Gather Evidence]
    - Logs
    - Metrics
    - Recent changes
    - User reports
          ↓
    [Form Hypothesis]
          ↓
    [Test & Fix]
          ↓
    [Document]
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Scenario 1: Accuracy Dropped from 87% to 65% Overnight

    **Symptoms**:
    - Model accuracy: 87% → 65% (25% drop)
    - Happened overnight
    - No code changes
    - Traffic volume normal

    **Debugging Checklist**:

    #### Step 1: Check Data Pipeline (5 minutes)
    ```bash
    # Check if data is flowing
    - ✅ Input count normal?
    - ✅ No unusual nulls/errors?
    - ✅ Feature distributions look normal?
    ```

    **Common Causes**:
    - Upstream service changed schema (new column format)
    - Database migration broke queries
    - Feature engineering bug introduced

    #### Step 2: Check Model Loading (2 minutes)
    ```python
    # Verify correct model is loaded
    - ✅ Model version correct?
    - ✅ Model file not corrupted?
    - ✅ Feature order correct?
    ```

    **Common Causes**:
    - Deployed wrong model version
    - Model file corrupted during deployment
    - Feature names mismatched

    #### Step 3: Compare Input Distribution (10 minutes)
    ```python
    # Compare today's data to training data
    for feature in features:
        train_mean = training_data[feature].mean()
        prod_mean = production_data[feature].mean()
        if abs(prod_mean - train_mean) / train_mean > 0.5:
            print(f"🚨 {feature} changed significantly!")
    ```

    **Common Causes**:
    - Sudden data drift (market change, new product)
    - Upstream service started sending different data
    - Bug in data preprocessing

    #### Step 4: Sample Predictions (5 minutes)
    ```python
    # Look at actual predictions
    sample_errors = predictions[predictions['correct'] == False].head(20)
    # What do errors have in common?
    ```

    **Common Causes**:
    - Model predicting same class for everything
    - Specific feature values cause failures
    - Edge cases not handled

    #### Step 5: Check Recent Changes (5 minutes)
    ```bash
    # What changed in last 24 hours?
    - ML service deployment?
    - Upstream service update?
    - Infrastructure change?
    - Data source change?
    ```

    ### Decision Tree: What to Do

    ```
    [Accuracy dropped]
          ↓
    [Check model loading]
          ↓ Wrong version? → Rollback
          ↓ Correct
    [Check data pipeline]
          ↓ Data broken? → Fix pipeline
          ↓ Data OK
    [Check input distribution]
          ↓ Massive drift? → Retrain urgent
          ↓ Minor drift
    [Sample errors]
          ↓ Specific pattern? → Fix that pattern
          ↓ Random
    [Escalate to senior engineer]
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Scenario 2: Latency Spiked from 50ms to 2000ms

    **Symptoms**:
    - P95 latency: 50ms → 2000ms (40x slower!)
    - Started 2 hours ago
    - Users complaining about slow predictions
    - CPU/GPU usage looks normal

    **Debugging Checklist**:

    #### Step 1: Check Infrastructure (2 minutes)
    ```bash
    # Resource usage
    - CPU: ___% (normal <80%)
    - Memory: ___% (normal <80%)
    - GPU: ___% (if using)
    - Disk I/O: ___ (could be bottleneck)
    ```

    **Common Causes**:
    - Resource exhaustion (CPU/memory maxed)
    - Disk I/O bottleneck
    - Network issues

    #### Step 2: Check Request Patterns (5 minutes)
    ```python
    # Analyze request logs
    - Average request size: ___
    - Slow requests have something in common?
    - Batch size changed?
    ```

    **Common Causes**:
    - Batch processing broke (processing 1 at a time)
    - Large requests timing out
    - No request batching

    #### Step 3: Check Model Size (5 minutes)
    ```bash
    # Model file size
    du -h model.pkl

    # Is model loaded in memory?
    # Or loaded from disk each time? (SLOW!)
    ```

    **Common Causes**:
    - Model not cached in memory
    - Loading model on every request
    - Model file too large for memory

    #### Step 4: Check External Dependencies (5 minutes)
    ```bash
    # Do you call other services?
    - Feature store: Response time ___
    - Database: Query time ___
    - Other APIs: Response time ___
    ```

    **Common Causes**:
    - Downstream service slow/down
    - Database query not optimized
    - Network latency increased

    #### Step 5: Profile the Code (10 minutes)
    ```python
    import time

    start = time.time()
    # Step 1: Load input
    t1 = time.time()
    # Step 2: Preprocess
    t2 = time.time()
    # Step 3: Model inference
    t3 = time.time()
    # Step 4: Postprocess
    t4 = time.time()

    print(f"Load: {t1-start:.3f}s")
    print(f"Preprocess: {t2-t1:.3f}s")
    print(f"Inference: {t3-t2:.3f}s")
    print(f"Postprocess: {t4-t3:.3f}s")
    ```

    ### Quick Fixes for Latency

    | Problem | Fix |
    |---------|-----|
    | Model not cached | Load model once at startup, keep in memory |
    | No batching | Batch requests (process 10-100 at once) |
    | Slow preprocessing | Vectorize operations, use numpy |
    | Large model | Model compression, quantization |
    | Slow database | Add index, cache results |
    | Too many features | Feature selection, remove unused |
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Scenario 3: All Predictions Are One Class

    **Symptoms**:
    - 95% of predictions are "Fire" type (should be ~15%)
    - Started 1 hour ago
    - Confidence scores look normal (0.7-0.9)
    - No errors in logs

    **This is Serious**: Model is broken but doesn't know it!

    **Debugging Checklist**:

    #### Step 1: Verify Inputs (5 minutes)
    ```python
    # Sample recent inputs
    recent_inputs.head(20)

    # Are inputs actually all Fire-type cards?
    # Or is model wrongly predicting?
    ```

    #### Step 2: Check Feature Engineering (10 minutes)
    ```python
    # Print actual feature values being fed to model
    print("Features:", feature_values)

    # Common bugs:
    - All zeros? (preprocessing broke)
    - All NaNs? (missing value handling broke)
    - Wrong scale? (normalization issue)
    - Wrong order? (feature order shuffled)
    ```

    **Common Causes**:
    - Preprocessing broke (features all 0 or NaN)
    - Feature scaling issue
    - Feature names mismatched

    #### Step 3: Check Model Classes (5 minutes)
    ```python
    # Verify model knows all classes
    print("Model classes:", model.classes_)

    # Verify prediction probabilities
    proba = model.predict_proba(sample_input)
    print("Probabilities:", proba)
    ```

    **Common Causes**:
    - Model only trained on Fire types (data bug)
    - Class imbalance in training
    - Model overfitting to one class

    #### Step 4: Test with Known Inputs (5 minutes)
    ```python
    # Create test input you KNOW should be Water type
    test_water_card = {
        'hp': 80, 'attack': 60, 'defense': 70,
        'sp_attack': 90, 'sp_defense': 80, 'speed': 70
    }

    prediction = predict(test_water_card)
    print(f"Predicted: {prediction}, Expected: Water")
    ```

    #### Step 5: Rollback Immediately

    ```bash
    # This is broken - rollback
    kubectl set image deployment/ml-service ml=ml:v1.2.3  # previous version
    ```

    ### Post-Incident

    - ✅ Rollback to working version
    - ✅ Investigate root cause offline
    - ✅ Add test for this scenario
    - ✅ Write postmortem
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Creating Incident Reports

    **Template**:

    ```markdown
    # Incident Report: [Brief Description]

    **Date**: YYYY-MM-DD
    **Severity**: P0/P1/P2
    **Duration**: X hours
    **Impact**: Y users affected, $Z revenue impact

    ## Summary
    [2-3 sentences: What happened, impact, resolution]

    ## Timeline
    - 14:00: Alert triggered (accuracy dropped)
    - 14:05: Engineer paged
    - 14:15: Root cause identified (data pipeline bug)
    - 14:30: Fix deployed
    - 14:45: System restored
    - 15:00: Monitoring confirmed resolution

    ## Root Cause
    [Detailed explanation of what went wrong and why]

    ## Impact
    - Users affected: 1,234
    - Duration: 45 minutes
    - Financial: $5,000 estimated (refunds + lost sales)
    - Reputational: 23 customer complaints

    ## Resolution
    [What fixed it]

    ## Action Items
    - [ ] Add test for this scenario (Owner: @engineer1, Due: next week)
    - [ ] Improve monitoring (Owner: @engineer2, Due: 2 weeks)
    - [ ] Update runbook (Owner: @engineer3, Due: 3 days)

    ## Prevention
    [How to prevent this in the future]
    ```

    ### Blameless Postmortems

    **Key Principles**:
    - ❌ Don't blame individuals ("John broke it")
    - ✅ Focus on systems ("We need better testing")
    - ✅ Assume good intentions
    - ✅ Learn from failures
    - ✅ Share widely (everyone learns)

    **Good Postmortem Questions**:
    - What broke and why?
    - How did we detect it? (Fast enough?)
    - How did we fix it? (Fast enough?)
    - How do we prevent this? (Process, not people)
    - What went well? (Celebrate fast response!)
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### 🎓 Socratic Question #3

    **Scenario**: You're on-call. At 3am, you get paged:

    "🚨 P1 ALERT: ML Service error rate 15% (threshold: 0.5%)"

    You check logs and see:
    ```
    ValueError: Input feature 'generation' expected int, got str
    ValueError: Input feature 'generation' expected int, got str
    ValueError: Input feature 'generation' expected int, got str
    (repeats 1,500 times in last 10 minutes)
    ```

    **Questions**:
    1. What's the likely root cause?
    2. What's your immediate action?
    3. How do you fix it long-term?
    4. Who do you need to coordinate with?
    5. What should you add to prevent this?

    <details>
    <summary>💡 Incident Response</summary>

    **Root Cause**: Upstream service changed data format
    - Was sending `generation: 1` (int)
    - Now sending `generation: "1"` (string)
    - Your preprocessing expects int, breaks on string

    **Immediate Action (3:05am)**:
    ```python
    # Option A: Quick Fix - Deploy type conversion
    generation = int(input_data['generation'])  # Convert string to int

    # Option B: Rollback upstream service (if you can)

    # Option C: Temporary - Make your service handle both
    if isinstance(input_data['generation'], str):
        input_data['generation'] = int(input_data['generation'])
    ```

    **Deploy**: Hot fix in 10 minutes

    **Long-term Fix** (Next Day):
    1. **Add input validation**:
       ```python
       def validate_input(data):
           # Type checking
           if not isinstance(data['generation'], (int, str)):
               raise ValueError("generation must be int or str")
           # Convert if needed
           return int(data['generation'])
       ```

    2. **Add integration tests**:
       ```python
       def test_handles_string_generation():
           input_data = {'generation': "1", ...}
           result = predict(input_data)
           assert result is not None
       ```

    3. **Add monitoring**:
       - Alert on input type errors
       - Track input schema changes

    **Who to Coordinate With**:
    - Upstream service team (why did they change format?)
    - Your team (review fix in morning)
    - Product (inform about outage)

    **Prevention**:
    - ✅ Input schema validation with Pydantic
    - ✅ Integration tests with upstream services
    - ✅ Schema change notifications
    - ✅ Gradual rollouts (catch this in 1% before 100%)

    **Postmortem Learning**:
    - We need API contracts between services
    - Input validation should be first line of defense
    - Need better testing for data type edge cases

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 5: Model Maintenance

    **ML Models are Not "Fire and Forget"**

    ### Retraining Strategies

    **Strategy 1: Scheduled Retraining**
    - **When**: Every N days/weeks/months
    - **Pros**: Predictable, planned
    - **Cons**: Might retrain when unnecessary
    - **Best for**: Stable domains (monthly retraining)

    **Strategy 2: Performance-Triggered**
    - **When**: Accuracy drops below threshold
    - **Pros**: Retrain only when needed
    - **Cons**: Might be too late (damage done)
    - **Best for**: Critical systems (retrain at 1% accuracy drop)

    **Strategy 3: Drift-Triggered**
    - **When**: Data drift detected
    - **Pros**: Proactive (before accuracy drops)
    - **Cons**: Might retrain too often
    - **Best for**: Fast-changing domains (daily monitoring)

    **Strategy 4: Hybrid**
    - **When**: Scheduled OR drift OR accuracy drop
    - **Pros**: Best of all worlds
    - **Cons**: Most complex
    - **Best for**: Production systems

    ### A/B Testing New Models

    **Never Replace Models Directly!**

    **Gradual Rollout Process**:

    ```
    1. Shadow Mode (0% traffic)
       - New model runs in parallel
       - Predictions logged but not served
       - Compare to old model
       - Duration: 1 week

    2. Canary (5% traffic)
       - Serve 5% of users with new model
       - Monitor metrics closely
       - Duration: 3 days
       - Rollback if issues

    3. Gradual Increase (25% → 50% → 100%)
       - Increase if metrics look good
       - Each step: 3-7 days
       - Rollback at any sign of problems

    4. Full Rollout (100% traffic)
       - Old model deprecated
       - Keep for quick rollback
    ```

    ### Model Deprecation

    **When to Retire a Model**:
    - ✅ New model clearly better (>5% improvement)
    - ✅ Running for 2+ weeks without issues
    - ✅ All stakeholders approve
    - ✅ Rollback plan documented

    **Deprecation Checklist**:
    - [ ] New model stable for 2+ weeks
    - [ ] Keep old model artifacts (for rollback)
    - [ ] Update documentation
    - [ ] Notify stakeholders
    - [ ] Monitor for 1 week after deprecation

    ### Technical Debt in ML Systems

    **Common ML Technical Debt**:

    1. **Pipeline Jungle**: Too many preprocessing steps
       - **Fix**: Simplify, document, test

    2. **Dead Experimental Code**: Old experiments never cleaned up
       - **Fix**: Regular code reviews, delete unused code

    3. **Glue Code**: Lots of custom code to make libraries work
       - **Fix**: Use standard interfaces, contribute to libraries

    4. **Configuration Debt**: Hard-coded values everywhere
       - **Fix**: Config files, environment variables

    5. **Data Debt**: Undocumented data dependencies
       - **Fix**: Data lineage tracking, documentation

    **Pay Down Debt Regularly**:
    - Dedicate 20% of sprint to refactoring
    - Quarterly "technical debt week"
    - Don't let debt compound
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Module 6 Summary

    ### Key Takeaways

    **Monitoring**:
    - ✅ Track 4 categories: Infrastructure, ML, Data, Business
    - ✅ Use P50/P95/P99 for latency (not max)
    - ✅ Alert on sustained issues, not noise
    - ✅ Dashboard should show trends, not just current values

    **Data Drift**:
    - ✅ Use KS test, Chi-square, or PSI
    - ✅ Monitor distributions daily
    - ✅ Retrain when drift >0.2 OR accuracy drops
    - ✅ A/B test new models before full rollout

    **Production Debugging**:
    - ✅ Be systematic: symptoms → hypothesis → test → fix → document
    - ✅ Three common issues: data pipeline, model loading, input distribution
    - ✅ Always have rollback plan
    - ✅ Write blameless postmortems

    **Model Maintenance**:
    - ✅ ML models degrade without maintenance
    - ✅ Use hybrid retraining (scheduled + drift + accuracy triggers)
    - ✅ Never replace models directly (gradual rollout)
    - ✅ Pay down technical debt regularly

    ### Self-Assessment

    You're production-ready when you can:

    - ✅ Set up comprehensive monitoring dashboard
    - ✅ Detect data drift using statistical tests
    - ✅ Debug the 3 most common production issues
    - ✅ Write an incident report
    - ✅ Plan retraining strategy
    - ✅ Roll out new models safely

    **If you can do all of these, you can keep ML systems healthy in production!**

    Most ML engineers learn this through painful production incidents.
    You now have the runbooks to avoid most of them.

    ---

    **Next**: Module 7 - Team Collaboration & Code Reviews

    Now that you can build and maintain ML systems, let's learn how to work
    effectively with other engineers through Git workflows and code reviews.
    """)
    return


if __name__ == "__main__":
    app.run()
