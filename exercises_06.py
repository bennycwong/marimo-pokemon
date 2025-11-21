"""
Module 6 Exercises: Production ML & Monitoring
==============================================

These exercises simulate real production incidents.
Practice debugging systematically - these scenarios happen in real companies!

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
    # Module 6 Exercises

    Complete these exercises to master production ML operations.

    ## Exercise 6.1: Debug Production Incidents (90 min)

    **Goal**: Practice systematic debugging of real production issues.

    **Instructions**: For each incident below:
    1. Read the symptoms and available data
    2. Form hypotheses (what could cause this?)
    3. Decide investigation steps
    4. Recommend immediate action
    5. Write prevention measures

    **Learning Objective**: Most ML engineers learn this through painful incidents. Learn it safely here!
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Incident #1: The Midnight Accuracy Drop

    **🚨 Alert at 2:47 AM**:
    ```
    P1 ALERT: ML Model Accuracy Drop
    - Current accuracy: 62%
    - Baseline: 87%
    - Duration: 2 hours
    - Affected users: ~5,000
    ```

    **Available Data**:

    **Logs (last 100 requests)**:
    ```python
    {
        'timestamp': '2025-01-20 02:45:00',
        'prediction': 'Fire',
        'confidence': 0.72,
        'features': {'hp': 90, 'attack': 85, 'defense': 65, ...},
        'actual_label': 'Water',  # Wrong!
        'model_version': 'v2.3.1'
    }
    # Pattern: 35/100 predictions are Fire type (should be ~15%)
    ```

    **Metrics Dashboard**:
    - Latency: 45ms P95 (normal)
    - Error rate: 0.02% (normal)
    - Traffic: 1,200 req/min (normal)
    - CPU: 45% (normal)
    - Memory: 62% (normal)

    **Recent Changes**:
    - No ML service deployments in 48 hours
    - Upstream data service deployed 3 hours ago (v1.8.2)
    - No infrastructure changes

    **TODO: Complete your incident analysis**

    ```
    ### Hypothesis 1:
    What could cause this?: [Your hypothesis]
    Evidence that supports this: [What you'd check]
    Likelihood: [High/Medium/Low]

    ### Hypothesis 2:
    What could cause this?: [Your hypothesis]
    Evidence that supports this: [What you'd check]
    Likelihood: [High/Medium/Low]

    ### Hypothesis 3:
    What could cause this?: [Your hypothesis]
    Evidence that supports this: [What you'd check]
    Likelihood: [High/Medium/Low]

    ### Investigation Plan (ordered by priority):
    1. [First thing to check - takes X minutes]
    2. [Second thing to check - takes X minutes]
    3. [Third thing to check - takes X minutes]

    ### Immediate Action (3am - what do you do NOW?):
    [ ] Rollback ML service to v2.3.0
    [ ] Rollback upstream service to v1.8.1
    [ ] Call upstream team
    [ ] Keep monitoring, investigate in morning
    [ ] Other: [Specify]

    Reasoning: [Why this action?]

    ### Root Cause (after investigation):
    [What actually broke?]

    ### Prevention Measures:
    1. [What to add to prevent this]
    2. [What test would have caught this]
    3. [What monitoring would detect this faster]
    ```

    <details>
    <summary>💡 Debugging Guide</summary>

    **Likely Hypotheses**:

    1. **Upstream service changed data format** (HIGH likelihood)
       - Timing matches (deployed 3 hours ago)
       - Check: Compare feature distributions before/after 00:00
       - Check: Look at upstream API response format

    2. **Data drift** (MEDIUM likelihood)
       - Sudden shift is unusual for drift
       - Check: Feature statistics vs training data
       - Check: Sample recent inputs

    3. **Model bug** (LOW likelihood)
       - No model deployment
       - But check: Model file integrity
       - Check: Model loading logs

    **Investigation Steps**:
    1. **Check upstream API changes** (5 min)
       - Contact on-call, ask what changed
       - Review their release notes

    2. **Compare feature distributions** (10 min)
       ```python
       before_midnight = get_features(time_range='23:00-00:00')
       after_midnight = get_features(time_range='02:00-03:00')

       for feature in features:
           print(f"{feature}: {before.mean():.2f} -> {after.mean():.2f}")
       ```

    3. **Sample predictions** (5 min)
       - Look at 20 failed predictions
       - What do they have in common?

    **Immediate Action**:
    - If upstream changed format: Call them to rollback OR
    - Quick fix in your code to handle new format
    - Don't rollback your ML service (not the problem)

    **Typical Root Cause**:
    "Upstream service v1.8.2 changed 'generation' field from int to string.
    Our preprocessing expected int, silently coerced string to 0.
    All cards with generation=0 classified as Fire (training data artifact)."

    **Prevention**:
    - Input validation with Pydantic (type checking)
    - Integration tests with upstream
    - Alert on unusual feature distributions
    - Gradual rollouts (1% → 100%)

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Incident #2: The Disappearing Predictions

    **🚨 Alert at 10:15 AM**:
    ```
    P0 ALERT: ML Service Error Rate Critical
    - Current error rate: 45%
    - Baseline: 0.05%
    - Errors per minute: 450
    - Users affected: ~10,000
    ```

    **Error Logs**:
    ```python
    FileNotFoundError: [Errno 2] No such file or directory: '/models/pokemon_classifier_v2.pkl'
    FileNotFoundError: [Errno 2] No such file or directory: '/models/pokemon_classifier_v2.pkl'
    FileNotFoundError: [Errno 2] No such file or directory: '/models/pokemon_classifier_v2.pkl'
    (repeating every 2 seconds)
    ```

    **Timeline**:
    - 09:30: Routine model update deployed (v2.3.1 → v2.4.0)
    - 09:45: All working normally
    - 10:00: First errors appear
    - 10:15: Error rate reaches 45%

    **Infrastructure**:
    - Kubernetes pods: 10 running
    - 5 pods showing errors, 5 pods working normally
    - Working pods using old model (v2.3.1)
    - Failing pods can't find new model (v2.4.0)

    **Deployment Process**:
    ```yaml
    # deployment.yaml
    - name: ml-service
      image: ml-service:v2.4.0
      volumeMounts:
        - name: model-storage
          mountPath: /models
      env:
        - name: MODEL_PATH
          value: /models/pokemon_classifier_v2.pkl
    ```

    **TODO: Complete your incident response**

    ```
    ### What's Wrong?
    Symptoms: [Describe what's broken]
    Root cause hypothesis: [What broke and why]

    ### Why Half Working, Half Broken?
    [Explain the pattern]

    ### Immediate Action (10:20am - 5 minutes after alert):
    [What do you do RIGHT NOW?]

    Options:
    [ ] Rollback deployment to v2.3.1
    [ ] Fix model file path
    [ ] Restart failing pods
    [ ] Scale down failing pods
    [ ] Other: [Specify]

    ### Quick Fix (10-15 minutes):
    [Step-by-step to restore service]

    ### Long-term Fix (next sprint):
    [How to prevent this forever]

    ### Postmortem Questions:
    1. Why didn't testing catch this? [Answer]
    2. How can we test deployments better? [Answer]
    3. What monitoring would detect this immediately? [Answer]
    ```

    <details>
    <summary>💡 Debugging Guide</summary>

    **Root Cause**: Model file path mismatch

    **What Happened**:
    - New code expects: `/models/pokemon_classifier_v2.pkl`
    - Actual file: `/models/pokemon_classifier_v2.4.0.pkl` (version in filename)
    - OR: Model file not uploaded to storage
    - OR: Permissions issue reading model file

    **Why Half Working**:
    - Kubernetes rolling deployment
    - Old pods (v2.3.1) still using old model file → working
    - New pods (v2.4.0) looking for new model file → failing
    - Rolling deployment paused at 50% due to errors

    **Immediate Action (10:20am)**:
    ```bash
    # ROLLBACK - fastest solution
    kubectl rollout undo deployment/ml-service

    # This will:
    # - Stop deploying new pods
    # - Scale down broken pods
    # - Scale up working pods
    # - Service restored in ~2 minutes
    ```

    **Investigation** (after rollback):
    ```bash
    # Check model storage
    kubectl exec ml-service-pod -- ls -la /models/
    # Expected: pokemon_classifier_v2.pkl
    # Actual: pokemon_classifier_v2.4.0.pkl (wrong name!)

    # OR: File missing entirely
    # OR: Permissions issue (can't read)
    ```

    **Long-term Fix**:
    1. **Consistent naming**:
       - Always use `model.pkl` (no version in filename)
       - OR: Parameterize filename in config

    2. **Health checks**:
       ```python
       @app.route('/health')
       def health():
           # Check model loads
           if not model_loaded:
               return 503  # Service Unavailable
           return 200
       ```

    3. **Deployment testing**:
       - Test model loading in CI/CD
       - Canary deployment (1 pod first, then scale)
       - Smoke tests before full rollout

    4. **Better error messages**:
       ```python
       if not os.path.exists(MODEL_PATH):
           logger.error(f"Model not found: {MODEL_PATH}")
           logger.error(f"Available files: {os.listdir('/models/')}")
           raise FileNotFoundError(...)
       ```

    **Postmortem Lessons**:
    - ✅ Health checks would catch this before deployment
    - ✅ Integration tests with actual model files
    - ✅ Canary deployments prevent 50% outage
    - ✅ Better logging helps debug faster

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Incident #3: The Slow Mo(del)

    **🚨 Alert at 3:30 PM**:
    ```
    P1 ALERT: ML Service Latency High
    - P95 latency: 2,400ms (target: 100ms)
    - P99 latency: 5,800ms (target: 500ms)
    - Users complaining: "App is so slow!"
    ```

    **Metrics**:
    - Latency increased gradually over 2 hours (14:00 → 16:00)
    - Request volume: Normal (1,000 req/min)
    - Error rate: Normal (0.05%)
    - CPU: 95% (was 40%)
    - Memory: 85% (was 60%)
    - Predictions still accurate

    **Logs show**:
    ```python
    INFO: Request received
    INFO: Model inference started
    WARNING: Model inference took 2.3s (expected <0.1s)
    INFO: Response sent

    INFO: Request received
    INFO: Model inference started
    WARNING: Model inference took 2.8s (expected <0.1s)
    INFO: Response sent
    ```

    **Recent Changes**:
    - No code deployments today
    - Traffic increased 10% (normal growth)
    - New feature "batch prediction" launched yesterday
      * Allows uploading 100 cards at once for bulk prediction

    **Investigation Data**:
    ```python
    # Profiling shows:
    - Loading model from disk: 0.01s (normal)
    - Preprocessing: 0.02s (normal)
    - Model inference: 2.5s (SLOW! was 0.05s)
    - Postprocessing: 0.01s (normal)

    # Request patterns:
    - 80% requests: Single card (1 prediction)
    - 20% requests: Batch (50-100 predictions)
    ```

    **TODO: Debug the slowdown**

    ```
    ### Symptoms Analysis:
    What's slow?: [Specific component]
    When did it start?: [Time correlation]
    What changed?: [Related changes]

    ### Hypothesis:
    Why is inference slow?: [Your theory]

    ### Investigation Steps:
    1. [What to check first]
    2. [What to check second]
    3. [What to check third]

    ### Root Cause:
    [After investigation, what's actually wrong?]

    ### Quick Fix (reduce latency in <1 hour):
    [What can you do NOW?]

    ### Proper Fix (next sprint):
    [Long-term solution]

    ### Performance Optimization Checklist:
    [ ] Batch predictions properly
    [ ] Cache model in memory
    [ ] Use numpy vectorization
    [ ] Optimize preprocessing
    [ ] Add timeout limits
    [ ] Scale horizontally
    [ ] Other: [Specify]
    ```

    <details>
    <summary>💡 Debugging Guide</summary>

    **Root Cause**: Batch requests processed inefficiently

    **What Happened**:
    - New feature allows 100-card batch predictions
    - Code processes them ONE AT A TIME in a loop:
      ```python
      # SLOW (current code)
      results = []
      for card in batch:
          result = model.predict(card)  # 0.05s × 100 = 5s!
          results.append(result)
      ```
    - Should be:
      ```python
      # FAST (vectorized)
      results = model.predict(batch)  # 0.1s for all 100!
      ```

    **Why Gradual**:
    - Batch feature launched yesterday
    - Adoption growing (20% of traffic now)
    - More batch requests → more slowness

    **Investigation Steps**:

    1. **Profile request types** (5 min):
       ```python
       # Check request sizes
       single_requests = requests[requests['batch_size'] == 1]
       batch_requests = requests[requests['batch_size'] > 1]

       print(f"Single latency: {single_requests['latency'].mean():.2f}s")  # 0.05s
       print(f"Batch latency: {batch_requests['latency'].mean():.2f}s")    # 2.5s
       ```
       → Batch requests are the problem!

    2. **Profile code** (10 min):
       ```python
       import time

       start = time.time()
       for card in batch:
           pred = model.predict(card)  # Called 100 times
       print(f"Loop time: {time.time() - start:.2f}s")  # 5.0s

       start = time.time()
       pred = model.predict(batch)  # Called once
       print(f"Batch time: {time.time() - start:.2f}s")  # 0.1s
       ```
       → Loop is 50x slower!

    **Quick Fix** (30 minutes):
    ```python
    # Old code
    def predict_batch(cards):
        results = []
        for card in cards:
            result = model.predict([card])  # SLOW
            results.append(result)
        return results

    # New code
    def predict_batch(cards):
        return model.predict(cards)  # FAST - vectorized
    ```

    Deploy this fix → latency back to normal

    **Proper Fix** (next sprint):

    1. **Async batch processing**:
       ```python
       # For very large batches (>100)
       async def predict_large_batch(cards):
           chunks = chunk_list(cards, size=50)
           results = await asyncio.gather(*[
               predict_batch(chunk) for chunk in chunks
           ])
           return flatten(results)
       ```

    2. **Request limits**:
       ```python
       if len(batch) > 100:
           return 400, "Max batch size: 100"
       ```

    3. **Load testing**:
       - Test with 100-card batches before launch
       - Measure latency under realistic load

    4. **Monitoring**:
       - Alert on P95 latency >200ms
       - Track latency by request type (single vs batch)

    **Performance Checklist Applied**:
    - ✅ Batch predictions properly (vectorize!)
    - ✅ Cache model in memory (already done)
    - ✅ Use numpy vectorization (model.predict handles this)
    - ✅ Add timeout limits (prevent 5s requests)
    - ✅ Scale horizontally if needed

    **Key Lesson**: Always vectorize ML operations. Never loop over predictions!

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Exercise 6.2: Design a Monitoring Dashboard (30 min)

    **Goal**: Create a comprehensive monitoring dashboard design.

    **Scenario**: You're setting up monitoring for the Pokemon type classifier in production.

    **Requirements**:
    - Dashboard for on-call engineers
    - Must show health at a glance
    - Drill down for debugging
    - Appropriate alerts

    **TODO: Design your dashboard**
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Dashboard Layout Design

    ```
    TODO: Sketch your dashboard structure

    ┌─────────────────────────────────────────┐
    │ System Health Overview                  │
    │                                         │
    │ [Design the top-level status here]     │
    │                                         │
    └─────────────────────────────────────────┘

    ┌──────────────────┬─────────────────────┐
    │ Infrastructure   │ ML Performance      │
    │                  │                     │
    │ [What metrics?]  │ [What metrics?]     │
    │                  │                     │
    │                  │                     │
    └──────────────────┴─────────────────────┘

    ┌──────────────────┬─────────────────────┐
    │ Data Quality     │ Business Impact     │
    │                  │                     │
    │ [What metrics?]  │ [What metrics?]     │
    │                  │                     │
    │                  │                     │
    └──────────────────┴─────────────────────┘
    ```

    ### Metrics to Track

    **TODO: Complete this table**

    | Metric | Why Track? | Alert Threshold | Chart Type |
    |--------|-----------|-----------------|------------|
    | P95 Latency | [Why?] | [When to alert?] | [Line/Bar/...?] |
    | Accuracy | [Why?] | [When to alert?] | [Line/Bar/...?] |
    | Error Rate | [Why?] | [When to alert?] | [Line/Bar/...?] |
    | [Add more] | | | |

    ### Alert Configuration

    **TODO: Design your alerts**

    ```
    Alert 1: [Name]
    - Trigger: [When]
    - Severity: [P0/P1/P2/P3]
    - Message: [What to say]
    - Runbook: [Link to debugging steps]

    Alert 2: [Name]
    - Trigger: [When]
    - Severity: [P0/P1/P2/P3]
    - Message: [What to say]
    - Runbook: [Link to debugging steps]

    [Add 3-5 alerts total]
    ```

    ### SLAs and SLOs

    **TODO: Define your targets**

    ```
    SLO (internal target):
    - Uptime: [99.X%]
    - P95 Latency: [Xms]
    - Accuracy: [X%]
    - Error Rate: [<X%]

    SLA (customer promise):
    - Uptime: [99.X%]
    - P95 Latency: [Xms]
    - Accuracy: [X%]

    Buffer: [SLO - SLA = buffer for catching issues early]
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Exercise 6.3: Write an Incident Report (30 min)

    **Goal**: Practice writing professional incident reports.

    **Scenario**: Use Incident #1 (Midnight Accuracy Drop) from Exercise 6.1

    **Assume**:
    - Root cause: Upstream service changed data format
    - Fixed by: Rolling back upstream service
    - Duration: 3 hours (03:00 - 06:00)
    - Impact: 5,000 users, 35% accuracy

    **TODO: Write the incident report**
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ## Incident Report Template

    **TODO: Complete this report**

    ```markdown
    # Incident Report: [Give it a descriptive name]

    **Date**: [Date]
    **Severity**: [P0/P1/P2]
    **Duration**: [X hours]
    **Impact**: [Users affected, business impact]
    **Status**: ✅ Resolved

    ---

    ## Executive Summary
    [2-3 sentences: What happened, what broke, how fixed, final outcome]

    ---

    ## Timeline (all times PST)

    **TODO: Fill in timeline**

    - 02:47: [Event]
    - 03:00: [Event]
    - 03:15: [Event]
    - [Continue timeline]

    ---

    ## Root Cause Analysis

    **What Happened**:
    [Detailed technical explanation]

    **Why It Happened**:
    [What allowed this to occur?]

    **Why It Wasn't Caught**:
    [What test/monitor would have caught this?]

    ---

    ## Impact Assessment

    **Users Affected**:
    - [Number and percentage]

    **Duration**:
    - [Time from start to resolution]

    **Financial Impact**:
    - [Estimate if possible]

    **Reputational Impact**:
    - [Customer complaints, social media, etc.]

    ---

    ## Resolution

    **Immediate Fix**:
    [What restored service]

    **Long-term Fix**:
    [What prevents recurrence]

    **Verification**:
    [How we confirmed it's fixed]

    ---

    ## Action Items

    **TODO: Create action items**

    - [ ] [Action 1] - Owner: [@person] - Due: [date]
    - [ ] [Action 2] - Owner: [@person] - Due: [date]
    - [ ] [Action 3] - Owner: [@person] - Due: [date]
    - [ ] [Action 4] - Owner: [@person] - Due: [date]

    ---

    ## What Went Well

    **TODO: Positive notes (blameless!)**

    - ✅ [Something that worked]
    - ✅ [Quick response]
    - ✅ [Good communication]

    ---

    ## What Could Be Improved

    **TODO: Learning opportunities (not blaming)**

    - 📝 [Better testing needed]
    - 📝 [Faster detection]
    - 📝 [Better communication]

    ---

    ## Prevention Measures

    **TODO: Specific, actionable prevention**

    1. **Technical**:
       - [What to build/add]

    2. **Process**:
       - [What to change in workflow]

    3. **Monitoring**:
       - [What to add to alerts]

    4. **Testing**:
       - [What tests to add]

    ---

    **Report Author**: [Your name]
    **Date**: [Today]
    **Reviewed By**: [Team lead]
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Exercise 6.4: Data Drift Detection (30 min)

    **Goal**: Implement drift detection for production monitoring.

    **Instructions**: Load the Pokemon data and simulate drift, then detect it.
    """)
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from scipy import stats

    # Load data
    DATA_PATH = Path("data/clean/pokemon_cards_clean_latest.csv")
    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df)} Pokemon cards")
    return DATA_PATH, df, np, pd, stats


@app.cell
def __(df, np):
    # TODO: Simulate production data with drift
    # Create a copy and modify some features

    df_production = None  # TODO: Create drifted version

    # TODO: Introduce drift in at least 2 features
    # Ideas:
    # - Increase attack values by 30%
    # - Shift generation distribution
    # - Add outliers to HP

    # Example starter code:
    # df_production = df.copy()
    # df_production['attack'] = df_production['attack'] * 1.3
    # ...

    print("TODO: Simulate production data with drift")
    return (df_production,)


@app.cell
def __(df, df_production, pd, stats):
    # TODO: Implement drift detection function

    def detect_drift(train_data: pd.DataFrame, prod_data: pd.DataFrame) -> dict:
        """
        Detect drift across all numeric features.

        Returns:
            Dict mapping feature name to drift results
        """
        results = {}

        # TODO: For each numeric feature:
        #   1. Perform KS test
        #   2. Calculate drift score
        #   3. Determine if drift is significant

        # Starter code:
        # numeric_features = train_data.select_dtypes(include=[np.number]).columns
        # for feature in numeric_features:
        #     ks_stat, p_value = stats.ks_2samp(
        #         train_data[feature],
        #         prod_data[feature]
        #     )
        #     results[feature] = {
        #         'ks_statistic': ks_stat,
        #         'p_value': p_value,
        #         'drift_detected': p_value < 0.05
        #     }

        return results

    # TODO: Run drift detection
    # drift_results = detect_drift(df, df_production)

    # TODO: Print results in a nice format
    # For each feature with drift, print severity and recommendation

    print("TODO: Implement and run drift detection")
    return (detect_drift,)


@app.cell
def __(mo):
    mo.md(r"""
    ### TODO: Drift Analysis Questions

    After implementing drift detection, answer:

    1. **Which features show drift?**
       ```
       [List features with significant drift]
       ```

    2. **What's the severity of drift?**
       ```
       Feature: [name] - p-value: [X] - Severity: [High/Medium/Low]
       ```

    3. **Should you retrain the model?**
       ```
       Decision: [Yes/No]
       Reasoning: [Why?]
       ```

    4. **What would you do immediately?**
       ```
       Action: [Monitor/Investigate/Retrain/Alert]
       Timeline: [When?]
       ```

    5. **How would you prevent this drift?**
       ```
       Prevention measures: [List]
       ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Bonus Exercise 6.5: Retraining Strategy (Optional - 20 min)

    **Scenario**: Your Pokemon type classifier is in production. Design a comprehensive retraining strategy.

    **TODO: Complete the strategy**

    ```
    ### Retraining Triggers

    **Scheduled Retraining**:
    - Frequency: [Every X days/weeks/months]
    - Reasoning: [Why this frequency?]

    **Drift-Triggered**:
    - Threshold: [PSI > X or p-value < Y]
    - Features to monitor: [List]
    - Action: [Immediate retrain or investigate first?]

    **Accuracy-Triggered**:
    - Threshold: [Accuracy < X%]
    - Duration: [Sustained for X hours/days]
    - Action: [Retrain urgently]

    **Manual Trigger**:
    - Events that require retraining: [List]
    - Examples: [Product launch, data fix, etc.]

    ### Retraining Process

    1. **Data Collection**:
       - How much data?: [Last X months]
       - Data quality checks: [List]

    2. **Training**:
       - Environment: [Where to train]
       - Duration: [Expected time]
       - Resources: [CPU/GPU needs]

    3. **Validation**:
       - Test on: [What data?]
       - Metrics: [What to check?]
       - Approval: [Who decides to deploy?]

    4. **Deployment**:
       - Strategy: [Canary/Blue-green/Rolling]
       - Rollout: [1% → 10% → 50% → 100%]
       - Monitoring: [What to watch]

    5. **Rollback Plan**:
       - Trigger: [When to rollback?]
       - Process: [How fast can you rollback?]
       - Communication: [Who to notify?]

    ### Cost-Benefit Analysis

    **Retraining Costs**:
    - Engineering time: [X hours]
    - Compute: [$Y]
    - Risk: [Deployment risk]

    **Benefits**:
    - Accuracy improvement: [Expected +X%]
    - Business value: [$Y]
    - User satisfaction: [+X points]

    **ROI**: [Calculate if worth it]
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Self-Assessment

    After completing these exercises, you should be able to:

    - ✅ Debug 3 common production incidents systematically
    - ✅ Design a comprehensive monitoring dashboard
    - ✅ Write professional incident reports
    - ✅ Detect data drift using statistical tests
    - ✅ Design a retraining strategy
    - ✅ Respond to alerts with confidence

    **If you can do all of these, you can keep ML systems healthy in production!**

    Most ML engineers learn production debugging through painful incidents at 2am.
    You've practiced the scenarios safely. When a real incident happens, you'll be ready.

    ---

    **Next**: Module 7 - Team Collaboration & Code Reviews

    Now that you can build, deploy, and maintain ML systems, let's learn how to work
    effectively with other engineers through Git workflows and code reviews.
    """)
    return


if __name__ == "__main__":
    app.run()
