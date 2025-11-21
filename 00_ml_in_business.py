"""
Module 0: ML in Business Context
=================================

Professional ML Engineering Onboarding Project

Learning Objectives:
- Evaluate when ML is appropriate vs alternatives
- Set realistic stakeholder expectations
- Define success metrics aligned with business goals
- Understand ML project lifecycle at companies
- Communicate effectively with non-technical teams

Duration: 2 hours

⚠️ START HERE: This module comes BEFORE data engineering because
understanding the business problem is more important than any technical skill.
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
    # Module 0: ML in Business Context

    **"Before you write any code, understand the problem"**

    ## Why This Module Comes First

    30% of ML projects fail not because of bad models, but because:
    - ML wasn't the right solution
    - Expectations weren't aligned
    - Success metrics were unclear
    - The business problem was misunderstood

    ## What You'll Learn

    1. **When to Use ML**: Decision framework for ML vs rules vs heuristics
    2. **Stakeholder Communication**: Set realistic expectations
    3. **Success Metrics**: Align technical and business goals
    4. **ML at Companies**: Understand the full project lifecycle
    5. **Common Pitfalls**: Learn from others' failures

    ## Industry Reality

    > "The hard part isn't building the model. It's figuring out what to build."
    > — Every ML team lead

    **Real Statistics**:
    - 87% of ML projects never make it to production (VentureBeat, 2019)
    - #1 reason: Solving the wrong problem
    - #2 reason: Misaligned expectations
    - #3 reason: Unclear success metrics

    **This module teaches you to avoid being part of that 87%.**
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 1: When to Use ML (and When Not To)

    **The Most Important Question**: Should we even use ML for this problem?

    ### Decision Framework

    ML is appropriate when ALL of these are true:
    1. ✅ The problem requires **pattern recognition** (not calculation)
    2. ✅ You have **data** (or can collect it)
    3. ✅ **Simple rules won't work** (too many exceptions)
    4. ✅ The problem is **worth the cost** (time, resources, maintenance)
    5. ✅ You can **measure success** objectively

    If ANY of these is false, reconsider!
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Real Examples: ML vs Rules vs Manual

    | Problem | Solution | Why? |
    |---------|----------|------|
    | **Calculate tax** | Rules/Code | Exact formula exists, no pattern recognition needed |
    | **Sort emails** | Rules + ML Hybrid | Some patterns (spam), some rules (from:boss = important) |
    | **Recommend movies** | ML | Millions of patterns, personal preferences, can't write rules |
    | **Detect fraud** | ML | Patterns evolve, attackers adapt, rules become obsolete |
    | **Price calculator** | Rules | If deterministic formula, use code |
    | **Price predictor** | ML | If depends on market, trends, many factors |
    | **Legal compliance** | Rules | Must be explainable and auditable |
    | **Medical diagnosis assist** | ML + Human | Pattern recognition + expert oversight |

    ### The "$100k Question"

    Before starting an ML project, answer:
    - What's the **current solution** and its cost?
    - What's the **expected improvement** from ML?
    - What does it **cost to build** and maintain?
    - **ROI** = (Improvement Value - Cost) / Cost

    **Example**:
    - Current: Manual review costs $50k/year, 80% accuracy
    - ML solution: Build cost $100k, maintenance $20k/year, 90% accuracy
    - Value of 10% improvement: $200k/year in saved errors
    - ROI Year 1: ($200k - $100k - $20k) / $120k = 67% ✅ Good!
    - ROI Year 2+: ($200k - $20k) / $20k = 900% ✅ Excellent!
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Interactive: Will Simple Rules Work?

    Before jumping to ML, ask these 5 questions:

    1. **Can I write down the rules?**
       - If yes → Use rules (cheaper, faster, maintainable)
       - If no → ML might be needed

    2. **Are there <20 rules?**
       - If yes → Use rules
       - If >100 rules → ML is better
       - If 20-100 → Consider hybrid

    3. **Do the rules change frequently?**
       - If static → Rules are fine
       - If change monthly → ML better (learns automatically)

    4. **Is this a pattern recognition problem?**
       - "Is this a cat?" → ML (visual patterns)
       - "Is this sum correct?" → Rules (deterministic)

    5. **Do I have examples to learn from?**
       - If no data → Rules (or collect data first)
       - If <100 examples → Rules probably better
       - If >10,000 examples → ML will likely outperform

    **Decision Tree**:
    ```
    Can write rules?
      ├─ Yes, <20 rules → Use rules
      └─ No, or >100 rules
           └─ Have >10k examples?
                ├─ Yes → Use ML
                └─ No → Collect data or use rules
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### 🎓 Socratic Question #1

    **Scenario**: Your team wants to build an ML model to automatically categorize support tickets.

    - Current system: Keywords + 50 if/else rules (85% accurate)
    - Proposed: ML classifier (estimated 92% accurate)
    - Volume: 10,000 tickets/month
    - Build cost: 3 engineers × 2 months = $60k
    - Improvement value: 7% fewer misrouted tickets = $30k/year saved

    **Questions**:
    1. Should you build the ML solution? Why or why not?
    2. What's the ROI?
    3. What's the break-even point?
    4. What hidden costs might you be missing?
    5. What would you recommend to your VP?

    **Think about it**, then expand below:

    <details>
    <summary>💡 Discussion Points</summary>

    **Analysis**:
    - Build cost: $60k
    - Annual savings: $30k
    - Break-even: 2 years
    - Maintenance: ~$10k/year (model updates, drift monitoring)
    - True ROI Year 1: ($30k - $60k - $10k) / $70k = -57% ❌ Negative!
    - True ROI Year 2: ($30k - $10k) / $10k = 200% ✅

    **Hidden Costs**:
    - Infrastructure for serving model
    - Monitoring and retraining
    - Debugging when wrong
    - Team learning curve

    **Recommendation Options**:
    1. **Don't build**: 2-year break-even is long, rules work "well enough"
    2. **Improve rules first**: Can you get to 90% with 20 more rules?
    3. **Build if**: Other benefits like "frees engineers for higher-value work"
    4. **Hybrid**: ML for confident cases, rules for edge cases

    **Best answer**: "Let's improve the rules first, then revisit if we hit diminishing returns."

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 2: Setting Stakeholder Expectations

    **The Second Most Important Skill**: Managing expectations

    ### Common Stakeholder Misconceptions

    | They Think | Reality |
    |------------|---------|
    | "AI can solve anything" | ML only works on specific, narrow problems |
    | "Just throw data at it" | Data quality > data quantity |
    | "It'll be 100% accurate" | All models are probabilistic |
    | "Build it in 2 weeks" | Real timeline: 2-6 months |
    | "Set it and forget it" | Models degrade, need monitoring |
    | "ML will replace humans" | ML assists, humans decide |

    ### How to Communicate Uncertainty

    **Bad**: "The model is 85% accurate"

    **Good**: "The model is correct 85% of the time. That means:
    - On 1000 predictions, expect ~150 errors
    - We're confident about 70% of predictions (>0.9 probability)
    - 30% need human review
    - Main failure mode: Confuses Fire and Fighting types"

    ### Translating "85% Accuracy" to Business Terms

    **For Different Audiences**:

    **To CEO**:
    - "Reduces manual review by 70%, saves $200k/year"
    - "Catches 9 out of 10 fraud cases vs 7 out of 10 currently"

    **To Product Manager**:
    - "Improves user satisfaction score from 3.5 to 4.2 stars"
    - "Reduces support tickets by 30%"

    **To Engineering Manager**:
    - "P95 latency <100ms, handles 1000 req/sec"
    - "Requires 2 hours/week maintenance, monthly retraining"

    **To Risk/Legal**:
    - "15% error rate means liability risk of $X/year"
    - "All predictions include confidence scores for audit trail"
    - "Human review for low-confidence predictions"
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Timeline Estimation: The Honest Version

    **Stakeholder Expectation**: 2 weeks
    **Reality**: 8-16 weeks

    **Typical ML Project Timeline**:

    | Phase | Time | Why This Long? |
    |-------|------|----------------|
    | **Discovery** | 1-2 weeks | Understand business problem, define success |
    | **Data Collection** | 2-6 weeks | Find data, get access, check quality |
    | **Data Cleaning** | 2-4 weeks | 60% of ML engineering time |
    | **EDA & Features** | 1-2 weeks | Understand patterns, engineer features |
    | **Model Training** | 1-2 weeks | Try many models, tune hyperparameters |
    | **Evaluation** | 1 week | Thorough testing, error analysis |
    | **Deployment** | 2-4 weeks | API, monitoring, integration |
    | **Monitoring** | Ongoing | Watch for drift, retrain |

    **Total**: 10-21 weeks for first production model

    ### How to Say "No" to Unrealistic Timelines

    **Stakeholder**: "Can you have this by next week?"

    **Bad Response**: "No, that's impossible"

    **Good Response**: "Let me break down what's needed:
    - Data validation: 3 days
    - Feature engineering: 5 days
    - Model training & evaluation: 5 days
    - Deployment: 3 days
    - **Fastest realistic timeline: 3 weeks**

    I can deliver a prototype in 1 week, but it won't be production-ready.
    What's more important: speed or quality?"

    **Pro Tip**: Always add a buffer. If you think 3 weeks, say 4 weeks. You'll hit unexpected issues.
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### 🎓 Socratic Question #2

    **Scenario**: Your VP asks: "Can we get 95% accuracy by next week?"

    Current model: 87% accurate, took 3 months to build

    **How do you respond?**

    <details>
    <summary>💡 Model Response</summary>

    **Good Response**:

    "Let me provide some context:

    1. **Current State**:
       - We're at 87% accuracy after 3 months of work
       - Getting from 87% to 95% is typically harder than 0% to 87%
       - This is the '90-90 rule': Last 10% takes 90% of the effort

    2. **What 95% Would Require**:
       - More training data (possibly 2-3x current size)
       - Better features (2-4 weeks of engineering)
       - More complex models (1-2 weeks testing)
       - Risk: May not be achievable with current data

    3. **Timeline Reality**:
       - Quick improvements: Can try new features (1 week, might get to 89%)
       - Meaningful jump: 4-6 weeks, might reach 92-93%
       - 95% target: 8-12 weeks, not guaranteed

    4. **Better Questions**:
       - Why 95%? What's the business need?
       - Current 87% - what's the cost of 13% errors?
       - If 95% is needed, should we scope the problem differently?
       - Can we solve 95% of cases perfectly, defer 5% to humans?

    **My Recommendation**:
    Let's have a 30-min meeting to understand the business driver for 95%.
    There might be a better solution than pushing accuracy higher."

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 3: Defining Success Metrics

    **The Third Critical Skill**: Know what you're optimizing for

    ### Business Metrics vs Technical Metrics

    **Common Mistake**: Optimizing the wrong thing

    | Stakeholder | Cares About | Technical Metric |
    |-------------|-------------|------------------|
    | **CEO** | Revenue impact | "Increases conversion by 15%" |
    | **Product** | User experience | "Reduces complaints by 30%" |
    | **Ops** | Efficiency | "Saves 100 hours/week of manual work" |
    | **Engineering** | Reliability | "99.9% uptime, <100ms latency" |
    | **Data Science** | Model quality | "AUC 0.93, F1 0.89" |

    **The Gap**: Data scientists optimize F1 score. CEO wants revenue.

    ### Translating Metrics

    **Exercise**: Connect technical to business

    | Business Goal | Technical Metric | Translation |
    |---------------|------------------|-------------|
    | Increase sales | Recommendation accuracy | "+5% CTR = $500k/year revenue" |
    | Reduce churn | Churn prediction recall | "Catch 80% of churners = 1000 saved customers" |
    | Cut costs | Automation rate | "80% auto → saves 100 hours/week = $250k/year" |
    | Improve quality | Defect detection precision | "False alarms cost $50/each, 90% precision = $100k/year saved" |

    ### The "North Star Metric"

    Every project needs ONE primary success metric that:
    1. Aligns with business value
    2. Is measurable
    3. Can be improved by ML

    **Bad**: "Improve the model"
    **Good**: "Reduce refund rate from 8% to 5% = $2M/year saved"

    **Example for Pokemon Card Classifier**:
    - Business goal: Accurate pricing for automated sales
    - North star: "95% of cards priced within $5 of market value"
    - Supporting: "Manual review rate <10%, avg review time <30 sec"
    - Technical: "Type classification accuracy >90%, confidence >0.8"
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Acceptable Error Rates

    **Critical Question**: How accurate is "good enough"?

    **It Depends on Context**:

    | Use Case | Acceptable Error | Why? |
    |----------|------------------|------|
    | Spam filter | 1-5% false positive | Angry if real email blocked |
    | Fraud detection | 5-10% false positive | Manual review is cheap |
    | Cancer screening | <1% false negative | Missing cancer = fatal |
    | Movie recommendation | 30-50% "error" | Low stakes, diversity is good |
    | Autonomous driving | <0.001% | Lives at stake |
    | Credit scoring | 5-10% error | Regulated, explainability required |

    ### The Cost Matrix

    **Make Errors Concrete**:

    | | Predict: Positive | Predict: Negative |
    |---|---|---|
    | **Actually: Positive** | ✅ True Positive (Good) | ❌ False Negative (Cost: $X) |
    | **Actually: Negative** | ❌ False Positive (Cost: $Y) | ✅ True Negative (Good) |

    **Example: Fraud Detection**
    - False Positive: Block legit transaction = Angry customer, lost sale ($50)
    - False Negative: Miss fraud = Company loses money ($500)
    - Ratio: 10:1 → Optimize for recall over precision

    **Example: Spam Filter**
    - False Positive: Block real email = Critical email missed ($1000 in lost opportunities)
    - False Negative: Spam gets through = Minor annoyance ($1)
    - Ratio: 1000:1 → Optimize for precision over recall
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### 🎓 Socratic Question #3

    **Scenario**: You're building a Pokemon card price predictor for automated sales.

    **Current manual process**:
    - Expert reviews each card: 5 min/card
    - 1000 cards/day = 83 hours/day
    - Expert time: $50/hour = $4,150/day
    - Expert accuracy: Within $5 on 95% of cards

    **Your ML system**:
    - Instant pricing
    - Two thresholds:
      * High confidence (>0.9): Auto-price
      * Medium confidence (0.7-0.9): Flag for quick review (30 sec)
      * Low confidence (<0.7): Full expert review (5 min)

    **Questions**:
    1. What's your north star metric?
    2. What accuracy do you need to beat the manual process?
    3. If the model is 90% accurate on 80% of cards (high confidence), what's the business value?
    4. What's your acceptable error rate?
    5. How would you measure success in the first month?

    <details>
    <summary>💡 Analysis Framework</summary>

    **North Star Metric**:
    "Reduce pricing cost per card from $5 to $1 while maintaining 95% accuracy"

    **Success Criteria**:
    - 70% of cards: High confidence, auto-priced (saves 4.2 min/card)
    - 20% of cards: Medium confidence, quick review (saves 4.5 min/card)
    - 10% of cards: Low confidence, full review (saves 0 min)
    - Weighted average: 70%×5 + 20%×4.5 + 10%×0 = 4.4 min saved per card
    - Cost savings: 4.4 min × $50/hr / 60 min = $3.67/card
    - Daily savings: 1000 cards × $3.67 = $3,670/day = $1.3M/year ✅

    **Acceptable Error Rate**:
    - High confidence predictions: 95% within $5 (match human)
    - Medium confidence: 90% within $5 (human catches the rest)
    - Overall: >92% within $5

    **First Month Success**:
    - Processed 20,000 cards
    - 72% high confidence, 93% accurate → ✅
    - 19% medium confidence, 88% accurate → ⚠️ Monitor
    - 9% low confidence → ✅ Below 10% target
    - Manual review time: Reduced by 68% → ✅
    - Pricing errors: 8% (vs 5% baseline) → ⚠️ Needs improvement

    **Recommendation**: Deploy with human oversight for 1 month, tune confidence thresholds, aim for <6% error rate.

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 4: ML Project Lifecycle at Companies

    **Understanding the Full Picture**

    ### The ML Project Lifecycle

    ```
    1. DISCOVERY (2-4 weeks)
       ├─ Problem definition
       ├─ Feasibility assessment
       ├─ Data availability check
       └─ Success criteria definition

    2. DATA (3-6 weeks)
       ├─ Data collection/access
       ├─ Data quality assessment
       ├─ Exploratory analysis
       └─ Feature engineering

    3. MODELING (2-4 weeks)
       ├─ Baseline models
       ├─ Model experiments
       ├─ Hyperparameter tuning
       └─ Error analysis

    4. EVALUATION (1-2 weeks)
       ├─ Offline validation
       ├─ A/B test design
       ├─ Model card creation
       └─ Stakeholder review

    5. DEPLOYMENT (2-4 weeks)
       ├─ API development
       ├─ Integration
       ├─ Monitoring setup
       └─ Gradual rollout

    6. MAINTENANCE (Ongoing)
       ├─ Performance monitoring
       ├─ Drift detection
       ├─ Periodic retraining
       └─ Incident response
    ```

    **Total**: 3-5 months to first production model

    ### Team Structures

    **Small Team (Startup, <5 people)**:
    - Everyone does everything
    - Full-stack ML engineers
    - Tools: Simple (pandas, sklearn, FastAPI)

    **Medium Team (10-50 people)**:
    - ML Engineers: Models + deployment
    - Data Engineers: Pipelines + infrastructure
    - ML Platform: Shared tools
    - Tools: MLflow, Airflow, feature stores emerging

    **Large Team (>50 people)**:
    - Research Scientists: Novel algorithms
    - ML Engineers: Production models
    - ML Platform: Infrastructure
    - Data Engineers: Data platform
    - Data Scientists: Analytics
    - Tools: Full enterprise stack

    **Where You'll Start**: Probably a medium team as an ML Engineer
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Common Failure Modes

    **Learn from Others' Mistakes**

    | Failure Mode | Example | Prevention |
    |--------------|---------|------------|
    | **Wrong Problem** | Built image classifier for low-res images | Validate problem first |
    | **Bad Data** | Trained on biased historical data | EDA before modeling |
    | **No Baseline** | "90% accuracy" but random guessing is 89% | Always compare to baseline |
    | **Overfitting** | 99% train, 60% test accuracy | Proper validation |
    | **Data Leakage** | Used future data to predict past | Careful feature engineering |
    | **Unmaintainable** | Custom code, no documentation | Production best practices |
    | **No Adoption** | Built model nobody uses | Involve stakeholders early |
    | **Drift** | Model degrades over time | Monitoring and retraining |

    ### Case Studies

    **✅ Success: Netflix Recommendations**
    - Clear value: More viewing = more subscriptions
    - Measurable: A/B tests show +10% engagement
    - Iterative: Hundreds of experiments
    - Aligned: Engineering, product, business all agree

    **❌ Failure: IBM Watson Health**
    - Wrong expectations: Promised cancer cure
    - Bad data: Limited training examples
    - No validation: Doctors didn't trust it
    - Result: $4B investment, shut down

    **🟨 Partial: Amazon Hiring AI**
    - Technical success: 85% accuracy
    - Ethical failure: Biased against women
    - Result: Cancelled despite working model
    - Lesson: Accuracy ≠ readiness
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### 🎓 Socratic Question #4

    **Scenario**: Your team is 6 weeks into an ML project. You discover:

    - The data has a 40% class imbalance you didn't notice
    - Your "95% accuracy" is actually worse than always predicting majority class
    - Stakeholders are expecting a demo next week
    - You'd need 4 more weeks to collect better data and retrain

    **Questions**:
    1. What went wrong in the process?
    2. How do you tell stakeholders the demo won't work?
    3. What do you propose as next steps?
    4. How could this have been prevented?

    <details>
    <summary>💡 How to Handle This</summary>

    **What Went Wrong**:
    1. ❌ No EDA before modeling (would have caught imbalance)
    2. ❌ No baseline comparison (majority class baseline)
    3. ❌ Wrong metric (accuracy instead of F1/recall)
    4. ❌ No checkpoints with stakeholders

    **How to Communicate**:

    "Hi team, I need to share an update on the ML project:

    **Situation**: We discovered a data quality issue that invalidates our current model.

    **What happened**:
    - The training data has a 60/40 class split
    - Our 95% accuracy is misleading - predicting majority class gets 60%
    - Our model only outperforms this baseline by 5%
    - This won't provide the value we expected

    **Why this happened**:
    - We moved too fast to modeling without thorough data analysis
    - We used the wrong success metric
    - This is my fault for not catching this earlier

    **What I learned**:
    - Always establish a baseline first
    - EDA is not optional
    - Regular checkpoints prevent late surprises

    **Options going forward**:

    Option A: Collect more minority class data (4 weeks)
    - Pros: Proper solution, better model
    - Cons: Timeline slip

    Option B: Use existing data with resampling (1 week)
    - Pros: Quick, might work
    - Cons: Less robust

    Option C: Pivot to different approach (2 weeks)
    - Pros: Might be faster
    - Cons: Unknown success rate

    **My recommendation**: Option B for next week's demo (proof of concept), then Option A for production (proper solution).

    **Timeline**:
    - Next week: Demo with resampled data (caveat: needs more data)
    - Next 4 weeks: Collect proper data
    - Week 6: Production-ready model

    **What do you think?**"

    **Key Points**:
    - ✅ Own the mistake
    - ✅ Explain clearly what happened
    - ✅ Provide options with tradeoffs
    - ✅ Give a clear recommendation
    - ✅ Ask for input

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Section 5: Communication Best Practices

    **How to Talk About ML to Different Audiences**

    ### The Pyramid Principle

    **Start with the conclusion**, then provide supporting details:

    **Bad**:
    "We tried Random Forest, XGBoost, and Neural Networks. Random Forest had 87% accuracy,
    XGBoost had 89%, Neural Networks had 88%. We tuned hyperparameters for 2 weeks.
    We used cross-validation with 5 folds. The precision was 0.86 and recall was 0.89..."

    **Good**:
    "**We should deploy the XGBoost model** (89% accurate, reduces errors by 15% vs current system).

    - **Value**: Saves $200k/year in misclassifications
    - **Risk**: Low - performs better than 3 alternatives we tested
    - **Timeline**: Ready to deploy next week
    - **Monitoring**: Alerts set up for accuracy <85%

    *Details available if needed: tried 3 model types, XGBoost best on all metrics*"

    ### Visualizations Matter

    **For Non-Technical Stakeholders**:
    - ❌ ROC curves, confusion matrices
    - ✅ Bar charts comparing current vs new system
    - ✅ Money saved over time
    - ✅ Error rate trends

    **For Technical Stakeholders**:
    - ✅ All the charts
    - ✅ Experiment tracking logs
    - ✅ Latency/throughput metrics

    ### Writing Project Briefs

    **1-Page ML Project Brief Template**:

    ```
    # [Project Name]

    ## Problem (2 sentences)
    Current state, pain point, opportunity

    ## Proposed Solution (2 sentences)
    ML approach, expected outcome

    ## Success Metrics (3 bullets)
    - Business metric: [Revenue, cost, satisfaction]
    - Technical metric: [Accuracy, latency]
    - User metric: [Adoption, engagement]

    ## Value (1 sentence)
    $X saved/earned, Y% improvement

    ## Approach (5 bullets)
    - Data source: [Where]
    - Model type: [What]
    - Timeline: [When]
    - Team: [Who]
    - Risks: [What could go wrong]

    ## Decision Needed
    [ ] Approve to proceed
    [ ] Need more information
    [ ] Reject with feedback
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Red Flags to Communicate Early

    **Don't Hide Problems - Escalate Early**

    | Red Flag | When to Escalate | How to Say It |
    |----------|------------------|---------------|
    | **Data quality issues** | Immediately | "We need to pause - 30% of data is unusable" |
    | **Timeline slipping** | As soon as you know | "We're behind by 2 weeks, here's why..." |
    | **Metric not improving** | After 3 failed attempts | "Accuracy stuck at 70%, we need to pivot" |
    | **Cost exceeding budget** | Before it's too late | "Infrastructure costs $X more than planned" |
    | **Ethical concerns** | Immediately | "Model shows bias, we can't ship this" |

    **Template for Bad News**:

    "**Update on [Project]**:

    **TL;DR**: [One sentence - the bad news]

    **What happened**: [Explanation]

    **Impact**: [Timeline, cost, or scope change]

    **What I've already done**: [Your mitigation efforts]

    **Options**:
    - Option A: [Fast but limited]
    - Option B: [Slow but thorough]

    **My recommendation**: [What you think, with reasoning]

    **Need from you**: [Decision or resources]"
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### 🎓 Socratic Question #5

    **Scenario**: You need to present to executives. You have 5 minutes.

    Your Pokemon type classifier:
    - 87% accurate (baseline: 65%)
    - Took 3 months, $80k cost
    - Will save 20 hours/week of manual work
    - Can process 10,000 cards/day
    - Has some issues with Fairy and Flying types

    **Create a 5-slide outline** (just bullet points):

    1. Slide 1: ___________
    2. Slide 2: ___________
    3. Slide 3: ___________
    4. Slide 4: ___________
    5. Slide 5: ___________

    <details>
    <summary>💡 Example Structure</summary>

    **Slide 1: Executive Summary**
    - Built Pokemon type classifier
    - 87% accurate (vs 65% baseline = 34% improvement)
    - Ready to deploy
    - ROI: $200k/year savings vs $80k cost = 2.5x Year 1

    **Slide 2: Business Value**
    - Saves 20 hours/week = 1000 hours/year
    - At $100/hr = $100k direct savings
    - Plus: Faster turnaround = happier customers = $100k in retention
    - Can scale to 10,000 cards/day (vs 100 manual)

    **Slide 3: How It Works** (Simple!)
    - Analyzes card stats (HP, attack, defense, etc.)
    - Compares to 800 training examples
    - Predicts type in <50ms
    - [Show a simple before/after diagram]

    **Slide 4: Limitations & Risks**
    - 13% error rate (130 errors per 1000 cards)
    - Main issue: Fairy/Flying types (working on it)
    - Mitigation: Confidence scores, human review for uncertain
    - Monitoring: Alert if accuracy drops below 80%

    **Slide 5: Recommendation & Next Steps**
    - **Deploy in 2 weeks** with 1-month pilot
    - Success criteria: >85% accuracy, <10% manual review
    - If successful: Scale to all cards
    - If issues: Rollback to manual (1-click)
    - Decision needed: Approve pilot?

    **Key Principles Used**:
    - ✅ Start with conclusion (deploy)
    - ✅ Business value first (money)
    - ✅ Simple explanation (no jargon)
    - ✅ Honest about limitations
    - ✅ Clear ask (decision)

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Module 0 Summary

    ### Key Takeaways

    **Before You Code**:
    1. ✅ Validate that ML is the right solution (vs rules/manual)
    2. ✅ Understand the business problem deeply
    3. ✅ Define success metrics (business + technical)
    4. ✅ Set realistic expectations with stakeholders
    5. ✅ Know the cost of errors (false positives vs negatives)

    **Communication Skills**:
    - 📊 Translate technical metrics to business value
    - 🎯 Start with conclusions, then details
    - ⚠️ Escalate problems early
    - 📈 Use appropriate visualizations for audience
    - 💬 Manage expectations proactively

    **Project Lifecycle**:
    - ⏰ Expect 3-5 months for first production model
    - 🔄 ML is iterative, not waterfall
    - 📉 30% of projects fail due to misalignment
    - 🛠️ Maintenance is ongoing, not one-time

    ### Self-Assessment

    You're ready for Module 1 when you can:

    - ✅ Evaluate whether ML is appropriate for a problem
    - ✅ Calculate ROI for an ML project
    - ✅ Set realistic timelines for ML work
    - ✅ Define success metrics that align business and technical goals
    - ✅ Communicate ML concepts to non-technical stakeholders
    - ✅ Write a 1-page project brief
    - ✅ Know when to escalate problems

    **If you can do all of these, you're ahead of 50% of ML practitioners!**

    Most people skip straight to modeling. You now understand the context that
    makes the difference between successful and failed ML projects.

    ---

    **Next**: Module 1 - Data Engineering Foundations

    Now that you understand WHEN and WHY to use ML, let's learn HOW to build reliable ML systems, starting with data.
    """)
    return


if __name__ == "__main__":
    app.run()
