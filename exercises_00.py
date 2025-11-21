"""
Module 0 Exercises: ML in Business Context
==========================================

These exercises reinforce business and communication skills for ML.
Complete each exercise - these skills are as important as technical skills!

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
    # Module 0 Exercises

    Complete these exercises to master the business side of ML.

    ## Exercise 0.1: Should You Use ML? (45 min)

    **Goal**: Develop intuition for when ML is appropriate vs alternatives.

    **Instructions**: For each scenario below, decide:
    1. Should you use ML? (Yes / No / Maybe)
    2. If not, what should you use instead?
    3. If yes, what data do you need?
    4. What's the expected ROI timeline?

    **Learning Objective**: Most problems DON'T need ML. Knowing when not to use it is as important as knowing when to use it.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Scenario A: Email Sorting

    **Context**:
    - Company receives 5,000 support emails/day
    - Currently: Manual sorting into 5 categories (sales, support, billing, refunds, other)
    - Takes 2 employees 4 hours/day = 8 hours/day total
    - Categories are clear: Sales emails have "purchase", "buy", "pricing"
    - But customers don't always use keywords

    **Current Rules-Based Attempt**:
    - 30 keyword rules (if contains "refund" → refunds category)
    - 75% accurate
    - Remaining 25% manually sorted

    **Questions**:
    1. Should you use ML for this?
    2. What's the current cost?
    3. What would ML cost to build and maintain?
    4. What's the expected ROI?
    5. What do you recommend?

    **TODO: Write your analysis below**
    ```
    Decision: [ML / Rules / Hybrid / Manual]

    Reasoning:
    - Current cost: [Calculate]
    - ML benefits: [List]
    - ML costs: [Estimate]
    - ROI: [Calculate]
    - Data availability: [Yes/No, how much]

    Recommendation:
    [Your recommendation with justification]

    Alternative approaches:
    [What else could work?]
    ```

    <details>
    <summary>💡 Analysis Hints</summary>

    **Cost Analysis**:
    - Manual sorting: 8 hrs/day × $30/hr × 250 days = $60k/year
    - ML build: 2 engineers × 3 weeks = ~$25k
    - ML maintenance: 5 hrs/month = $1.8k/year
    - Savings if 95% automated: Save 7.6 hrs/day = $57k/year
    - ROI Year 1: ($57k - $25k - $2k) / $27k = 111% ✅

    **Data Availability**:
    - Have historical emails with labels (from manual sorting)
    - Probably 10,000+ examples
    - Good for ML

    **Considerations**:
    - 75% with rules is actually not bad
    - Could improve rules to 85-90% for much cheaper
    - ML would get to 92-95%
    - Hybrid approach: Rules for clear cases, ML for ambiguous

    **Best Recommendation**:
    "Start by improving rules to 85% (1 week, low cost). If that's not good enough, then invest in ML for the remaining 15% of ambiguous cases."

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Scenario B: Inventory Forecasting

    **Context**:
    - Retail store chain (100 locations)
    - Currently: Store managers manually forecast next month's inventory
    - Based on "gut feel" and last year's data
    - 30% over-ordering (waste) and 20% under-ordering (lost sales)
    - Each location does $500k/month in sales
    - Waste costs ~$50k/month, lost sales ~$80k/month

    **Factors Affecting Inventory**:
    - Seasonality (holidays, weather)
    - Local events (concerts, sports games)
    - Economic trends
    - Past sales patterns
    - 50+ variables

    **Questions**:
    1. Is this a good ML use case? Why or why not?
    2. What's the current cost of bad forecasting?
    3. What would you need to build this?
    4. What's the potential value?
    5. What are the risks?

    **TODO: Write your analysis**
    ```
    Decision: [Should build ML / Should not]

    Current problem cost:
    - Waste: $___/month × 100 stores = $___/year
    - Lost sales: $___/month × 100 stores = $___/year
    - Total cost: $___/year

    ML Requirements:
    - Data: [What data is needed?]
    - Model: [What type?]
    - Timeline: [How long to build?]
    - Team: [Who needs to be involved?]

    Expected Improvement:
    - If ML reduces waste by X% and stock-outs by Y%
    - Annual value: $___

    ROI Calculation:
    [Show your math]

    Recommendation:
    [Deploy / Don't deploy / Pilot first]
    ```

    <details>
    <summary>💡 Analysis Hints</summary>

    **Problem Cost**:
    - Waste: $50k/mo × 100 × 12 = $60M/year
    - Lost sales: $80k/mo × 100 × 12 = $96M/year
    - **Total: $156M/year** - THIS IS HUGE

    **ML Value**:
    - If ML reduces waste by 50% and stock-outs by 50%
    - Savings: $78M/year
    - Even 10% improvement = $15.6M/year

    **ML Costs**:
    - Build: 3 data engineers + 2 ML engineers × 6 months = ~$500k
    - Infrastructure: $50k/year
    - Maintenance: $200k/year
    - **Total Year 1: $750k**

    **ROI**:
    - Even with conservative 10% improvement: $15.6M - $750k = $14.85M
    - ROI: $14.85M / $750k = 1980% ✅✅✅
    - Payback period: <1 month

    **Risks**:
    - Model wrong → Worse than humans? (Mitigate with gradual rollout)
    - Data quality issues? (Need clean historical data)
    - Store managers don't trust it? (Education + gradual adoption)

    **Recommendation**:
    "STRONG YES - Build this immediately. Even if we only improve by 5%, ROI is massive.
    Start with pilot in 10 stores, validate, then roll out to all 100."

    **This is a PERFECT ML use case**:
    - ✅ Pattern recognition problem
    - ✅ Lots of data
    - ✅ Rules won't work (50+ variables, interactions)
    - ✅ High value (millions)
    - ✅ Measurable success

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Scenario C: Legal Contract Review

    **Context**:
    - Law firm reviews 500 contracts/month
    - Each review takes a lawyer 2 hours = 1000 hours/month
    - Lawyer cost: $200/hour = $200k/month
    - Looking for specific clauses (liability, termination, payment terms)
    - Contracts are 20-100 pages, dense legal language

    **Proposed ML Solution**:
    - NLP model to extract key clauses
    - Flag risky terms
    - Auto-summarize contracts
    - Estimated 70% time savings

    **Questions**:
    1. Should you build this?
    2. What are the risks that matter most here?
    3. What would make you confident to deploy?
    4. What's the acceptable error rate?
    5. How would you validate it works?

    **TODO: Write your analysis**
    ```
    Technical Feasibility: [Yes/No + reasoning]

    Risk Analysis:
    - What happens if ML misses a risky clause? [Impact]
    - What happens if ML flags too many false alarms? [Impact]
    - Can you afford to be wrong? [Yes/No]

    Regulatory/Legal Concerns:
    - Can you use ML for legal work? [Research this]
    - Do you need human oversight? [Yes/No]
    - Liability if ML makes a mistake? [Who is responsible?]

    Deployment Strategy:
    [If you build this, how would you deploy it safely?]

    Recommendation:
    [Yes, build / No, don't build / Build with constraints]
    ```

    <details>
    <summary>💡 Analysis Hints</summary>

    **Technical Feasibility**:
    - ✅ NLP is good at extracting clauses
    - ✅ Summarization is doable
    - ⚠️ Legal language is complex, lots of edge cases
    - ⚠️ "Missing a risky clause" = potential lawsuit

    **Risk Assessment**:
    - Miss a risky clause → Client loses lawsuit → Firm liability = Millions
    - Too many false alarms → Lawyers ignore tool → Wasted investment
    - **This is HIGH STAKES**

    **Regulatory**:
    - Most jurisdictions: ML can ASSIST but not REPLACE lawyer judgment
    - Humans must review and sign off
    - ML tool is a "recommendation system", not "decision maker"

    **Acceptable Error Rate**:
    - False Negative (miss risky clause): <0.1% (can't afford misses)
    - False Positive (flag unnecessary): 10-20% ok (lawyers review anyway)
    - Must optimize for RECALL over precision

    **Deployment Strategy**:
    - Build as "first pass" tool, not replacement
    - ML highlights sections to review
    - Lawyers still read everything, but faster
    - Track: Do lawyers catch what ML misses?

    **Recommendation**:
    "Build as ASSISTANT TOOL, not replacement:
    - ML does first pass, highlights key sections
    - Lawyers review 100% of contracts (legally required anyway)
    - Measure success by time saved, not by automation rate
    - Conservative model: Prioritize recall (catch everything, false alarms OK)
    - ROI: $200k/mo × 30% time saved = $60k/month, $720k/year savings
    - Build cost: ~$200k, payback in 3-4 months"

    **Key Insight**: High-stakes domains require different thinking. Assist, don't replace.

    </details>
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Exercise 0.2: Write a Project Proposal (45 min)

    **Goal**: Practice communicating an ML project idea to stakeholders.

    **Scenario**: You want to build a Pokemon card **price predictor**.

    **Current State**:
    - Online marketplace sells Pokemon cards
    - 1000 new cards listed per day
    - Currently: Sellers manually price cards (often wrong)
    - Result: 20% of cards under-priced (lost revenue for sellers)
    - Result: 15% of cards over-priced (don't sell)

    **Your Idea**:
    - ML model predicts card price based on:
      * Type, stats, rarity, generation
      * Historical sales data
      * Current market trends
    - Suggest price to seller
    - Goal: 90% of suggestions within $5 of optimal price

    **Instructions**: Write a 1-page project brief using the template below.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ## Pokemon Card Price Predictor - Project Brief

    **TODO: Complete this template**

    ---

    ### Problem Statement (2-3 sentences)
    ```
    [What is the current problem?]
    [Who does it affect?]
    [Why does it matter?]
    ```

    ---

    ### Proposed Solution (2-3 sentences)
    ```
    [What are you building?]
    [How will it work?]
    [What will change for users?]
    ```

    ---

    ### Success Metrics

    **Business Metrics**:
    - [ ] Metric 1: [e.g., "Increase seller revenue by X%"]
    - [ ] Metric 2: [e.g., "Reduce unsold inventory by Y%"]
    - [ ] Metric 3: [e.g., "Improve seller satisfaction to Z stars"]

    **Technical Metrics**:
    - [ ] Accuracy: [e.g., "90% of predictions within $5"]
    - [ ] Coverage: [e.g., "Provide predictions for 95% of cards"]
    - [ ] Latency: [e.g., "<1 second response time"]

    **How we'll measure**: [Describe A/B test or pilot]

    ---

    ### Business Value

    **Current Cost of Problem**:
    ```
    - Under-priced cards: [Calculate lost revenue]
    - Over-priced cards: [Calculate lost sales]
    - Seller time spent pricing: [Calculate]
    - Total cost: $___/year
    ```

    **Expected Improvement**:
    ```
    - Reduce under-pricing by X% = $___/year
    - Reduce over-pricing by Y% = $___/year
    - Save Z hours of seller time = $___/year
    - Total value: $___/year
    ```

    **ROI**:
    ```
    - Build cost: $___
    - Maintenance: $___/year
    - Net value Year 1: $___
    - ROI: ___%
    ```

    ---

    ### Approach

    **Data Sources**:
    - [ ] [List what data you need and where it comes from]
    - [ ] [e.g., "Historical sales data (2 years, 50k transactions)"]
    - [ ] [e.g., "Card attributes (database)"]

    **Model Type**:
    - [ ] [What kind of model? e.g., "Regression model (XGBoost)"]
    - [ ] [Why this approach?]

    **Timeline**:
    - Weeks 1-2: [Phase]
    - Weeks 3-4: [Phase]
    - Weeks 5-6: [Phase]
    - Weeks 7-8: [Phase]
    - **Total: ___ weeks to production**

    **Team**:
    - [ ] [Who needs to be involved?]
    - [ ] [e.g., "1 ML Engineer, 1 Data Engineer, 1 Product Manager"]

    **Risks & Mitigation**:
    1. **Risk**: [What could go wrong?]
       - **Mitigation**: [How will you handle it?]
    2. **Risk**: [e.g., "Model prices too low, revenue loss"]
       - **Mitigation**: [e.g., "Start with suggestions only, not auto-pricing"]
    3. **Risk**: [e.g., "Sellers don't trust ML prices"]
       - **Mitigation**: [e.g., "Show confidence scores, explain reasoning"]

    ---

    ### Alternatives Considered

    **Option A: Manual Pricing** (Current state)
    - Pros: [List]
    - Cons: [List]

    **Option B: Simple Rules** (e.g., price = average of similar cards)
    - Pros: [List]
    - Cons: [List]

    **Option C: ML Model** (Proposed)
    - Pros: [List]
    - Cons: [List]

    **Why ML is Best**: [Justify]

    ---

    ### Decision Needed

    [ ] **Approve**: Proceed with building ML price predictor
    [ ] **Need More Info**: [What questions do you have?]
    [ ] **Reject**: [Why?]

    ---

    ### Next Steps (if approved)

    **Week 1**:
    - [ ] [e.g., "Collect and validate historical sales data"]

    **Week 2**:
    - [ ] [e.g., "Build baseline model"]

    **Week 3**:
    - [ ] [Next step]

    **Checkpoints**:
    - Week 4 demo: [What you'll show]
    - Week 6 evaluation: [What you'll measure]
    - Week 8 launch: [Deployment plan]

    ---

    **Project Owner**: [Your Name]
    **Date**: [Today's Date]
    **Contact**: [How to reach you with questions]

    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Exercise 0.3: Metric Translation Workshop (30 min)

    **Goal**: Practice converting business requirements to technical metrics.

    **Instructions**: For each business requirement below:
    1. Translate to a technical ML metric
    2. Explain why that metric captures the business need
    3. Specify how you'd measure it

    **Learning Objective**: Business and technical teams must speak the same language.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Translation Exercise

    **Business Requirement #1**: "Improve customer satisfaction"

    ```
    Technical Metric: [What would you measure?]

    Why this captures the business need: [Explanation]

    How to measure:
    - Training: [How do you train for this?]
    - Validation: [How do you validate?]
    - Production: [How do you monitor?]

    Example:
    - Metric: "Reduce customer support tickets by 30%"
    - OR: "Increase recommendation click-through rate from 5% to 8%"
    - OR: "Reduce time to resolution from 24hrs to 6hrs"

    Why: More satisfied customers = fewer complaints, more engagement
    ```

    ---

    **Business Requirement #2**: "Reduce operational costs"

    ```
    Technical Metric: [What would you optimize?]

    Translation: [How does this reduce costs?]

    Measurement: [How to track impact?]

    Example metrics to consider:
    - Automation rate (% of tasks automated)
    - False positive rate (wasted manual review)
    - Processing time per item
    ```

    ---

    **Business Requirement #3**: "Ensure fair treatment of all customers"

    ```
    Technical Metric: [What fairness metrics?]

    Why this matters: [Explain business + ethical concerns]

    Measurement: [How to validate fairness?]

    Consider:
    - Demographic parity
    - Equal opportunity
    - Predictive parity
    - Which is most important for your use case?
    ```

    ---

    **Business Requirement #4**: "We need this to be explainable to regulators"

    ```
    Technical Approach: [What does this mean for your model choice?]

    Tradeoffs: [What do you give up?]

    Solution: [How to balance performance vs explainability?]

    Think about:
    - Model types (linear vs neural network vs trees)
    - SHAP values, LIME, feature importance
    - Model cards and documentation
    ```

    ---

    **Business Requirement #5**: "Must handle 10x growth in next 2 years"

    ```
    Technical Requirements: [What does this mean for architecture?]

    Metrics to track: [What tells you you're ready?]

    Preparation: [What do you build now for future scale?]

    Consider:
    - Inference latency at scale
    - Training time as data grows
    - Infrastructure costs
    - Model retraining frequency
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Exercise 0.4: Stakeholder Q&A Roleplay (30 min)

    **Goal**: Practice answering tough questions from stakeholders.

    **Scenario**: You're presenting your Pokemon type classifier to different stakeholders.
    Your model is 87% accurate, took 3 months to build.

    **Instructions**: Write how you'd respond to each question below.

    **Learning Objective**: Communication is as important as technical skill.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Stakeholder Questions

    ---

    **VP of Product** asks:
    "Why did this take 3 months? Can't we just use an API or buy something?"

    **Your Response**:
    ```
    [TODO: Write your answer]

    Tips:
    - Acknowledge the question
    - Explain what took time (data cleaning, experimentation)
    - Compare to alternatives (buying may not fit our data)
    - Show value delivered
    ```

    ---

    **CFO** asks:
    "What's the ROI on this project? How do I know it's worth the investment?"

    **Your Response**:
    ```
    [TODO: Write your answer]

    Include:
    - Build cost: $___
    - Annual value: $___
    - Payback period: ___
    - Ongoing costs: $___
    ```

    ---

    **CTO** asks:
    "What happens if this crashes in production? What's your monitoring strategy?"

    **Your Response**:
    ```
    [TODO: Write your answer]

    Address:
    - Graceful degradation (fallback to rules)
    - Monitoring metrics
    - Alert thresholds
    - On-call plan
    ```

    ---

    **Head of Customer Support** asks:
    "My team has to deal with the errors. What happens when your model is wrong?"

    **Your Response**:
    ```
    [TODO: Write your answer]

    Discuss:
    - Error rate (13%)
    - Error handling process
    - Customer communication
    - Continuous improvement
    ```

    ---

    **Legal/Compliance** asks:
    "How do you ensure this doesn't discriminate against certain card types?
    Can you explain how it makes decisions?"

    **Your Response**:
    ```
    [TODO: Write your answer]

    Cover:
    - Bias testing
    - Fairness metrics
    - Explainability (feature importance)
    - Audit trail
    ```

    ---

    **Engineer on Another Team** asks:
    "What happens when I need to update the card database schema?
    Will your model break?"

    **Your Response**:
    ```
    [TODO: Write your answer]

    Explain:
    - Input validation
    - Schema versioning
    - Breaking change handling
    - Communication plan
    ```

    ---

    **CEO** asks:
    "What can we build next with this technology?"

    **Your Response**:
    ```
    [TODO: Write your answer]

    Suggest:
    - Extensions of current system
    - Related problems
    - Bigger opportunities
    - Realistic timelines
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Exercise 0.5: Red Flag Communication (20 min)

    **Goal**: Practice escalating problems early and professionally.

    **Scenario**: Things are going wrong. Practice communicating bad news.

    **Instructions**: Write the email/Slack message you'd send for each scenario.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Red Flag Scenarios

    ---

    **Scenario A**: You're 4 weeks into an 8-week project. You just discovered the data quality is much worse than expected. You need 3 more weeks to clean it properly. This will blow the deadline.

    **Your Message to Manager**:
    ```
    Subject: [TODO]

    [TODO: Write the message]

    Structure to follow:
    - TL;DR (one sentence with the bad news)
    - What happened (explanation)
    - Impact (timeline, scope, cost)
    - What you've already done
    - Options going forward
    - Your recommendation
    - What you need (decision, resources, etc.)
    ```

    ---

    **Scenario B**: Your model is showing bias - it's 95% accurate on Fire types but only 65% accurate on Fairy types. You're supposed to demo tomorrow. You don't have a fix yet.

    **Your Message to Stakeholders**:
    ```
    Subject: [TODO]

    [TODO: Write the message]

    Be honest about:
    - The bias discovered
    - Why it happened
    - Why it's a problem
    - What you're doing about it
    - What you'll show in tomorrow's demo
    ```

    ---

    **Scenario C**: In production for 1 week, your model's accuracy has dropped from 87% to 71%. You don't know why yet. Users are complaining.

    **Your Incident Report**:
    ```
    Subject: [TODO]

    [TODO: Write the incident alert]

    Include:
    - What's happening (symptoms)
    - Impact (how many users affected)
    - Current status (investigating / mitigated / resolved)
    - Immediate actions taken
    - ETA for resolution
    - Updates: How often will you communicate?
    ```

    ---

    **Scenario D**: Your ML project is technically working but users aren't adopting it. Only 10% usage after 1 month. You need to pivot.

    **Your Pivot Proposal**:
    ```
    Subject: [TODO]

    [TODO: Write the proposal]

    Address:
    - What you built
    - Why adoption is low (user feedback)
    - Proposed changes
    - Resource needs
    - Success metrics for v2
    ```
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Bonus Exercise 0.6: Build vs Buy Analysis (Optional - 30 min)

    **Scenario**: Your manager asks: "Why don't we just use OpenAI's API or buy a solution?"

    **Assignment**: Write a build vs buy analysis for the Pokemon type classifier.

    **Consider**:

    ### Option A: Build Custom (Your proposal)
    - Costs: [Development, infrastructure, maintenance]
    - Benefits: [Custom to our data, full control, IP ownership]
    - Risks: [Time, expertise needed, maintenance burden]
    - Timeline: [8 weeks to production]

    ### Option B: Buy SaaS Solution
    - Costs: [Research potential vendors and pricing]
    - Benefits: [Fast deployment, professional support]
    - Risks: [Doesn't fit our data, vendor lock-in, recurring costs]
    - Timeline: [2 weeks to integrate]

    ### Option C: Use General AI API (e.g., GPT Vision)
    - Costs: [API costs per call]
    - Benefits: [Zero development, works immediately]
    - Risks: [Expensive at scale, accuracy unknown, data privacy]
    - Timeline: [1 week proof of concept]

    ### Your Recommendation Matrix:

    | Criteria | Build | Buy | API | Winner |
    |----------|-------|-----|-----|--------|
    | **Cost (Year 1)** | $80k | $50k | $120k | Buy |
    | **Cost (Year 3)** | $110k | $150k | $360k | Build |
    | **Accuracy** | 87% | 75%? | 82%? | Build |
    | **Customization** | High | Low | None | Build |
    | **Time to Deploy** | 8 weeks | 3 weeks | 1 week | API |
    | **Maintenance** | High | Low | None | Buy |

    **Final Recommendation**: [Which option and why?]
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Self-Assessment

    After completing these exercises, you should be able to:

    - ✅ Evaluate 5+ scenarios for ML appropriateness
    - ✅ Calculate ROI for ML projects
    - ✅ Write a professional project proposal
    - ✅ Translate business requirements to technical metrics
    - ✅ Answer tough stakeholder questions confidently
    - ✅ Communicate bad news professionally
    - ✅ Make build vs buy decisions

    **If you can do all of these, you have business skills that 80% of ML practitioners lack!**

    Most ML education focuses only on technical skills. You now have the context
    and communication skills that separate good ML engineers from great ones.

    ---

    **Next**: Module 1 - Data Engineering Foundations

    Now that you can evaluate whether to use ML and communicate with stakeholders,
    let's learn how to build reliable ML systems, starting with data quality.
    """)
    return


if __name__ == "__main__":
    app.run()
