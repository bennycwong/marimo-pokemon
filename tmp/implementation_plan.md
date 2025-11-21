# ML Course Implementation Plan
## Detailed Step-by-Step Execution Plan

**Project**: ML Engineering Course v2.0
**Approach**: Option 3 - Hybrid (Complete technical core, add context modules)
**Timeline**: 13-18 hours remaining work

---

## Phase 2: Module 0 - Business Context [NEXT]

### Estimated Time: 2-3 hours

### Step 1: Create `00_ml_in_business.py` (90 min)

**Structure**:
```python
# Section 1: When to Use ML (vs Rules vs Heuristics)
- Decision tree framework
- Examples: Email sorting (rules), Recommendations (ML), Tax calculation (rules)
- Cost-benefit analysis template
- "Will simple rules work?" checklist (5 questions)

# Section 2: Setting Stakeholder Expectations
- What 85% accuracy really means
- Timeline estimation (discovery 2w, data 4w, model 2w, prod 2w)
- Managing uncertainty ("we'll know after EDA")
- Red flags to communicate early

# Section 3: Defining Success Metrics
- Business metrics vs technical metrics table
- Converting accuracy to business value
- Acceptable error rate discussions
- North star metric examples (e.g., "reduce refunds by 20%")

# Section 4: ML at Companies
- Team structure (small/medium/large)
- ML project lifecycle flowchart
- Common failure modes (wrong problem, bad data, no adoption)
- Communication patterns

# Section 5: Case Studies
- Success: Netflix recommendations (clear value)
- Failure: IBM Watson Health (wrong expectations)
- Lessons learned
```

**Deliverables**:
- Interactive Marimo notebook
- 4-5 Socratic questions
- Industry examples (Netflix, Google, Amazon, Uber)

### Step 2: Create `exercises_00.py` (60 min)

**Exercises**:

1. **Exercise 0.1: Should You Use ML?** (20 min)
   - 5 scenarios (email spam, inventory, fraud, text translation, calculator)
   - For each: Yes/No/Maybe + justification
   - Decision framework practice

2. **Exercise 0.2: Project Proposal** (30 min)
   - Write 1-page ML project brief
   - Scenario: "Pokemon card price predictor"
   - Must include: problem, why ML, success metrics, timeline, risks
   - Template provided

3. **Exercise 0.3: Metrics Translation** (20 min)
   - Business requirement → Technical metric
   - Example: "Improve customer satisfaction" → "Reduce prediction errors on high-value items by 15%"
   - 3 scenarios to translate

4. **Checkpoint: Stakeholder Q&A** (30 min)
   - Answer tough questions from fictional stakeholders
   - VP: "Why can't we just use rules?"
   - CFO: "What's the ROI?"
   - CTO: "What if it fails?"

**Deliverables**:
- Exercise notebook with TODOs
- Rubric for self-assessment

### Step 3: Test & Validate (15 min)
```bash
uvx marimo check 00_ml_in_business.py
uvx marimo check exercises_00.py
```

### Step 4: Commit Phase 2 (5 min)
```bash
git add 00_ml_in_business.py exercises_00.py tmp/
git commit -m "Phase 2: Module 0 - ML in Business Context

- When to use ML vs alternatives
- Setting stakeholder expectations
- Business vs technical metrics
- Company ML lifecycle
- Case studies and exercises

Fills critical gap: Business alignment (15% → 80%)
"
```

---

## Phase 3: Module 6 - Production ML & Monitoring

### Estimated Time: 3-4 hours

### Step 1: Create `06_production_monitoring.py` (150 min)

**Structure**:
```python
# Section 1: What to Monitor in Production (30 min)
- Technical metrics (latency P50/P95/P99, throughput, errors)
- ML metrics (accuracy over time, confidence distribution)
- Data metrics (input distribution, feature drift)
- Business metrics (user satisfaction, revenue impact)
- Dashboard design best practices

# Section 2: Setting Up Alerts (30 min)
- When to alert (severity levels)
- Alert fatigue prevention
- SLAs and SLOs for ML (99.9% uptime, <100ms P95 latency)
- On-call rotation best practices

# Section 3: Data Drift Detection (45 min)
- Distribution shift visualization
- Statistical tests (KS test, Chi-square)
- Feature drift vs concept drift
- Practical implementation with Pokemon data
- When to retrain (trigger thresholds)

# Section 4: Production Debugging Runbook (45 min) ⭐ CRITICAL
- Incident response flowchart
- Scenario 1: Accuracy Drop
  * Check data distribution (compare to training)
  * Check upstream dependencies
  * Check feature engineering pipeline
  * Check model version loaded
  * Rollback decision tree
- Scenario 2: Latency Spike
  * Check model size
  * Check batch size
  * Check infrastructure (CPU/GPU)
  * Check external dependencies
- Scenario 3: Prediction Anomalies
  * Check input validation
  * Check preprocessing
  * Check model loading
  * Data type mismatches
- Creating incident reports
- Blameless postmortem template

# Section 5: Model Maintenance (30 min)
- Retraining schedules (calendar vs performance-triggered)
- A/B testing new versions
- Model deprecation process
- Technical debt in ML systems
```

**Deliverables**:
- Production monitoring notebook
- Debugging flowcharts
- Incident response checklist

### Step 2: Create `exercises_06.py` (90 min)

**Exercises**:

1. **Exercise 6.1: Debug Production Incidents** (60 min)
   - 3 simulated scenarios with logs
   - Diagnose root cause
   - Propose fix
   - Write incident report

2. **Exercise 6.2: Design Monitoring Dashboard** (30 min)
   - What metrics to track?
   - What alerts to set?
   - Dashboard layout mockup

3. **Exercise 6.3: Drift Detection** (30 min)
   - Implement simple drift detector
   - Test on shifted Pokemon data
   - Set retraining threshold

**Deliverables**:
- Exercise notebook with simulated incidents
- Incident report templates

### Step 3: Test & Validate (15 min)

### Step 4: Commit Phase 3 (5 min)

---

## Phase 4: Module 7 - Team Collaboration

### Estimated Time: 2-3 hours

### Step 1: Create `07_collaboration.py` (90 min)

**Structure**:
```python
# Section 1: Git for ML Projects (30 min)
- Branch naming (feature/exp-xgboost-tuning)
- What to commit (code yes, models no, data no)
- .gitignore for ML (.pkl, .h5, data/, .ipynb_checkpoints)
- PR descriptions for ML ("Improved accuracy by 3% using...")
- Semantic commits

# Section 2: ML Code Reviews (45 min)
- What to look for:
  * Data leakage patterns
  * Reproducibility (random seeds, versions)
  * Hardcoded paths/values
  * Missing validation
  * Performance issues
- Common ML code smells
- Giving constructive feedback
- Examples of good vs bad ML code

# Section 3: Working with Existing Systems (30 min)
- Reading ML codebases
- Understanding feature pipelines
- Safely modifying production models
- Documentation standards (docstrings, READMEs, model cards)
- Technical debt management

# Section 4: Team Communication (15 min)
- Slack/email etiquette for experiments
- Sharing results effectively
- Asking for help
- Pair programming for ML
```

**Deliverables**:
- Collaboration guide notebook
- Code review checklist
- Sample good/bad ML code

### Step 2: Create Sample ML Pull Requests (30 min)

**Create 3 PRs for review practice**:
1. **PR #1**: Feature engineering with data leakage bug
2. **PR #2**: Model training with reproducibility issues
3. **PR #3**: Deployment code with missing error handling

**Deliverables**:
- 3 sample PRs as markdown files
- Review checklist

### Step 3: Create `exercises_07.py` (45 min)

**Exercises**:

1. **Exercise 7.1: Review ML Pull Requests** (45 min)
   - Review 3 provided PRs
   - Identify issues
   - Suggest improvements
   - Write review comments

2. **Exercise 7.2: Refactor Messy Code** (30 min)
   - Given: Messy ML notebook
   - Task: Refactor into clean, modular code
   - Add documentation

3. **Exercise 7.3: Write PR Description** (15 min)
   - Describe your Module 5 changes as PR
   - Include: what, why, how, testing, metrics

**Deliverables**:
- Exercise notebook
- Sample messy code to refactor

### Step 4: Test & Validate (15 min)

### Step 5: Commit Phase 4 (5 min)

---

## Phase 5: Module 8 - Capstone Project

### Estimated Time: 4-5 hours

### Step 1: Prepare Capstone Dataset (45 min)

**Create Pokemon Card Price Dataset**:
- Extend existing dataset with price column
- Price = f(stats, rarity, type, generation) + noise
- Introduce realistic issues:
  * Some missing prices
  * Price inflation over time
  * Outliers (rare cards)
  * Data quality issues

**Deliverables**:
- `pokemon_cards_with_prices.csv`
- Data generation script
- Data dictionary

### Step 2: Create `08_capstone_project.py` (120 min)

**Structure**:
```python
# Introduction & Project Brief
- Problem: Predict Pokemon card prices
- Why ML: Price varies by many factors, rules are insufficient
- Success metric: RMSE < $5, 90% predictions within $10
- Business value: Automated pricing for 1000s of cards

# Deliverable Checklist
- [ ] Module 0: Business justification
- [ ] Module 1: Data validation pipeline
- [ ] Module 2: EDA + features
- [ ] Module 3: Train 3+ models
- [ ] Module 4: Evaluation report
- [ ] Module 5: Deployed API
- [ ] Module 6: Monitoring plan
- [ ] Module 7: PR-ready code

# Step-by-Step Guide
1. Load and validate data (Module 1 skills)
2. EDA - understand price distribution (Module 2)
3. Feature engineering (Module 2)
4. Train multiple models (Module 3)
5. Evaluate with appropriate metrics (Module 4)
6. Deploy best model (Module 5)
7. Create monitoring plan (Module 6)
8. Document everything (Module 7)

# Provided:
- Starter code template
- Evaluation rubric
- Submission checklist

# Evaluation Rubric (detailed)
```

**Deliverables**:
- Capstone project notebook
- Starter template
- Evaluation rubric

### Step 3: Create Evaluation Rubric (30 min)

**Rubric Breakdown**:
- Technical Correctness (40%)
  * Data pipeline (10%)
  * Feature engineering (10%)
  * Model training (10%)
  * Evaluation (10%)
- Communication (30%)
  * Documentation (10%)
  * Stakeholder reports (10%)
  * Code quality (10%)
- Production Readiness (20%)
  * Deployment (10%)
  * Monitoring plan (10%)
- Collaboration (10%)
  * Git usage (5%)
  * Code review readiness (5%)

**Deliverables**:
- `capstone_rubric.md`
- Self-assessment checklist

### Step 4: Test Capstone Flow (30 min)
- Do a quick run-through
- Ensure all steps are achievable
- Verify data works

### Step 5: Commit Phase 5 (5 min)

---

## Phase 6: Documentation & Polish

### Estimated Time: 2-3 hours

### Step 1: Update README.md (45 min)

**New Structure**:
```markdown
# Professional ML Engineering Course v2.0

## Course Structure (8 Modules + Capstone)

### Foundation
- Module 0: ML in Business Context (2h)
- Module 1: Data Engineering (3h)
- Module 2: EDA & Features (4h)

### Core ML
- Module 3: Model Training & Experimentation (4h)
- Module 4: Model Evaluation (3h)
- Module 5: Deployment (3h)

### Professional Skills
- Module 6: Production Monitoring (3h)
- Module 7: Team Collaboration (2.5h)

### Integration
- Module 8: Capstone Project (5-6h)

## What Makes This Complete?
[Compare to other courses]

## Quick Start
[Installation and first module]

## Learning Path
[Recommended timeline]
```

### Step 2: Create COURSE_OUTLINE.md (45 min)

**Complete Learning Path**:
- Week 1: Modules 0-2
- Week 2: Modules 3-4
- Week 3: Modules 5-6
- Week 4: Modules 7-8
- Detailed hour-by-hour breakdown
- Self-assessment checklist

### Step 3: Update ml_cheatsheet.md (30 min)
- Add sections for Modules 0, 6, 7
- Business context cheat sheet
- Monitoring cheat sheet
- Collaboration cheat sheet

### Step 4: Final Integration Testing (30 min)
```bash
# Test every single module
for i in 0{0..5} 0{6..8}; do
    echo "Testing module ${i}..."
    uvx marimo check ${i}*.py
done

# Test all exercises
for f in exercises_*.py; do
    echo "Testing $f..."
    uvx marimo check $f
done
```

### Step 5: Update Progress Tracker (15 min)
- Update `progress_tracker.md` for learners
- Add Module 0, 6-8 sections
- Update completion checklist

### Step 6: Commit Final Version (5 min)
```bash
git add .
git commit -m "🎉 ML Engineering Course v2.0 Complete

Complete 8-module professional ML engineering course with:
- ✅ Technical foundation (Modules 1-5)
- ✅ Business context (Module 0)
- ✅ Production operations (Module 6)
- ✅ Team collaboration (Module 7)
- ✅ End-to-end capstone (Module 8)

Total: 25-30 hours of comprehensive training
Coverage: 85-90% of company ML needs

Ready for production use!

🤖 Generated with Claude Code
"
```

---

## Success Criteria

### Module Completion:
- [ ] All 8 modules created
- [ ] All exercises created
- [ ] All modules pass validation
- [ ] No broken cells or errors

### Content Quality:
- [ ] Clear learning objectives
- [ ] Industry context throughout
- [ ] Socratic questions in each module
- [ ] Hands-on exercises
- [ ] Real-world scenarios

### Documentation:
- [ ] README.md updated
- [ ] COURSE_OUTLINE.md created
- [ ] ml_cheatsheet.md complete
- [ ] progress_tracker.md updated
- [ ] All TODOs are intentional

### Testing:
- [ ] Integration test passes
- [ ] Each module runs independently
- [ ] Exercises have clear instructions
- [ ] Capstone is achievable

---

## Risk Mitigation

### Potential Issues:
1. **Time Overrun**: Modules take longer than estimated
   - Mitigation: Focus on core content, skip nice-to-haves

2. **Scope Creep**: Adding too much content
   - Mitigation: Stick to learning outcomes, defer to v3.0

3. **Technical Issues**: Marimo compatibility issues
   - Mitigation: Test frequently, commit often

4. **Content Quality**: Not meeting 80-20 standard
   - Mitigation: Regular self-review against gap analysis

---

## Commit Strategy

### Commit After Each Phase:
- Phase 2: Module 0 complete
- Phase 3: Module 6 complete
- Phase 4: Module 7 complete
- Phase 5: Module 8 complete
- Phase 6: Documentation + Final

### Commit Message Format:
```
Phase X: [Module Name]

- [Key feature 1]
- [Key feature 2]
- [Key feature 3]

[Impact on gap analysis]

🤖 Generated with Claude Code
```

---

## Timeline Estimate

### Optimistic (13 hours):
- Module 0: 2h
- Module 6: 3h
- Module 7: 2h
- Module 8: 4h
- Documentation: 2h

### Realistic (16 hours):
- Module 0: 2.5h
- Module 6: 3.5h
- Module 7: 2.5h
- Module 8: 4.5h
- Documentation: 3h

### Pessimistic (20 hours):
- Module 0: 3h
- Module 6: 4.5h
- Module 7: 3h
- Module 8: 5.5h
- Documentation: 4h

**Target**: Complete within 1 week (2-3 work sessions)

---

## Next Action

**Immediately**: Start Phase 2 - Create Module 0

```bash
# Create the file
touch 00_ml_in_business.py

# Open in Marimo
uvx marimo edit 00_ml_in_business.py
```

**Focus**: When to use ML, stakeholder communication, business metrics

**Outcome**: Fills the #1 gap - business context and alignment

---

**Last Updated**: 2025-11-21
**Current Phase**: Phase 2 (Module 0) - Ready to start
**Overall Progress**: 62% complete (5/8 modules)
