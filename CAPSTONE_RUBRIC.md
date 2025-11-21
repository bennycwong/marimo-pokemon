# Capstone Project Rubric

## Pokemon Card Price Prediction - Self-Assessment Guide

Rate yourself 1-5 on each criterion (5 = Excellent, 1 = Needs Work)

---

## 1. Business Context (Module 0) - 15%

### Problem Framing
- [ ] 1 - No business context provided
- [ ] 2 - Vague problem statement
- [ ] 3 - Clear problem but missing stakeholders or success criteria
- [ ] 4 - Well-defined problem with stakeholders and metrics
- [ ] 5 - **Excellent**: Clear problem, stakeholders, use cases, and constraints documented

**Questions to answer**:
- Who will use this model and how?
- What decision does it help them make?
- What does success look like (business metrics)?
- What error rate is acceptable?

### ROI Analysis
- [ ] 1 - No ROI calculation
- [ ] 2 - ROI mentioned but not calculated
- [ ] 3 - Basic cost-benefit analysis
- [ ] 4 - Detailed ROI with costs and expected value
- [ ] 5 - **Excellent**: Comprehensive ROI with timeline, risks, and sensitivity analysis

**Your Score**: _____ / 10 points

---

## 2. Data Engineering (Module 1) - 15%

### Data Validation
- [ ] 1 - No data validation
- [ ] 2 - Basic checks (null values only)
- [ ] 3 - Type checks and range validation
- [ ] 4 - Pandera schema with constraints
- [ ] 5 - **Excellent**: Comprehensive schema + custom validators + quality reports

**Must have**:
- Pandera schema for all columns
- Range checks (e.g., hp between 1-300)
- Type validation
- Uniqueness constraints

### Reproducibility
- [ ] 1 - Code doesn't run
- [ ] 2 - Runs but not reproducible (no seeds)
- [ ] 3 - Basic reproducibility (seeds set)
- [ ] 4 - Functions with clear interfaces
- [ ] 5 - **Excellent**: Pipeline with version control, configs, and documentation

**Must have**:
- `load_and_validate_data()` function
- Random seeds set
- Train/val/test split documented
- Test set never touched until final evaluation

### Data Quality
- [ ] 1 - No quality checks
- [ ] 2 - Checked for missing values only
- [ ] 3 - Missing values + outliers
- [ ] 4 - Missing values + outliers + distributions + relationships
- [ ] 5 - **Excellent**: Comprehensive quality report with visualizations and decisions documented

**Your Score**: _____ / 15 points

---

## 3. Feature Engineering (Module 2) - 15%

### Feature Design
- [ ] 1 - Used raw features only
- [ ] 2 - Added 1-2 simple features
- [ ] 3 - Added 3-5 thoughtful features
- [ ] 4 - Added 5+ features with clear rationale
- [ ] 5 - **Excellent**: Systematic feature creation with documentation and domain knowledge applied

**Must have**:
- At least 3 engineered features
- Feature documentation table
- Rationale for each feature

### Leakage Prevention
- [ ] 1 - Clear data leakage present (e.g., used target in features)
- [ ] 2 - Possible leakage not checked
- [ ] 3 - Basic leakage checks performed
- [ ] 4 - Explicit leakage checks for all features
- [ ] 5 - **Excellent**: Documented leakage check for every feature + time-based validation

**Red flags**:
- Using `price_usd` to create features
- Fitting scalers on full dataset (train + test)
- Using future information
- Aggregating across train and test sets

### Implementation
- [ ] 1 - Code doesn't work
- [ ] 2 - Works but not in a pipeline
- [ ] 3 - Basic pipeline with ColumnTransformer
- [ ] 4 - Clean pipeline with custom transformers
- [ ] 5 - **Excellent**: Reusable pipeline with proper fit/transform separation

**Your Score**: _____ / 15 points

---

## 4. Model Training (Module 3) - 15%

### Baseline & Comparison
- [ ] 1 - Tried only one model
- [ ] 2 - Tried 2 models
- [ ] 3 - Tried 3+ models with comparison
- [ ] 4 - Systematic comparison with cross-validation
- [ ] 5 - **Excellent**: Baseline + 4+ models + cross-validation + documented decision process

**Must have**:
- Baseline model (predict mean)
- At least 3 different algorithms
- Cross-validation (not just single train/val split)
- Comparison table with metrics

### Hyperparameter Tuning
- [ ] 1 - No tuning (used defaults)
- [ ] 2 - Manual tuning without process
- [ ] 3 - Grid search on small grid
- [ ] 4 - Randomized or grid search with CV
- [ ] 5 - **Excellent**: Systematic tuning + documented search space + computational cost considered

**Must have**:
- Used GridSearchCV or RandomizedSearchCV
- Tuned at least 3 hyperparameters
- Used cross-validation
- Best parameters documented

### Experiment Tracking
- [ ] 1 - No tracking
- [ ] 2 - Manual notes
- [ ] 3 - Spreadsheet or notebook outputs
- [ ] 4 - MLflow or similar tool
- [ ] 5 - **Excellent**: Full experiment tracking with reproducible configs and model versioning

**Your Score**: _____ / 15 points

---

## 5. Model Evaluation (Module 4) - 15%

### Metric Selection
- [ ] 1 - Used only accuracy or default metric
- [ ] 2 - Used 1-2 metrics without justification
- [ ] 3 - Used multiple metrics with basic justification
- [ ] 4 - Chose metrics aligned with business goals
- [ ] 5 - **Excellent**: Multiple metrics with clear business justification and trade-off analysis

**Must have**:
- At least 3 metrics (MAE, RMSE, R²)
- Explanation of why each metric matters
- Primary metric identified

### Error Analysis
- [ ] 1 - No error analysis
- [ ] 2 - Looked at overall metrics only
- [ ] 3 - Analyzed errors by category
- [ ] 4 - Error analysis + residual plots + identified patterns
- [ ] 5 - **Excellent**: Deep error analysis with failure modes identified and improvement suggestions

**Must have**:
- Residual plots
- Error distribution by category (type, rarity)
- Identified worst predictions
- Hypotheses for why errors occur

### Model Card
- [ ] 1 - No documentation
- [ ] 2 - Basic README
- [ ] 3 - Model card with key sections
- [ ] 4 - Complete model card
- [ ] 5 - **Excellent**: Comprehensive model card with use cases, limitations, ethical considerations, and update plan

**Must have**:
- Model details (algorithm, version, date)
- Intended use and out-of-scope uses
- Training data description
- Performance metrics
- Limitations and risks

**Your Score**: _____ / 15 points

---

## 6. Deployment Design (Module 5) - 10%

### API Design
- [ ] 1 - No API code
- [ ] 2 - Basic Flask/FastAPI without validation
- [ ] 3 - API with input validation
- [ ] 4 - API with validation + error handling + tests
- [ ] 5 - **Excellent**: Production-ready API with validation, error handling, logging, tests, and documentation

**Must have**:
- FastAPI endpoint
- Pydantic schema for input validation
- Error handling (try/except)
- At least 3 test cases

### Edge Cases
- [ ] 1 - No edge case handling
- [ ] 2 - Handled 1 edge case
- [ ] 3 - Handled 2-3 edge cases
- [ ] 4 - Handled 4+ edge cases with tests
- [ ] 5 - **Excellent**: Comprehensive edge case testing + graceful degradation + user feedback

**Examples**:
- Unknown Pokemon type
- All stats at maximum
- Impossible combinations
- Missing optional fields
- Very old or new generations

### Performance & Testing
- [ ] 1 - No performance considerations
- [ ] 2 - Mentioned performance but not measured
- [ ] 3 - Basic performance testing
- [ ] 4 - Load testing + SLO definition
- [ ] 5 - **Excellent**: Load testing + SLOs + optimization + scalability plan

**Your Score**: _____ / 10 points

---

## 7. Monitoring Strategy (Module 6) - 10%

### Metrics & Alerts
- [ ] 1 - No monitoring plan
- [ ] 2 - Listed some metrics
- [ ] 3 - Metrics + thresholds defined
- [ ] 4 - Metrics + thresholds + alerting plan
- [ ] 5 - **Excellent**: Comprehensive monitoring with dashboards, alerts, and automated responses

**Must have**:
- Model performance metrics
- Data quality metrics
- Business metrics
- Alert thresholds (warning and critical)

### Drift Detection
- [ ] 1 - No drift detection
- [ ] 2 - Mentioned drift but no implementation
- [ ] 3 - Basic drift detection code
- [ ] 4 - Statistical drift detection + visualization
- [ ] 5 - **Excellent**: Automated drift detection + alerting + retraining triggers

**Must have**:
- Code to detect distribution shifts
- Plan for what to do when drift detected
- Retraining schedule (trigger-based or time-based)

### Incident Response
- [ ] 1 - No incident plan
- [ ] 2 - Generic "we'll fix it" statement
- [ ] 3 - Basic runbook for one scenario
- [ ] 4 - Runbooks for 2+ scenarios
- [ ] 5 - **Excellent**: Detailed runbooks with diagnostic steps, mitigation, and postmortem template

**Your Score**: _____ / 10 points

---

## 8. Documentation & Collaboration (Module 7) - 5%

### README & Setup
- [ ] 1 - No documentation
- [ ] 2 - Minimal README
- [ ] 3 - README with setup instructions
- [ ] 4 - Comprehensive README with project structure
- [ ] 5 - **Excellent**: Professional README + architecture diagrams + examples + FAQs

**Must have**:
- Installation instructions
- How to train model
- How to run API
- Project structure
- Performance results

### Pull Request Description
- [ ] 1 - No PR description
- [ ] 2 - One-line summary
- [ ] 3 - Summary + changes
- [ ] 4 - Summary + changes + testing + risks
- [ ] 5 - **Excellent**: Professional PR with context, changes, testing, visuals, and questions for reviewers

**Must have**:
- Clear summary
- Performance results table
- Testing checklist
- Deployment plan
- Questions for reviewers

### Code Quality
- [ ] 1 - No structure, hard to read
- [ ] 2 - Basic structure
- [ ] 3 - Organized with functions
- [ ] 4 - Clean code with type hints and docstrings
- [ ] 5 - **Excellent**: Production-quality code with tests, type hints, docstrings, and style consistency

**Your Score**: _____ / 5 points

---

## Final Scoring

| Category | Your Score | Max Points |
|----------|------------|------------|
| 1. Business Context | _____ | 10 |
| 2. Data Engineering | _____ | 15 |
| 3. Feature Engineering | _____ | 15 |
| 4. Model Training | _____ | 15 |
| 5. Model Evaluation | _____ | 15 |
| 6. Deployment Design | _____ | 10 |
| 7. Monitoring Strategy | _____ | 10 |
| 8. Documentation | _____ | 5 |
| **TOTAL** | **_____** | **100** |

---

## Grading Scale

- **90-100**: 🌟 **Excellent** - Hire-ready ML engineer
- **80-89**: 🎯 **Strong** - Solid foundation, ready for junior role
- **70-79**: 📈 **Good** - Understands core concepts, needs more practice
- **60-69**: 📝 **Needs Work** - Review weak areas and redo those modules
- **< 60**: 🔄 **Redo** - Revisit modules and try again

---

## Specific Feedback Areas

### If you scored low on Business Context (Module 0):
- Review Module 0 again
- Practice translating model metrics to business impact
- Interview someone in product or business roles
- Read case studies of ML projects

### If you scored low on Data Engineering (Module 1):
- Review Module 1 again
- Practice writing Pandera schemas
- Learn about data quality frameworks
- Study production data pipelines

### If you scored low on Feature Engineering (Module 2):
- Review Module 2 again
- Practice identifying data leakage
- Read about feature engineering for your domain
- Study how to use scikit-learn pipelines properly

### If you scored low on Model Training (Module 3):
- Review Module 3 again
- Practice systematic model comparison
- Learn more about cross-validation
- Study hyperparameter tuning strategies

### If you scored low on Model Evaluation (Module 4):
- Review Module 4 again
- Practice error analysis
- Learn when to use different metrics
- Study model interpretability techniques

### If you scored low on Deployment Design (Module 5):
- Review Module 5 again
- Build more APIs with FastAPI
- Learn about production ML serving
- Study edge case testing

### If you scored low on Monitoring Strategy (Module 6):
- Review Module 6 again
- Learn about observability in ML systems
- Study incident response
- Practice writing runbooks

### If you scored low on Documentation (Module 7):
- Review Module 7 again
- Practice writing technical documentation
- Study good READMEs from popular ML projects
- Get feedback on your PR descriptions

---

## Next Steps Based on Your Score

### 90-100 🌟
**Congratulations!** You're ready to:
- Apply for ML engineer roles
- Contribute to open-source ML projects
- Build more complex projects (e.g., computer vision, NLP)
- Learn advanced topics (MLOps, model serving, feature stores)

### 80-89 🎯
**Strong work!** To level up:
- Strengthen weak areas identified above
- Build 1-2 more end-to-end projects
- Get code reviews from experienced engineers
- Study production ML systems

### 70-79 📈
**Good progress!** Keep going:
- Revisit modules where you scored < 4
- Redo exercises for weak areas
- Pair program with someone more experienced
- Build simpler projects to reinforce basics

### < 70 🔄
**Don't give up!** Here's the plan:
- Identify your weakest module
- Redo that module completely
- Do all exercises for that module
- Build a mini-project focusing on that skill
- Then move to next weak area

---

## Peer Review (Optional but Recommended)

Ask a peer or mentor to review your project using this rubric. External feedback is invaluable!

**Reviewer**: _______________
**Date**: _______________

**Reviewer Score**: _____ / 100

**Reviewer Comments**:

---

## Submission

If you're taking this as part of a course or interview:

**Name**: _______________
**Date**: _______________
**GitHub Repo**: _______________
**Self-Assessed Score**: _____ / 100

**What I'm most proud of**:

**What I struggled with most**:

**What I learned**:

**How I would improve this with more time**:

---

## Course Creator Notes

This rubric is designed to be:
- **Objective**: Clear criteria for each level
- **Comprehensive**: Covers all ML engineering skills
- **Actionable**: Specific feedback for improvement
- **Realistic**: Aligned with industry expectations

Students should aim for 80+ to be considered job-ready. 90+ indicates strong senior potential.

**Remember**: This is about learning, not perfection. Use this rubric to identify growth areas, not to feel discouraged. Every ML engineer started where you are now!
