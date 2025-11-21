# ML Engineering Learning Progress Tracker

**Your Name**: _________________
**Start Date**: _________________
**Target Completion**: _________________ (3-4 weeks recommended)

---

## 📊 Overall Progress

**Modules Completed**: __ / 8
**Exercises Completed**: __ / 7
**Capstone Score**: __ / 100 (Target: 80+)
**Current Level**: Level 0 (Beginner) → Target: Level 3 (Proficient)

---

## 🎯 Pre-Course Setup

- [ ] Python 3.13+ installed
- [ ] **uv** installed ([install here](https://docs.astral.sh/uv/))
- [ ] Dependencies synced (`uv sync`)
- [ ] Dataset generated (`uv run python data/generate_dataset.py`)
- [ ] Can run `uvx marimo edit ./` successfully (workspace mode)

**💡 Pro Tip**: Use `uvx marimo edit ./` to open the entire project and easily switch between modules!

---

## Phase 1: Business & Technical Foundations (Week 1: 8-10 hours)

### 📚 Module 0: ML in Business Context (1-2 hours)

**Why this module**: Most ML projects fail because engineers build the wrong thing. Learn to frame problems correctly.

#### Core Notebook
- [ ] Opened `00_ml_in_business.py` with `uvx marimo edit ./`
- [ ] Ran all cells successfully
- [ ] Understood when to use ML (and when NOT to)
- [ ] Learned ROI calculation
- [ ] Practiced stakeholder communication
- [ ] Completed inline exercises

#### Exercise Notebook
- [ ] Exercise 0.1: 5 "Should you use ML?" scenarios (all correct)
- [ ] Exercise 0.2: Project proposal with ROI calculation
- [ ] Exercise 0.3: Translate metrics to business impact
- [ ] Exercise 0.4: Stakeholder Q&A roleplay

#### Socratic Questions (Can you answer these?)
- [ ] "When should you NOT use ML and use a simple rule instead?"
- [ ] "How do you calculate ROI for an ML project?"
- [ ] "What's the difference between a model metric and a business metric?"
- [ ] "How do you set realistic expectations with stakeholders?"

#### Mastery Checklist
- [ ] ✅ Can explain when ML is appropriate for a problem
- [ ] ✅ Can calculate and justify ROI for ML projects
- [ ] ✅ Can translate technical metrics to business impact
- [ ] ✅ Can communicate with non-technical stakeholders

**Module 0 Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed

**"Aha Moments" I had**:
-
-
-

---

### 📚 Module 1: Data Engineering Foundations (2-3 hours)

**Why this module**: 80% of ML work is data engineering. Master this and you'll stand out.

#### Core Notebook
- [ ] Opened `01_data_engineering.py` with `uvx marimo edit ./`
- [ ] Ran all cells successfully
- [ ] Understood data loading concepts
- [ ] Understood data validation with Pandera
- [ ] Compared pandas vs polars performance
- [ ] Completed inline exercises

#### Exercise Notebook
- [ ] Exercise 1.1: Break the data (introduce 5 quality issues)
- [ ] Exercise 1.2: Refactor messy data loading code
- [ ] Exercise 1.3: Optimize slow pandas operation with polars

#### Socratic Questions (Can you answer these?)
- [ ] "If I run your data pipeline twice on the same input, will I get identical output? Why or why not?"
- [ ] "Your model performed great in training but fails in production. The data looks similar. What could be wrong?"
- [ ] "Someone changed a column name upstream. How would you detect this before it breaks your model?"
- [ ] "When would you choose polars over pandas? When would you stick with pandas?"

#### Mastery Checklist
- [ ] ✅ Can write a Pandera validation schema from scratch in <10 minutes
- [ ] ✅ Can explain why data versioning matters to a junior engineer
- [ ] ✅ Can debug a data quality issue by reading test failures
- [ ] ✅ Can choose the right tool (pandas/polars/SQL) for a given task

**Module 1 Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed

**"Aha Moments" I had**:
-
-
-

---

### 📚 Module 2: EDA & Feature Engineering (3-4 hours)

**Why this module**: Good features > fancy algorithms. This is where domain knowledge matters.

#### Core Notebook
- [ ] Opened `02_eda_and_features.py` with `uvx marimo edit ./`
- [ ] Ran all cells successfully
- [ ] Understood statistical analysis techniques
- [ ] Learned feature engineering patterns
- [ ] **Critical**: Learned to prevent data leakage
- [ ] Built scikit-learn Pipelines
- [ ] Completed inline exercises

#### Exercise Notebook
- [ ] Exercise 2.1: Engineer 5 new features with justifications
- [ ] Exercise 2.2: Spot the leakage - found all bugs
- [ ] Exercise 2.3: Created feature that improves model >5%

#### Socratic Questions (Can you answer these?)
- [ ] "You found that HP and Defense are highly correlated. Should you remove one? Why or why not?"
- [ ] "Your model achieves 99% accuracy. You're suspicious. What do you check first?"
- [ ] "Why do we fit preprocessing (like scalers) only on training data, not all data?"
- [ ] "You're creating a 'power_ratio' feature (Attack/Defense). What happens when Defense is 0?"
- [ ] "A domain expert suggests a feature. It has zero correlation with the target. Do you include it?"

#### Mastery Checklist
- [ ] ✅ Can generate 10 feature ideas in 5 minutes
- [ ] ✅ Can spot data leakage in someone else's code
- [ ] ✅ Can explain feature engineering decisions to stakeholders
- [ ] ✅ Can build a scikit-learn Pipeline from scratch

**Module 2 Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed

**"Aha Moments" I had**:
-
-
-

---

## Phase 2: Model Development (Week 2: 8-10 hours)

### 📚 Module 3: Model Training & Experimentation (3-4 hours)

**Why this module**: Don't just try random models. Be systematic.

#### Core Notebook
- [ ] Opened `03_model_training.py` with `uvx marimo edit ./`
- [ ] Ran all cells successfully
- [ ] Trained baseline models (always start simple!)
- [ ] Compared multiple model types
- [ ] Implemented cross-validation
- [ ] Tracked experiments systematically
- [ ] Completed inline exercises

#### Exercise Notebook
- [ ] Exercise 3.1: Implemented cross-validation from scratch
- [ ] Exercise 3.2: Model selection with stakeholder explanation
- [ ] Exercise 3.3: Hyperparameter tuning within latency budget

#### Socratic Questions (Can you answer these?)
- [ ] "Your validation accuracy is 90% but test accuracy is 70%. What happened?"
- [ ] "When would you choose Logistic Regression over XGBoost, even if XGBoost is more accurate?"
- [ ] "You're tuning hyperparameters. Should you use test data to select the best model? Why not?"
- [ ] "Your model is 90% accurate but your PM is unhappy. What might you be missing?"
- [ ] "Training takes 8 hours. How do you experiment efficiently?"

#### Mastery Checklist
- [ ] ✅ Can set up an experiment from scratch in 15 minutes
- [ ] ✅ Can explain bias-variance tradeoff with concrete example
- [ ] ✅ Can read experiment logs and know what to try next
- [ ] ✅ Can estimate required training time before starting

**Module 3 Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed

**"Aha Moments" I had**:
-
-
-

---

### 📚 Module 4: Model Evaluation & Validation (2-3 hours)

**Why this module**: Accuracy is not enough. Learn to evaluate deeply.

#### Core Notebook
- [ ] Opened `04_model_evaluation.py` with `uvx marimo edit ./`
- [ ] Ran all cells successfully
- [ ] Understood precision vs recall tradeoffs
- [ ] Analyzed confusion matrices
- [ ] Performed thorough error analysis
- [ ] Created a model card
- [ ] Completed inline exercises

#### Exercise Notebook
- [ ] Exercise 4.1: Metric selection for 5 business scenarios
- [ ] Exercise 4.2: Error analysis - categorized misclassifications
- [ ] Exercise 4.3: Threshold tuning with business costs
- [ ] Exercise 4.4: Created professional model card
- [ ] Exercise 4.5: Presented results to mock stakeholder

#### Socratic Questions (Can you answer these?)
- [ ] "Your model is 95% accurate on spam detection. Is that good?"
- [ ] "Would you rather have high precision or high recall for: (a) cancer detection, (b) spam filtering, (c) credit card fraud? Why?"
- [ ] "Your model has perfect training accuracy but mediocre test accuracy. List 5 possible causes."
- [ ] "How do you explain 'the model is 80% confident' to a non-technical user?"
- [ ] "Your model is unfair to a demographic group. List 3 ways this could have happened."

#### Mastery Checklist
- [ ] ✅ Can choose the right metric for any business problem
- [ ] ✅ Can explain model limitations without being asked
- [ ] ✅ Can conduct thorough error analysis in 30 minutes
- [ ] ✅ Can write a production-ready model card

**Module 4 Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed

**"Aha Moments" I had**:
-
-
-

---

### 📚 Module 5: Deployment & Inference (2-3 hours)

**Why this module**: A model in a notebook is worth $0. Learn to deploy.

#### Core Notebook
- [ ] Opened `05_inference_service.py` with `uvx marimo edit ./`
- [ ] Ran all cells successfully
- [ ] Built inference API with error handling
- [ ] Implemented input validation with Pydantic
- [ ] Optimized inference latency
- [ ] Understood model serving patterns
- [ ] Completed inline exercises

#### Socratic Questions (Can you answer these?)
- [ ] "Training takes 1 hour. Inference must complete in <100ms. How do you approach this?"
- [ ] "A user sends malformed input that crashes your API. Who's responsible - you or the user?"
- [ ] "Your model was trained on 2024 data. It's now 2025. What changes might break it?"
- [ ] "When do you retrain: (a) on a schedule, (b) when accuracy drops, (c) when data drifts? Why?"

#### Mastery Checklist
- [ ] ✅ Can deploy a model behind an API in <30 minutes
- [ ] ✅ Can handle edge cases gracefully
- [ ] ✅ Can explain inference latency tradeoffs to eng team
- [ ] ✅ Can design an A/B test for model deployment

**Module 5 Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed

**"Aha Moments" I had**:
-
-
-

---

## Phase 3: Production & Collaboration (Week 3: 6-8 hours)

### 📚 Module 6: Production ML & Monitoring (2-3 hours)

**Why this module**: Your model will break. Be ready.

#### Core Notebook
- [ ] Opened `06_production_monitoring.py` with `uvx marimo edit ./`
- [ ] Ran all cells successfully
- [ ] Learned production debugging runbook
- [ ] Implemented data drift detection
- [ ] Understood monitoring strategies
- [ ] Designed alerting thresholds
- [ ] Completed inline exercises

#### Exercise Notebook
- [ ] Exercise 6.1: Debugged incident - accuracy drop (87% → 65%)
- [ ] Exercise 6.2: Debugged incident - latency spike (50ms → 2000ms)
- [ ] Exercise 6.3: Debugged incident - all predictions one class
- [ ] Exercise 6.4: Designed monitoring dashboard
- [ ] Exercise 6.5: Wrote incident report

#### Socratic Questions (Can you answer these?)
- [ ] "You deployed a model. A week later, accuracy drops from 85% to 60%. What do you investigate first?"
- [ ] "What's the difference between data drift and concept drift?"
- [ ] "Should you alert on every metric violation? Why or why not?"
- [ ] "How often should you retrain your model?"

#### Mastery Checklist
- [ ] ✅ Can debug production issues using logs and metrics
- [ ] ✅ Can detect data drift using statistical tests
- [ ] ✅ Can write a runbook for model operations
- [ ] ✅ Can design an effective monitoring strategy

**Module 6 Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed

**"Aha Moments" I had**:
-
-
-

---

### 📚 Module 7: Team Collaboration & Code Reviews (2-3 hours)

**Why this module**: You'll work with other engineers. Learn to collaborate effectively.

#### Core Notebook
- [ ] Opened `07_collaboration.py` with `uvx marimo edit ./`
- [ ] Ran all cells successfully
- [ ] Learned Git workflows for ML
- [ ] Understood ML code review best practices
- [ ] Learned to write clear PR descriptions
- [ ] Understood common ML code smells
- [ ] Completed inline exercises

#### Exercise Notebook
- [ ] Exercise 7.1: Reviewed PR #1 - Feature engineering (found data leakage!)
- [ ] Exercise 7.2: Reviewed PR #2 - Model training (found test set peeking!)
- [ ] Exercise 7.3: Reviewed PR #3 - Deployment (found missing validation!)
- [ ] Exercise 7.4: Wrote PR description for capstone project
- [ ] Exercise 7.5: Created code review checklist

#### Socratic Questions (Can you answer these?)
- [ ] "What files should you commit to Git? What should you .gitignore?"
- [ ] "You found a critical bug in a teammate's PR. How do you communicate it?"
- [ ] "Should you commit trained model files to Git? Why or why not?"
- [ ] "How do you onboard to an existing ML codebase?"

#### Mastery Checklist
- [ ] ✅ Can use Git workflows for ML projects
- [ ] ✅ Can review ML code effectively
- [ ] ✅ Can write clear PR descriptions
- [ ] ✅ Can document models professionally

**Module 7 Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed

**"Aha Moments" I had**:
-
-
-

---

## Phase 4: Capstone Project (Week 4: 4-6 hours)

### 📚 Module 8: Capstone - Pokemon Card Price Prediction

**Why this module**: Apply everything you've learned in one complete project.

#### Core Notebook
- [ ] Opened `08_capstone.py` with `uvx marimo edit ./`
- [ ] Read through all 9 phases
- [ ] Understood deliverables for each phase
- [ ] Reviewed CAPSTONE_RUBRIC.md (100 points)

#### Your Deliverables

**Phase 1: Business Context** (10 points)
- [ ] Problem statement with stakeholders and use cases
- [ ] Success metrics (model + business + user)
- [ ] ROI calculation with timeline
- [ ] Stakeholder pitch (1 paragraph)

**Phase 2: Data Engineering** (15 points)
- [ ] Pandera schema with all validations
- [ ] Data quality checks with visualizations
- [ ] Reproducible `load_and_validate_data()` function
- [ ] Train/val/test split (never touched test until final!)

**Phase 3: Feature Engineering** (15 points)
- [ ] 5+ engineered features with documentation
- [ ] Leakage checks for EVERY feature
- [ ] Feature documentation table
- [ ] scikit-learn preprocessing pipeline

**Phase 4: Model Training** (15 points)
- [ ] Baseline model results
- [ ] 4+ algorithms compared with cross-validation
- [ ] Hyperparameter tuning with best params
- [ ] MLflow experiment tracking

**Phase 5: Model Evaluation** (15 points)
- [ ] Appropriate metrics chosen with justification
- [ ] Error analysis with visualizations
- [ ] Residual plots and pattern identification
- [ ] Professional model card

**Phase 6: Deployment Design** (10 points)
- [ ] FastAPI endpoint with Pydantic validation
- [ ] 5+ edge cases handled
- [ ] Performance requirements defined (SLOs)
- [ ] A/B test plan

**Phase 7: Monitoring Strategy** (10 points)
- [ ] Key metrics to monitor
- [ ] Data drift detection code
- [ ] Alert thresholds (warning + critical)
- [ ] 2+ incident response runbooks

**Phase 8: Documentation** (5 points)
- [ ] Professional README with setup instructions
- [ ] PR description with results and testing
- [ ] Type hints and docstrings
- [ ] Code review checklist

**Phase 9: Final Evaluation** (5 points)
- [ ] Test set results (only used once!)
- [ ] Comparison of validation vs test metrics
- [ ] Self-assessment with CAPSTONE_RUBRIC.md
- [ ] Reflection questions answered

#### Self-Assessment
- [ ] Scored myself using CAPSTONE_RUBRIC.md
- [ ] **My Score**: _____ / 100
- [ ] **Target**: 80+ (hire-ready)

#### Mastery Checklist
- [ ] ✅ Can build end-to-end ML systems independently
- [ ] ✅ Can frame problems with business context
- [ ] ✅ Can engineer features without leakage
- [ ] ✅ Can train and evaluate models systematically
- [ ] ✅ Can deploy with monitoring
- [ ] ✅ Can document professionally
- [ ] ✅ Can collaborate effectively

**Module 8 Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed

**What I'm most proud of**:


**What I struggled with most**:


**What I learned**:


**How I would improve this with more time**:


---

## 🎓 Course Completion

### Overall Statistics
**Completion Date**: _________________
**Total Time Invested**: _____ hours (Target: 20-24 hours)
**Final Capstone Score**: _____ / 100 (Target: 80+)
**Final Skill Level**: _________________

### Skills Assessment

#### Technical Skills (check if proficient)
- [ ] Build reproducible data pipelines
- [ ] Engineer features without data leakage
- [ ] Train models systematically
- [ ] Evaluate with appropriate metrics
- [ ] Deploy models to production
- [ ] Debug production ML issues
- [ ] Detect and respond to data drift

#### Business Skills (check if proficient)
- [ ] Frame ML problems correctly
- [ ] Calculate and justify ROI
- [ ] Set realistic expectations
- [ ] Communicate with non-technical stakeholders
- [ ] Know when NOT to use ML

#### Collaboration Skills (check if proficient)
- [ ] Use Git workflows for ML
- [ ] Review ML code effectively
- [ ] Write clear documentation
- [ ] Onboard to existing codebases
- [ ] Work in ML teams

### What's Next?

**Immediate (Next 2 weeks)**:
- [ ] Add capstone to GitHub portfolio
- [ ] Update LinkedIn/resume with new skills
- [ ] Write blog post about learning journey
- [ ] Share capstone project with community

**Short-term (1-3 months)**:
- [ ] Build 2-3 more ML projects in different domains
- [ ] Contribute to open-source ML project
- [ ] Apply for junior ML engineer roles
- [ ] Get feedback from senior engineers

**Medium-term (3-6 months)**:
- [ ] Specialize in domain: ⬜ NLP | ⬜ Computer Vision | ⬜ Recommender Systems | ⬜ Time Series
- [ ] Learn advanced topics: deep learning, MLOps, model compression
- [ ] Mentor someone learning ML
- [ ] Give talk at meetup

---

## 📝 Reflection

### What surprised me most about ML engineering:


### The most valuable skill I learned:


### If I could redo the course, I would:


### My advice for future learners:


---

## 🏆 I'm Ready!

I have completed all 8 modules + capstone and am ready to contribute to production ML systems as a junior ML engineer.

**Skills I can now do**:
- ✅ Frame ML problems with business context
- ✅ Build production-quality data pipelines
- ✅ Engineer features systematically
- ✅ Train and evaluate models rigorously
- ✅ Deploy models to production
- ✅ Debug production ML issues
- ✅ Collaborate effectively in ML teams

**My portfolio**: _________________
**LinkedIn**: _________________
**GitHub**: _________________

**Signature**: _________________ **Date**: _________________

---

*This course teaches the 80-20 most important skills for ML engineers: 60% technical, 20% business, 20% collaboration. You're now ready for ML engineering roles! 🚀*
