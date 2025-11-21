# ML Engineering Learning Progress Tracker

**Your Name**: _________________
**Start Date**: _________________
**Target Completion**: _________________ (2-3 weeks recommended)

---

## 📊 Overall Progress

**Modules Completed**: __ / 5
**Exercises Completed**: __ / 15
**Checkpoints Passed**: __ / 3
**Current Level**: Level 0 (Beginner) → Target: Level 3 (Proficient)

---

## 🎯 Pre-Course Setup

- [ ] Python 3.9+ installed
- [ ] **uv** installed ([install here](https://docs.astral.sh/uv/))
- [ ] Dependencies synced (`uv sync`)
- [ ] Dataset downloaded and verified (`uv run python data/generate_dataset.py`)
- [ ] Can run `uvx marimo edit 01_data_engineering.py` successfully

---

## 📚 Module 1: Data Engineering Foundations (2-3 hours)

### Core Notebook
- [ ] Opened `01_data_engineering.py` with `uvx marimo edit 01_data_engineering.py`
- [ ] Ran all cells successfully
- [ ] Understood data loading concepts
- [ ] Understood data validation with Pandera
- [ ] Compared pandas vs polars performance
- [ ] Completed inline exercises

### Exercise Notebook
- [ ] Exercise 1.1: Break the data (introduce 5 quality issues)
- [ ] Exercise 1.2: Refactor messy data loading code
- [ ] Exercise 1.3: Optimize slow pandas operation with polars

### Socratic Questions (Can you answer these?)
- [ ] "If I run your data pipeline twice on the same input, will I get identical output? Why or why not?"
- [ ] "Your model performed great in training but fails in production. The data looks similar. What could be wrong?"
- [ ] "Someone changed a column name upstream. How would you detect this before it breaks your model?"
- [ ] "When would you choose polars over pandas? When would you stick with pandas?"

### Checkpoint Assessment
- [ ] **Challenge**: "New Pokemon Data Drop" completed
- [ ] Wrote validation pipeline for new schema
- [ ] Caught all data quality issues
- [ ] Documented findings

### Mastery Checklist
- [ ] ✅ Can write a data validation test from scratch in <10 minutes
- [ ] ✅ Can explain why data versioning matters to a junior engineer
- [ ] ✅ Can debug a data quality issue by reading test failures
- [ ] ✅ Can choose the right tool (pandas/polars/SQL) for a given task

**Module 1 Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed
**"Aha Moments" I had**:
-
-
-

---

## 📚 Module 2: EDA & Feature Engineering (3-4 hours)

### Core Notebook
- [ ] Opened `02_eda_and_features.py` with `uvx marimo edit 02_eda_and_features.py`
- [ ] Ran all cells successfully
- [ ] Understood statistical analysis techniques
- [ ] Learned feature engineering patterns
- [ ] Built scikit-learn Pipelines
- [ ] Completed inline exercises

### Exercise Notebook
- [ ] Exercise 2.1: Engineer 5 new features with justifications
- [ ] Exercise 2.2: Spot the leakage - found all bugs
- [ ] Exercise 2.3: Created feature that improves model >5%

### Socratic Questions (Can you answer these?)
- [ ] "You found that HP and Defense are highly correlated. Should you remove one? Why or why not?"
- [ ] "Your model achieves 99% accuracy. You're suspicious. What do you check first?"
- [ ] "Why do we fit preprocessing (like scalers) only on training data, not all data?"
- [ ] "You're creating a 'power_ratio' feature (Attack/Defense). What happens when Defense is 0?"
- [ ] "A domain expert suggests a feature. It has zero correlation with the target. Do you include it?"

### Checkpoint Assessment
- [ ] **Challenge**: "Feature Engineering Competition" completed
- [ ] Improved baseline model from 65% to >75% accuracy
- [ ] Used ONLY feature engineering (no model changes)
- [ ] Documented what worked and why

### Mastery Checklist
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

## 📚 Module 3: Model Development & Experimentation (3-4 hours)

### Core Notebook
- [ ] Opened `03_model_training.py` with `uvx marimo edit 03_model_training.py`
- [ ] Ran all cells successfully
- [ ] Trained baseline models
- [ ] Compared multiple model types
- [ ] Implemented cross-validation
- [ ] Tracked experiments systematically
- [ ] Completed inline exercises

### Exercise Notebook
- [ ] Exercise 3.1: Implemented cross-validation from scratch
- [ ] Exercise 3.2: Model selection with stakeholder explanation
- [ ] Exercise 3.3: Hyperparameter tuning within latency budget

### Socratic Questions (Can you answer these?)
- [ ] "Your validation accuracy is 90% but test accuracy is 70%. What happened?"
- [ ] "When would you choose Logistic Regression over XGBoost, even if XGBoost is more accurate?"
- [ ] "You're tuning hyperparameters. Should you use test data to select the best model? Why not?"
- [ ] "Your model is 90% accurate but your PM is unhappy. What might you be missing?"
- [ ] "Training takes 8 hours. How do you experiment efficiently?"

### Checkpoint Assessment
- [ ] **Challenge**: "Debug the Failing Model" completed
- [ ] Diagnosed what was wrong with provided model
- [ ] Fixed the issue
- [ ] Explained reasoning process

### Mastery Checklist
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

## 📚 Module 4: Model Evaluation & Validation (2-3 hours)

### Core Notebook
- [ ] Opened `04_model_evaluation.py` with `uvx marimo edit 04_model_evaluation.py`
- [ ] Ran all cells successfully
- [ ] Understood precision vs recall tradeoffs
- [ ] Analyzed confusion matrices
- [ ] Performed thorough error analysis
- [ ] Created a model card
- [ ] Completed inline exercises

### Exercise Notebook
- [ ] Exercise 4.1: Metric selection for 3 business scenarios
- [ ] Exercise 4.2: Error analysis - categorized 50 misclassifications
- [ ] Exercise 4.3: Threshold tuning with business costs

### Socratic Questions (Can you answer these?)
- [ ] "Your model is 95% accurate on spam detection. Is that good?"
- [ ] "Would you rather have high precision or high recall for: (a) cancer detection, (b) spam filtering, (c) credit card fraud? Why?"
- [ ] "Your model has perfect training accuracy but mediocre test accuracy. List 5 possible causes."
- [ ] "How do you explain 'the model is 80% confident' to a non-technical user?"
- [ ] "Your model is unfair to a demographic group. List 3 ways this could have happened."

### Checkpoint Assessment
- [ ] **Challenge**: "Model Review Board" completed
- [ ] Prepared executive summary
- [ ] Created comprehensive model card
- [ ] Conducted risk analysis
- [ ] Made deployment recommendation

### Mastery Checklist
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

## 📚 Module 5: Deployment, Inference & Monitoring (2-3 hours)

### Core Notebook
- [ ] Opened `05_inference_service.py` with `uvx marimo edit 05_inference_service.py`
- [ ] Ran all cells successfully
- [ ] Built inference API with error handling
- [ ] Implemented input validation
- [ ] Optimized inference latency
- [ ] Created monitoring dashboard
- [ ] Completed inline exercises

### Exercise Notebook
- [ ] Exercise 5.1: API handling 5 different error cases
- [ ] Exercise 5.2: Optimized latency from 500ms to <100ms
- [ ] Exercise 5.3: Implemented data drift detection

### Socratic Questions (Can you answer these?)
- [ ] "You deployed a model. A week later, accuracy drops from 85% to 60%. What do you investigate first?"
- [ ] "Training takes 1 hour. Inference must complete in <100ms. How do you approach this?"
- [ ] "A user sends malformed input that crashes your API. Who's responsible - you or the user?"
- [ ] "Your model was trained on 2024 data. It's now 2025. What changes might break it?"
- [ ] "When do you retrain: (a) on a schedule, (b) when accuracy drops, (c) when data drifts? Why?"

### Checkpoint Assessment
- [ ] **Challenge**: "Production Incident Simulation" completed
- [ ] Alert 1: Diagnosed and fixed latency spike
- [ ] Alert 2: Diagnosed and fixed low confidence scores
- [ ] Alert 3: Diagnosed and fixed classification bias
- [ ] Documented prevention measures for all

### Mastery Checklist
- [ ] ✅ Can deploy a model behind an API in <30 minutes
- [ ] ✅ Can debug production issues using logs and metrics
- [ ] ✅ Can explain inference latency tradeoffs to eng team
- [ ] ✅ Can write a runbook for model operations

**Module 5 Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed
**"Aha Moments" I had**:
-
-
-

---

## 🎓 Course Checkpoints

### Checkpoint 1: End of Week 1 (Modules 1-2)
- [ ] Completed self-assessment quiz (20 questions)
- [ ] Score: _____ / 20 (Target: 16+)
- [ ] Completed practical exam: Customer churn dataset
- [ ] Submitted working validation pipeline
- [ ] Delivered 3+ actionable insights from EDA
- [ ] **Status**: ⬜ Pass | ⬜ Need Review

### Checkpoint 2: End of Week 2 (Modules 3-4)
- [ ] Completed self-assessment quiz (25 questions)
- [ ] Score: _____ / 25 (Target: 20+)
- [ ] Completed practical exam: Credit risk model
- [ ] Trained 3+ models with proper validation
- [ ] Wrote professional model card
- [ ] **Status**: ⬜ Pass | ⬜ Need Review

### Checkpoint 3: Final Assessment (After Module 5)
- [ ] Started capstone project: Pokemon card price predictor
- [ ] Completed data validation notebook
- [ ] Completed EDA and feature engineering
- [ ] Trained and evaluated 3+ models
- [ ] Selected and justified best model
- [ ] Built inference API with UI
- [ ] Wrote deployment documentation
- [ ] Prepared 10-slide presentation
- [ ] **Final Score**: _____ / 100 (Target: 80+)

---

## 📈 Skill Level Assessment

### Current Level (check one):
- [ ] **Level 0: Beginner** - Following tutorials, confused when things break
- [ ] **Level 1: Advanced Beginner** - Can load/clean data, understand common issues
- [ ] **Level 2: Competent** - Can train/evaluate models independently
- [ ] **Level 3: Proficient** - Build end-to-end ML systems, production-ready
- [ ] **Level 4: Expert** - Design ML systems, mentor others (6-12 months post-course)

### Skills Acquired

#### Technical Skills
- [ ] Load, validate, and version data
- [ ] Conduct EDA and extract insights
- [ ] Engineer features systematically
- [ ] Train multiple models and compare fairly
- [ ] Evaluate models with appropriate metrics
- [ ] Deploy models behind APIs
- [ ] Monitor model health in production

#### Professional Skills
- [ ] Write production-quality ML code
- [ ] Document decisions and tradeoffs
- [ ] Communicate with stakeholders
- [ ] Review others' ML code
- [ ] Debug production ML issues
- [ ] Estimate effort for ML projects

#### Mindset
- [ ] Think probabilistically, not deterministically
- [ ] Always validate assumptions with data
- [ ] Consider production from day one
- [ ] Document for future you
- [ ] Embrace uncertainty and iteration

---

## 📝 Weekly Reflection

### Week 1 Reflection (After Modules 1-2)
**What surprised me**:


**What was harder than expected**:


**What clicked for me**:


**What I need to review**:


**How I'll do things differently next time**:


---

### Week 2 Reflection (After Modules 3-4)
**What surprised me**:


**What was harder than expected**:


**What clicked for me**:


**What I need to review**:


**How I'll do things differently next time**:


---

### Week 3 Reflection (After Module 5 & Capstone)
**What surprised me**:


**What was harder than expected**:


**What clicked for me**:


**What I need to review**:


**How I'll do things differently next time**:


---

## 🎯 Post-Course Action Plan

### Immediate Next Steps (Week 4)
- [ ] Contribute to open source ML project (found project: _____________)
- [ ] Build personal ML project in different domain
- [ ] Publish project on GitHub
- [ ] Write blog post about learning journey
- [ ] Join ML communities (list 2-3):
  -
  -
  -

### 3-Month Goals
- [ ] Complete deep learning course (Fast.ai / PyTorch)
- [ ] Experiment with MLOps tools (Airflow, MLflow, etc.)
- [ ] Choose specialization: ⬜ NLP | ⬜ Computer Vision | ⬜ Recommender Systems
- [ ] Study advanced topic: _________________

### 6-Month Goals
- [ ] Contribute to 3+ ML projects (work or open source)
- [ ] Give a talk/presentation on ML topic
- [ ] Mentor someone learning ML
- [ ] Interview for ML engineer roles

---

## 🏆 Course Completion Certificate

I have completed the Professional ML Engineering Onboarding Project and am ready to contribute to production ML systems.

**Completion Date**: _________________
**Final Level Achieved**: _________________
**Capstone Score**: _____ / 100
**Total Time Invested**: _____ hours

**Signature**: _________________

---

## 📚 Additional Resources Used

**Books**:
-
-

**Online Courses**:
-
-

**Articles/Blog Posts**:
-
-

**Other**:
-
-

---

## 💭 Learning Journal

Use this space to track your daily progress, insights, and questions:

**Date**: ___________
**What I worked on**:


**What I learned**:


**Questions I have**:


**Tomorrow's goals**:


---

*Keep this tracker updated throughout your journey. Your future self will thank you!*
