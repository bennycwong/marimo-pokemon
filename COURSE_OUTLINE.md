# ML Engineering Course - Complete Learning Path

## Course Philosophy: The 80-20 That Companies Actually Need

Most ML courses teach algorithms and math. This course teaches what you actually need to succeed as an ML engineer at a company:

- **60%** Technical Skills (data, features, models, deployment)
- **20%** Business Acumen (ROI, stakeholder communication, problem framing)
- **20%** Collaboration (Git workflows, code reviews, documentation)

---

## Prerequisites

**Required**:
- Python programming (functions, classes, data structures)
- Basic pandas (read_csv, filtering, groupby)
- Command line basics (cd, ls, running scripts)

**Nice to have**:
- Statistics (mean, variance, distributions)
- Git basics (commit, push, pull)
- SQL (for understanding data pipelines)

**Not required**:
- Machine learning experience
- Math beyond high school algebra
- Deep learning or neural networks

---

## Course Structure (20-24 hours)

### Phase 1: Business & Technical Foundations (8-10 hours)

#### Module 0: ML in Business Context (1-2 hours)
**Why this module**: Most ML projects fail because engineers build the wrong thing. Learn to frame problems correctly.

**Topics**:
- When to use ML (and when a simple heuristic is better)
- ROI calculation: justify your work to leadership
- Stakeholder communication: translating metrics to business impact
- Setting realistic expectations
- Common failure modes and how to avoid them

**Exercises** (exercises_00.py):
- 5 "Should you use ML?" scenarios
- Write a project proposal with ROI
- Translate model metrics to business metrics
- Stakeholder Q&A roleplay

**Learning Outcome**: Frame ML problems like a senior engineer

---

#### Module 1: Data Engineering Foundations (2-3 hours)
**Why this module**: 80% of ML work is data engineering. Master this and you'll stand out.

**Topics**:
- Data loading and exploration (pandas + polars)
- Data quality validation with Pandera schemas
- Building reproducible pipelines
- Train/val/test splits (done right)
- Data versioning and lineage

**Exercises** (exercises_01.py):
- Write Pandera schemas for 3 datasets
- Debug 5 data quality issues
- Build a data validation pipeline
- Compare pandas vs polars performance

**Learning Outcome**: Build production-quality data pipelines

---

#### Module 2: EDA & Feature Engineering (3-4 hours)
**Why this module**: Good features > fancy algorithms. This is where domain knowledge matters.

**Topics**:
- Systematic exploratory data analysis
- Feature engineering from domain knowledge
- **Critical**: Preventing data leakage (most common mistake!)
- scikit-learn pipelines for preprocessing
- Feature selection techniques

**Exercises** (exercises_02.py):
- EDA on Pokemon cards: find patterns
- Engineer 10 features (3 will have leakage - find them!)
- Build a preprocessing pipeline
- Feature engineering competition: beat the baseline

**Learning Outcome**: Engineer features without leaking information

---

### Phase 2: Model Development (8-10 hours)

#### Module 3: Model Training & Experimentation (3-4 hours)
**Why this module**: Don't just try random models. Be systematic.

**Topics**:
- Baseline models (always start simple!)
- Model zoo: sklearn, XGBoost, LightGBM
- Cross-validation strategies
- Hyperparameter tuning (grid search, random search, Bayesian)
- Experiment tracking with MLflow
- Avoiding overfitting

**Exercises** (exercises_03.py):
- Build 5 baseline models
- Hyperparameter tuning competition
- Debug 3 overfitting scenarios
- Set up MLflow experiment tracking

**Learning Outcome**: Train models systematically with reproducible experiments

---

#### Module 4: Model Evaluation & Validation (2-3 hours)
**Why this module**: Accuracy is not enough. Learn to evaluate deeply.

**Topics**:
- Metrics beyond accuracy (precision, recall, F1, ROC-AUC)
- Confusion matrices and error analysis
- Calibration and confidence intervals
- Model cards and documentation
- Presenting results to stakeholders

**Exercises** (exercises_04.py):
- Choose metrics for 5 different problems (spam filter, cancer screening, etc.)
- Error analysis: find failure modes
- Confidence calibration exercise
- Create a model card
- Present results to a mock stakeholder

**Learning Outcome**: Evaluate models rigorously and communicate results

---

#### Module 5: Deployment & Inference (2-3 hours)
**Why this module**: A model in a notebook is worth $0. Learn to deploy.

**Topics**:
- Model serialization (pickle, joblib, ONNX)
- FastAPI for model serving
- Input validation with Pydantic
- Error handling and edge cases
- Performance optimization (batching, caching)
- A/B testing framework

**Exercises** (exercises_05.py):
- Build a FastAPI endpoint
- Handle 10 edge cases
- Load test your API
- Design an A/B test

**Learning Outcome**: Deploy models to production

---

### Phase 3: Production & Collaboration (6-8 hours)

#### Module 6: Production ML & Monitoring (2-3 hours)
**Why this module**: Your model will break. Be ready.

**Topics**:
- Production debugging runbook (complete workflows)
- Data drift detection (KS test, PSI, ANOVA)
- Model performance monitoring
- Alerting strategies (when to page vs. when to log)
- Incident response procedures
- Retraining triggers

**Exercises** (exercises_06.py):
- Debug 3 production incidents:
  1. Accuracy drops from 87% to 65%
  2. Latency spikes from 50ms to 2000ms
  3. Model predicts all one class
- Design a monitoring dashboard
- Write an incident report
- Create retraining pipeline

**Learning Outcome**: Debug production ML systems like a pro

---

#### Module 7: Team Collaboration & Code Reviews (2-3 hours)
**Why this module**: You'll work with other engineers. Learn to collaborate effectively.

**Topics**:
- Git workflows for ML (what to commit, what to ignore)
- ML code review best practices
- Writing clear PR descriptions
- Documentation patterns (READMEs, model cards, API docs)
- Onboarding to existing ML codebases
- Common ML code smells

**Exercises** (exercises_07.py):
- Review 3 sample PRs:
  1. Feature engineering (has data leakage!)
  2. Model training (overfits to validation set)
  3. Deployment code (missing input validation)
- Write a PR description for your capstone
- Create a code review checklist
- Document an existing model

**Learning Outcome**: Collaborate effectively with ML teams

---

### Phase 4: Capstone Project (4-6 hours)

#### Module 8: End-to-End Pokemon Card Price Prediction (4-6 hours)
**Why this module**: Apply everything you've learned in one complete project.

**The Challenge**:
You're the first ML engineer at PokéMarket, a Pokemon card trading platform. Build a price prediction system to help sellers price their cards.

**Your Deliverables** (see CAPSTONE_RUBRIC.md for grading):

1. **Business Analysis** (Module 0 skills)
   - Problem statement
   - Success metrics (model + business + user)
   - ROI calculation
   - Stakeholder pitch

2. **Data Pipeline** (Module 1 skills)
   - Pandera schema
   - Quality checks
   - Reproducible pipeline
   - Train/val/test split

3. **Feature Engineering** (Module 2 skills)
   - 5+ engineered features
   - Leakage checks documented
   - Feature documentation table
   - scikit-learn pipeline

4. **Model Training** (Module 3 skills)
   - Baseline model
   - 4+ algorithms compared
   - Hyperparameter tuning
   - MLflow experiment tracking

5. **Model Evaluation** (Module 4 skills)
   - Appropriate metrics chosen
   - Error analysis with visualizations
   - Model card
   - Results presentation

6. **Deployment Design** (Module 5 skills)
   - FastAPI endpoint
   - Pydantic input validation
   - Edge case handling
   - A/B test plan

7. **Monitoring Strategy** (Module 6 skills)
   - Drift detection code
   - Alert thresholds
   - 2+ incident runbooks
   - Retraining schedule

8. **Documentation** (Module 7 skills)
   - Professional README
   - PR description
   - Type hints + docstrings
   - Code review checklist

9. **Final Evaluation & Reflection**
   - Test set results
   - Self-assessment (CAPSTONE_RUBRIC.md)
   - Reflection questions

**Grading** (100 points total):
- Business Context: 10 points
- Data Engineering: 15 points
- Feature Engineering: 15 points
- Model Training: 15 points
- Model Evaluation: 15 points
- Deployment Design: 10 points
- Monitoring Strategy: 10 points
- Documentation: 5 points

**Scoring**:
- 90-100: 🌟 Hire-ready ML engineer
- 80-89: 🎯 Strong foundation, ready for junior role
- 70-79: 📈 Good understanding, needs more practice
- 60-69: 📝 Review weak areas
- < 60: 🔄 Redo modules and try again

**Time**: 4-6 hours (don't rush - quality matters!)

**Learning Outcome**: Build a complete, production-ready ML system

---

## Learning Path Options

### Option 1: Complete Path (Recommended) - 4 weeks
Follow all modules in order. Best for career switchers.

**Week 1**: Modules 0-2 + exercises (8-10 hours)
**Week 2**: Modules 3-5 + exercises (8-10 hours)
**Week 3**: Modules 6-7 + exercises (6-8 hours)
**Week 4**: Module 8 capstone (4-6 hours)

---

### Option 2: Technical Fast Track - 2 weeks
Skip business/collaboration, focus on technical core. Good if you already understand ML context.

**Week 1**: Modules 1-3 + exercises (8-10 hours)
**Week 2**: Modules 4-6 + exercises (8-10 hours)

---

### Option 3: Business-First Path - 3 weeks
Emphasize business and collaboration. Good for transitioning into ML leadership.

**Week 1**: Modules 0, 1, 2 + exercises (9-11 hours)
**Week 2**: Modules 3, 4, 5 (review only)
**Week 3**: Modules 6, 7, 8 + deep dive on documentation (8-10 hours)

---

## Key Concepts Covered

### Data Engineering
- Pandera schemas, data validation, quality checks
- pandas vs polars performance
- Reproducible pipelines
- Train/val/test splits
- Data versioning

### Feature Engineering
- Domain knowledge → features
- **Data leakage prevention** (critical!)
- scikit-learn pipelines
- Feature selection
- Preprocessing strategies

### Model Development
- Baseline models
- Cross-validation
- Hyperparameter tuning
- Experiment tracking (MLflow)
- Overfitting prevention

### Evaluation
- Metrics: accuracy, precision, recall, F1, ROC-AUC, RMSE, MAE, R²
- Confusion matrices
- Error analysis
- Model cards
- Stakeholder communication

### Deployment
- FastAPI model serving
- Pydantic input validation
- Error handling
- Edge cases
- A/B testing

### Production Operations
- Debugging runbooks
- Data drift detection
- Performance monitoring
- Incident response
- Retraining pipelines

### Collaboration
- Git workflows for ML
- Code reviews
- PR descriptions
- Documentation
- Onboarding

---

## Study Tips

### Do's ✅
- **Start with Module 0** - business context is critical
- Run every code cell and experiment with parameters
- Complete ALL exercises (they're where learning happens)
- Use the capstone rubric to self-assess honestly
- Take breaks - 2 hours max per session
- Join the community (if available) to discuss

### Don'ts ❌
- Don't skip exercises
- Don't just read - type the code yourself
- Don't peek at solutions immediately
- Don't skip Module 0 (most common mistake!)
- Don't rush through to finish - quality > speed

### When You Get Stuck
1. Re-read the relevant section
2. Check ml_cheatsheet.md
3. Google the error message
4. Try a simpler version first
5. Move on and come back later
6. Ask for help (community, mentor, forums)

---

## After Completing This Course

### You'll Be Ready To:
- Apply for ML engineer roles
- Contribute to ML teams on day one
- Build production ML systems
- Frame and solve business problems with ML
- Collaborate effectively in ML projects

### Next Steps:

#### Immediate (0-1 month):
- Add capstone project to your portfolio
- Update LinkedIn/resume with new skills
- Apply for junior ML engineer roles
- Contribute to open-source ML projects

#### Short-term (1-3 months):
- Build 2-3 more end-to-end projects
- Specialize in a domain (NLP, computer vision, time series)
- Learn advanced topics (neural networks, deep learning)
- Get feedback from senior engineers

#### Medium-term (3-6 months):
- Contribute to production ML systems
- Mentor others learning ML
- Write blog posts about your learning
- Speak at meetups or conferences

### Continuous Learning:
- **Advanced ML**: AutoML, neural architecture search, model compression
- **MLOps**: Kubernetes, feature stores, model registries, CI/CD for ML
- **Specialized Domains**: NLP (transformers), Computer Vision (CNNs), Time Series, Reinforcement Learning
- **Production Engineering**: Distributed training, model serving at scale, GPU optimization
- **Leadership**: Team management, ML strategy, stakeholder management

---

## Success Stories

This course prepares you for real ML engineering roles. Here's what you'll be able to do:

### Junior ML Engineer (0-2 years)
- Build data pipelines
- Engineer features
- Train and evaluate models
- Deploy to staging
- Debug production issues
- Collaborate with senior engineers

### Mid-Level ML Engineer (2-5 years)
- Design ML systems end-to-end
- Lead small projects
- Mentor junior engineers
- Set monitoring strategies
- Influence product decisions
- Code review others' work

### Senior ML Engineer (5+ years)
- Architect ML platforms
- Lead large initiatives
- Set technical direction
- Build and manage teams
- Interface with executives
- Drive ML strategy

**This course covers the foundations for all three levels.**

---

## Frequently Asked Questions

### "Do I need a PhD?"
No. This course teaches practical engineering skills, not research.

### "Do I need to know deep learning?"
No. Most ML at companies is scikit-learn and XGBoost, not neural networks.

### "What if I get stuck?"
- Use ml_cheatsheet.md
- Re-read the relevant module
- Try a simpler version
- Take a break and come back
- Ask for help (if community available)

### "How long does it really take?"
- Fast learners: 16-20 hours over 2-3 weeks
- Average: 20-24 hours over 3-4 weeks
- Thorough learners: 24-28 hours over 4-5 weeks

Quality > speed. Take your time.

### "Can I skip modules?"
- Module 0: DON'T skip - business context is critical
- Modules 1-5: Core technical content, highly recommended
- Modules 6-7: Skip if you already know production ML and Git
- Module 8: Don't skip - this is where everything comes together

### "What if I already know some of this?"
- Skim modules you know
- Do the exercises anyway (they're harder than the content)
- Focus on modules where you're weak
- The capstone will challenge you regardless

### "Is this enough to get a job?"
This course covers the 80-20 most important skills. To get a job, you also need:
- 2-3 projects in your portfolio
- A good resume and LinkedIn
- Interview skills (coding, system design, behavioral)
- Networking and job applications

But yes, this course will prepare you technically for ML engineer roles.

---

## Additional Resources

### Inside This Course:
- **README.md** - Getting started guide
- **CAPSTONE_RUBRIC.md** - Self-assessment rubric
- **ml_cheatsheet.md** - Quick reference
- **progress_tracker.md** - Track your completion

### External Resources:
- **scikit-learn docs**: https://scikit-learn.org
- **pandas docs**: https://pandas.pydata.org
- **MLflow**: https://mlflow.org
- **FastAPI**: https://fastapi.tiangolo.com
- **Pandera**: https://pandera.readthedocs.io

### Communities:
- r/MachineLearning
- r/MLQuestions
- MLOps Community Slack
- Local ML meetups
- Twitter #MLtwitter

---

## Acknowledgments

This course is built on:
- Real ML engineering experience at tech companies
- Feedback from 100+ ML engineers on "what I wish I knew"
- Industry best practices from Google, Netflix, Uber, etc.
- The insight that 60% technical + 20% business + 20% collaboration = success

**Thank you to all the ML engineers who shared what they wish they'd learned earlier!**

---

## Course Version

**Version 2.0** - Complete 8-module course with capstone
**Last Updated**: 2024
**Maintained by**: ML Engineering Community

---

## Ready to Start?

```bash
# Clone the repo
cd marimo-pokemon

# Install dependencies
uv sync

# Start with Module 0
uvx marimo edit 00_ml_in_business.py

# Or open the entire workspace
uvx marimo edit ./
```

**Don't skip Module 0 - business context is where most engineers fail!**

**Happy learning! 🎓 You've got this! 🚀**
