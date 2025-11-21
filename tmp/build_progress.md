# ML Course Build Progress Tracker

**Last Updated**: 2025-11-21
**Goal**: Complete 8-module ML Engineering Course (Option 3: Hybrid Approach)

---

## ✅ Phase 1: Technical Core Complete (Modules 1-5)

**Status**: DONE ✅
**Commit**: 5a17927 - "Phase 1: Complete Technical Core (Modules 1-5 + All Exercises)"
**Date**: 2025-11-21

### Completed:
- [x] Module 1: Data Engineering Foundations + exercises_01.py
- [x] Module 2: EDA & Feature Engineering + exercises_02.py
- [x] Module 3: Model Training & Experimentation + exercises_03.py
- [x] Module 4: Model Evaluation & Validation + exercises_04.py (NEW)
- [x] Module 5: Deployment & Inference
- [x] All modules validated with `uvx marimo check`

### Files Created/Verified:
- `01_data_engineering.py` (21K)
- `02_eda_and_features.py` (27K)
- `03_model_training.py` (29K)
- `04_model_evaluation.py` (17K)
- `05_inference_service.py` (21K)
- `exercises_01.py` (14K)
- `exercises_02.py` (17K)
- `exercises_03.py` (15K)
- `exercises_04.py` (NEW - comprehensive evaluation exercises)

### Technical Coverage:
- ✅ Data pipelines & validation
- ✅ Feature engineering & leakage prevention
- ✅ Systematic experimentation
- ✅ Model evaluation beyond accuracy
- ✅ Production deployment basics

---

## ✅ Phase 2: Module 0 - Business Context COMPLETE

**Status**: DONE ✅
**Commit**: 0befd26 - "Phase 2: Module 0 - ML in Business Context Complete"
**Date**: 2025-11-21

### Completed:
- [x] `00_ml_in_business.py` - When to use ML, stakeholder communication ✅
- [x] `exercises_00.py` - Business case studies and metric selection ✅

### Learning Outcomes:
- Evaluate if ML is appropriate for a problem
- Set realistic stakeholder expectations
- Define success metrics aligned with business goals
- Estimate ML project timelines

### Content Plan:
1. **When to use ML vs rules vs heuristics**
   - Decision framework
   - Cost-benefit analysis
   - "Will simple rules work?" checklist

2. **Setting stakeholder expectations**
   - Communicating uncertainty
   - Timeline estimation
   - Managing technical vs business metrics

3. **Defining success metrics**
   - North star metrics
   - Acceptable error rates
   - Business impact calculations

4. **ML project lifecycle at companies**
   - Team roles and responsibilities
   - Common failure modes
   - Exploration → Production workflow

### Exercises:
- Case study analysis (3 scenarios)
- Write project proposal for non-technical exec
- Convert business requirements to technical metrics
- **Checkpoint**: 1-page ML project brief

**Estimated Time**: 2-3 hours to build, 2 hours for learners

---

## 📋 Phase 3: Module 6 - Production ML & Monitoring (PENDING)

**Status**: Pending
**Target**: Production operations and debugging skills

### To Create:
- [ ] `06_production_monitoring.py` - Monitoring, debugging, incident response
- [ ] `exercises_06.py` - Debug production scenarios

### Learning Outcomes:
- Set up comprehensive monitoring
- Debug production ML systems
- Respond to incidents effectively
- Plan model maintenance

### Content Plan:
1. **Monitoring & Alerting** (60 min)
   - What to monitor (latency, accuracy, drift)
   - Setting up alerts
   - SLAs and SLOs for ML
   - Dashboards for stakeholders

2. **Data Drift Detection** (45 min)
   - Distribution shift detection
   - Feature drift vs concept drift
   - When to retrain
   - Automatic vs manual retraining

3. **Production Debugging Runbook** (60 min) - CRITICAL
   - **Scenario 1**: "Accuracy dropped from 85% to 65%"
     - Data distribution checks
     - Input quality verification
     - Upstream dependency analysis
   - **Scenario 2**: "Latency spiked from 50ms to 2000ms"
     - Model size investigation
     - Batch processing issues
     - Infrastructure problems
   - **Scenario 3**: "All predictions are same class"
     - Preprocessing bugs
     - Model loading issues
     - Data pipeline failures
   - Creating incident reports
   - Blameless postmortems

4. **Model Maintenance** (30 min)
   - Retraining schedules
   - Performance degradation tracking
   - Technical debt management
   - Model deprecation

### Exercises:
- Debug 3 simulated production incidents
- Write incident postmortem
- Design monitoring dashboard

**Estimated Time**: 3-4 hours to build, 3 hours for learners

---

## 📋 Phase 4: Module 7 - Team Collaboration (PENDING)

**Status**: Pending
**Target**: Git workflows, code reviews, teamwork

### To Create:
- [ ] `07_collaboration.py` - Git, code reviews, working with teams
- [ ] `exercises_07.py` - Review ML PRs, write documentation
- [ ] Sample ML pull requests for review practice

### Learning Outcomes:
- Use Git professionally for ML projects
- Conduct ML code reviews
- Work with existing codebases
- Collaborate effectively with ML teams

### Content Plan:
1. **Git Workflows for ML** (45 min)
   - Branch strategies for experiments
   - Committing notebooks vs scripts
   - .gitignore for ML projects
   - PR descriptions for ML changes
   - Semantic commits

2. **Code Reviews for ML** (60 min)
   - Reviewing ML code vs regular code
   - Common ML code smells:
     - Data leakage patterns
     - Reproducibility issues
     - Hardcoded values
     - Missing validation
   - Providing constructive feedback
   - Practice: Review 3 provided ML PRs

3. **Working with Existing ML Systems** (30 min)
   - Reading others' ML code
   - Understanding feature pipelines
   - Safely modifying production models
   - Documentation standards

4. **Pair Programming for ML** (15 min)
   - When to pair on ML problems
   - Debugging together
   - Knowledge sharing

### Exercises:
- Review 3 sample ML pull requests
- Refactor messy ML code
- Write PR description for Module 5 deployment

**Estimated Time**: 2-3 hours to build, 2.5 hours for learners

---

## 📋 Phase 5: Module 8 - Capstone Project (PENDING)

**Status**: Pending
**Target**: End-to-end project demonstrating all skills

### To Create:
- [ ] `08_capstone_project.py` - Pokemon card **price prediction** (regression)
- [ ] Capstone dataset (price data)
- [ ] Evaluation rubric

### Project Scope:
**Scenario**: Building a Pokemon card price predictor

**Deliverables** (5-6 hours):
1. **Module 0 Skills**: Project brief with business justification
2. **Module 1 Skills**: Data validation pipeline
3. **Module 2 Skills**: EDA report + feature engineering
4. **Module 3 Skills**: Experiment tracking (3+ models)
5. **Module 4 Skills**: Model evaluation report with A/B test design
6. **Module 5 Skills**: Deployed API with optimization
7. **Module 6 Skills**: Monitoring plan + incident runbook
8. **Module 7 Skills**: PR-ready code + documentation

### Evaluation Criteria:
- Technical correctness (40%)
- Communication & documentation (30%)
- Production readiness (20%)
- Collaboration readiness (10%)

**Estimated Time**: 4-5 hours to build, 5-6 hours for learners

---

## 📋 Phase 6: Documentation & Polish (PENDING)

**Status**: Pending

### To Create:
- [ ] Update `README.md` with complete 8-module structure
- [ ] Create `COURSE_OUTLINE.md` with full learning path
- [ ] Update `progress_tracker.md` for learners
- [ ] Create `ml_cheatsheet.md` enhancements (if needed)
- [ ] Final integration testing

### Tasks:
- Test all 8 modules run without errors
- Verify all exercises have clear instructions
- Check all TODOs are intentional
- Spell check and grammar pass
- Generate course completion certificate template

**Estimated Time**: 2-3 hours

---

## 🎯 Overall Progress

### Modules Status:
- [x] Module 0: Business Context ✅
- [x] Module 1: Data Engineering ✅
- [x] Module 2: EDA & Features ✅
- [x] Module 3: Model Training ✅
- [x] Module 4: Model Evaluation ✅
- [x] Module 5: Deployment ✅
- [ ] Module 6: Production & Monitoring 🔄 (NEXT)
- [ ] Module 7: Team Collaboration
- [ ] Module 8: Capstone Project

### Overall Completion: 75% (6/8 modules)

### Estimated Remaining Time:
- Module 0: 2-3 hours
- Module 6: 3-4 hours
- Module 7: 2-3 hours
- Module 8: 4-5 hours
- Documentation: 2-3 hours
- **Total**: 13-18 hours remaining

### Course Coverage:
- **Technical Skills**: 85% complete
- **Business Context**: 0% complete (starting next)
- **Production/MLOps**: 40% complete (monitoring pending)
- **Collaboration**: 0% complete (Module 7 pending)

---

## 📊 Gap Analysis: Are We Hitting 80-20?

### Original Gaps Identified:
1. ❌ **Business Context** - Module 0 will fix (0% → 80%)
2. ❌ **Stakeholder Communication** - Distributed across modules (20% → 70%)
3. ❌ **Production Debugging** - Module 6 will fix (40% → 90%)
4. ❌ **Team Collaboration** - Module 7 will fix (10% → 85%)
5. ✅ **Data Engineering** - Already excellent (100%)
6. ✅ **Feature Engineering** - Already excellent (100%)
7. ⚠️ **Experiment Tracking** - Could add MLflow hands-on (30% → would be 90%)
8. ❌ **A/B Testing** - Need to add to Module 4 (0% → 75%)

### After Completion (Projected):
| Skill Area | Current | After v2.0 | Target |
|------------|---------|------------|--------|
| Data pipelines | 100% | 100% | 100% ✅ |
| Stakeholder communication | 20% | 70% | 80% ⚠️ |
| Feature engineering | 100% | 100% | 100% ✅ |
| Production debugging | 40% | 90% | 80% ✅ |
| Model training | 80% | 85% | 80% ✅ |
| Team collaboration | 10% | 85% | 80% ✅ |
| Business alignment | 15% | 80% | 80% ✅ |

**Projected Final Coverage: 85-90% of company ML needs**

---

## 🚀 Next Steps

### Immediate (Today):
1. Create Module 0: ML in Business Context
2. Test and validate
3. Commit Phase 2

### This Week:
4. Create Module 6: Production Monitoring
5. Create Module 7: Team Collaboration
6. Commit Phases 3 & 4

### Next Week:
7. Create Module 8: Capstone Project
8. Update all documentation
9. Final integration testing
10. Commit Phase 5 & Final v2.0

---

## 📝 Notes & Decisions

### Design Decisions:
- **Marimo over Jupyter**: Better for production mindset (reactive, git-friendly)
- **Pokemon theme**: Engaging, complex enough for real ML, simple enough to understand
- **8 modules instead of 5**: Fills critical soft skill gaps
- **Capstone as regression**: Different from classification in main modules

### Key Features:
- All modules have exercises
- Real-world scenarios throughout
- Production code quality
- Stakeholder communication emphasis
- Debugging runbooks with actual scenarios

### Potential Enhancements (Future v3.0):
- Add MLflow/W&B hands-on to Module 3
- Add A/B testing hands-on to Module 4
- Create video walkthroughs
- Add quiz bank
- Create certificate generator

---

**Last Commit**: 5a17927
**Next Milestone**: Module 0 creation
**Target Completion**: 2025-11-25
