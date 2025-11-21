# Professional ML Engineering Onboarding Project
## Pokemon Card Machine Learning - Complete ML Engineering Course

**The comprehensive, production-focused ML course that teaches the 80-20 most important skills for ML engineers at companies**

---

## 🎯 What This Is

A hands-on learning system that teaches you to build production ML systems from scratch, covering both technical skills AND the business/collaboration skills that companies actually need. By the end, you'll be ready to contribute meaningfully to an ML team on day one.

**Themes**: Pokemon card type classification + price prediction
**Format**: Interactive Marimo notebooks with exercises
**Duration**: 20-24 hours over 3-4 weeks
**Outcome**: Production-ready ML engineering skills + business acumen + team collaboration skills

---

## 📚 Status: ✅ COMPLETE - All 8 Modules + Capstone Ready!

### ✅ All Modules Completed

#### Module 0: ML in Business Context (1-2 hours)
- When to use ML (and when not to)
- ROI calculation and business metrics
- Stakeholder communication
- Setting realistic expectations
- **Status**: ✅ Complete with 5 real-world scenarios

#### Module 1: Data Engineering Foundations (2-3 hours)
- Data loading, validation, and quality analysis
- Pandera schema validation
- Data cleaning pipelines
- Pandas vs Polars performance comparison
- **Status**: ✅ Complete with exercises

#### Module 2: EDA & Feature Engineering (3-4 hours)
- Comprehensive exploratory data analysis
- Feature engineering with domain knowledge
- Data leakage prevention
- Scikit-learn preprocessing pipelines
- **Status**: ✅ Complete with exercises

#### Module 3: Model Training & Experimentation (3-4 hours)
- Baseline models and model zoo
- Cross-validation and hyperparameter tuning
- Experiment tracking
- Systematic model selection
- **Status**: ✅ Complete with exercises

#### Module 4: Model Evaluation & Validation (2-3 hours)
- Metrics beyond accuracy
- Confusion matrices and error analysis
- Confidence calibration
- Model cards and documentation
- **Status**: ✅ Complete with exercises

#### Module 5: Deployment & Inference (2-3 hours)
- Model serialization and versioning
- Production inference API
- Input validation and error handling
- Model serving patterns
- **Status**: ✅ Complete with interactive UI

#### Module 6: Production ML & Monitoring (2-3 hours)
- Production debugging runbook
- Data drift detection techniques
- Incident response procedures
- Monitoring strategies and alerts
- **Status**: ✅ Complete with 3 incident scenarios

#### Module 7: Team Collaboration & Code Reviews (2-3 hours)
- Git workflows for ML projects
- ML code review best practices
- Writing effective PR descriptions
- Working with existing ML codebases
- **Status**: ✅ Complete with 3 sample PR reviews

#### Module 8: Capstone Project (4-6 hours)
- End-to-end Pokemon card price prediction
- Integrates all concepts from Modules 0-7
- 9 phases: business → data → features → training → evaluation → deployment → monitoring → docs → reflection
- Self-assessment rubric (100 points)
- **Status**: ✅ Complete with detailed rubric

### 📦 Additional Resources
- ✅ **ML Cheatsheet** - Quick reference for all key concepts
- ✅ **Progress Tracker** - Track your learning journey
- ✅ **CAPSTONE_RUBRIC.md** - 100-point self-assessment rubric
- ✅ **800 Pokemon Cards Dataset** - With price_usd for regression tasks

---

## 🚀 Quick Start

### Prerequisites
- **uv** - Fast Python package installer ([install here](https://docs.astral.sh/uv/))
- Python 3.13+ (uv will handle this if needed)
- That's it! `uvx` will handle everything else

### Get Started (Recommended: 3 Steps)

```bash
# 1. Navigate to the project directory
cd marimo-pokemon

# 2. Install dependencies with uv
uv sync

# 3. Start learning with Workspace Mode (RECOMMENDED!)
uvx marimo edit ./
```

**🌟 Why Workspace Mode (`uvx marimo edit ./`)?**
- Opens the **entire project** in one window
- File browser to easily switch between modules
- No need to remember file names
- See your progress across all modules
- This is the recommended way to use the course!

### Alternative: Run Specific Modules

If you prefer to open individual notebooks:

```bash
# Start with Module 0 (business context)
uvx marimo edit 00_ml_in_business.py

# Or jump to a specific module
uvx marimo edit 03_model_training.py

# View-only mode (read without editing)
uvx marimo run 00_ml_in_business.py
```

### Generate the Dataset

```bash
# Generate the Pokemon card dataset (with price_usd)
uv run python data/generate_dataset.py

# This creates data/pokemon_cards.csv (800 cards)
```

---

## 📌 Key uv/uvx Commands

| Command | What it does | When to use |
|---------|--------------|-------------|
| `uv sync` | Install dependencies | Once at start, or when dependencies change |
| `uvx marimo edit ./` | **Open entire project** | **Recommended way to use the course** |
| `uvx marimo edit <file>` | Open specific notebook | When you know which module you want |
| `uvx marimo run <file>` | View notebook (read-only) | Just reading, not editing |
| `uv run python <file>` | Run Python script | For dataset generation, utilities |

**💡 Pro Tip**: Always use `uvx marimo edit ./` from the project root to get the best experience!

**Why uvx is awesome**:
- No need to install marimo globally
- Automatically uses the right version from your environment
- Works from any directory
- Perfect for ephemeral environments (containers, Codespaces)
- No virtual environment activation needed

---

## 📖 Course Structure

### Complete Learning Path (20-24 hours)

**Phase 1: Business & Technical Foundations (8-10 hours)**
1. **Module 0**: ML in Business Context → `uvx marimo edit 00_ml_in_business.py`
2. **Exercises 0**: Business case studies → `uvx marimo edit exercises_00.py`
3. **Module 1**: Data Engineering → `uvx marimo edit 01_data_engineering.py`
4. **Exercises 1**: Practice validation and pipelines → `uvx marimo edit exercises_01.py`
5. **Module 2**: EDA & Features → `uvx marimo edit 02_eda_and_features.py`
6. **Exercises 2**: Feature engineering competition → `uvx marimo edit exercises_02.py`

**Phase 2: Model Development (8-10 hours)**
7. **Module 3**: Model Training → `uvx marimo edit 03_model_training.py`
8. **Exercises 3**: CV and tuning → `uvx marimo edit exercises_03.py`
9. **Module 4**: Model Evaluation → `uvx marimo edit 04_model_evaluation.py`
10. **Exercises 4**: Metrics and error analysis → `uvx marimo edit exercises_04.py`
11. **Module 5**: Deployment → `uvx marimo edit 05_inference_service.py`

**Phase 3: Production & Collaboration (6-8 hours)**
12. **Module 6**: Production ML & Monitoring → `uvx marimo edit 06_production_monitoring.py`
13. **Exercises 6**: Incident response scenarios → `uvx marimo edit exercises_06.py`
14. **Module 7**: Team Collaboration → `uvx marimo edit 07_collaboration.py`
15. **Exercises 7**: Code review practice → `uvx marimo edit exercises_07.py`

**Phase 4: Capstone (4-6 hours)**
16. **Module 8**: End-to-End Project → `uvx marimo edit 08_capstone.py`
17. **Reference**: Capstone Rubric → `CAPSTONE_RUBRIC.md`
18. **Reference**: ML Cheatsheet → `ml_cheatsheet.md`

### Pro Tip: Workspace Mode
Open all notebooks at once with:
```bash
uvx marimo edit ./
```
This gives you a file browser and lets you switch between modules easily!

---

## 🎯 What You'll Learn

### Technical Skills (Core ML Engineering)
- ✅ Build end-to-end ML systems from data to deployment
- ✅ Write production-quality ML code with proper validation
- ✅ Engineer features using domain knowledge without data leakage
- ✅ Train and evaluate models systematically
- ✅ Deploy models with monitoring and error handling
- ✅ Debug production ML issues using runbooks
- ✅ Detect and respond to data drift

### Business Skills (What Companies Actually Need)
- ✅ Frame ML problems and calculate ROI
- ✅ Communicate with non-technical stakeholders
- ✅ Set realistic expectations and success metrics
- ✅ Know when NOT to use ML
- ✅ Translate business metrics to model metrics

### Collaboration Skills (Working in Teams)
- ✅ Use Git workflows for ML projects
- ✅ Review ML code effectively
- ✅ Write clear PR descriptions
- ✅ Document models with model cards
- ✅ Onboard to existing ML codebases

**You'll be ready to contribute meaningfully to an ML team on day one!**

---

## 📁 Project Structure

```
marimo-pokemon/
├── README.md                          # This file
├── CAPSTONE_RUBRIC.md                # 100-point self-assessment rubric
├── prompt.md                          # Complete course specification
├── progress_tracker.md                # Track your learning
├── ml_cheatsheet.md                   # Quick reference guide
│
├── data/
│   ├── generate_dataset.py           # Dataset generator
│   ├── pokemon_cards.csv             # 800 Pokemon cards (with price_usd)
│   └── clean/                        # Cleaned data (generated)
│
├── 00_ml_in_business.py              # Module 0: Business context
├── exercises_00.py                    # Module 0 exercises
├── 01_data_engineering.py            # Module 1: Data engineering
├── exercises_01.py                    # Module 1 exercises
├── 02_eda_and_features.py            # Module 2: EDA & features
├── exercises_02.py                    # Module 2 exercises
├── 03_model_training.py              # Module 3: Model training
├── exercises_03.py                    # Module 3 exercises
├── 04_model_evaluation.py            # Module 4: Evaluation
├── exercises_04.py                    # Module 4 exercises
├── 05_inference_service.py           # Module 5: Deployment
├── 06_production_monitoring.py       # Module 6: Production & monitoring
├── exercises_06.py                    # Module 6 exercises
├── 07_collaboration.py               # Module 7: Team collaboration
├── exercises_07.py                    # Module 7 exercises
├── 08_capstone.py                    # Module 8: End-to-end capstone
│
└── models/                            # Saved models (generated)
```

---

## 💡 Key Features

### Production-Focused
- Code written to production standards
- Type hints, docstrings, error handling
- Industry best practices throughout

### Hands-On Learning
- 15+ exercises across all modules
- Interactive UIs in Marimo notebooks
- Real-world scenarios and challenges

### Complete Coverage
- Full ML lifecycle from data to deployment
- Both classification and regression examples
- Tools: pandas, polars, scikit-learn, XGBoost

### Professional Content
- Industry context in every module
- Real company examples (Netflix, Google, etc.)
- Common pitfalls and how to avoid them

---

## 🎓 Learning Approach

### Recommended Path
1. **Week 1**: Modules 0-2 + exercises (8-10 hours)
   - Business context, data engineering, feature engineering
2. **Week 2**: Modules 3-5 + exercises (8-10 hours)
   - Model training, evaluation, deployment
3. **Week 3**: Modules 6-7 + exercises (6-8 hours)
   - Production monitoring, team collaboration
4. **Week 4**: Module 8 Capstone (4-6 hours)
   - End-to-end project applying all skills

### Study Tips
- Start with Module 0 - business context is critical!
- Run every code cell and experiment
- Answer socratic questions before moving on
- Complete ALL exercises (they're where real learning happens!)
- Use progress_tracker.md to track completion
- Refer to ml_cheatsheet.md when stuck
- For capstone, use CAPSTONE_RUBRIC.md to self-assess

---

## 🏆 Success Metrics

**You've completed the course when you can**:
- [ ] Frame ML problems with business context and ROI
- [ ] Load, validate, and clean data like a pro
- [ ] Engineer features without data leakage
- [ ] Train multiple models and pick the best systematically
- [ ] Evaluate models with appropriate metrics
- [ ] Deploy a model to production with monitoring
- [ ] Debug production ML systems using runbooks
- [ ] Collaborate effectively with ML teams
- [ ] Score 80+ on the capstone rubric

**Ready for your first ML engineer role!**

### What Companies Are Looking For
This course covers the 80-20 most important skills for ML engineers:
- **Technical Core** (60%): Data → Features → Models → Deployment
- **Business Acumen** (20%): ROI, stakeholder communication, when NOT to use ML
- **Collaboration** (20%): Git workflows, code reviews, documentation

Most courses only teach the technical core. This course teaches all three.

---

## 📚 Additional Resources

- **Testing Results**: `TESTING_RESULTS.md` - ✅ Complete test results (all tests passed!)
- **Testing Guide**: `TESTING_GUIDE.md` - How to test each module
- **Review Summary**: `REVIEW_SUMMARY.md` - Assessment criteria
- **Module 1 Evaluation**: `MODULE_1_EVALUATION.md` - Quality metrics

---

## 🛠️ Tech Stack

- **Python 3.13+**
- **Marimo** - Reactive notebooks (better than Jupyter for production)
- **pandas** - Data manipulation
- **polars** - High-performance data processing
- **scikit-learn** - ML algorithms
- **XGBoost** - Gradient boosting
- **Pandera** - Data validation
- **Matplotlib/Seaborn** - Visualization

---

## 🚀 Getting Started

**Ready to become an ML engineer?**

### Recommended: Workspace Mode

```bash
cd marimo-pokemon
uv sync
uvx marimo edit ./
```

This opens the entire project in one window with a file browser. Start with `00_ml_in_business.py` and work through modules 0-8 in order.

### Alternative Options

**Option 1: Start with Module 0**
```bash
uvx marimo edit 00_ml_in_business.py
```
Start here to understand the business context before diving into code!

**Option 2: Jump to specific modules**
```bash
uvx marimo edit 03_model_training.py   # Jump to specific module
uvx marimo edit 08_capstone.py         # Jump to capstone (after 0-7)
```

**Option 3: View-only mode**
```bash
uvx marimo run 00_ml_in_business.py    # View without editing
```

---

**Happy learning! 🎓**

*P.S. This course will prepare you for the 80-20 most important skills companies need in ML engineers. Use `uvx marimo edit ./` for the best experience, and start with Module 0 - don't skip the business context!*
