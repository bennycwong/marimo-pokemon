# Professional ML Engineering Onboarding Project
## Pokemon Card Type Classification

**A complete, production-focused ML course for software engineers transitioning to ML**

---

## 🎯 What This Is

A hands-on learning system that teaches you to build production ML systems from scratch. By the end, you'll be able to contribute to an ML team on day one.

**Theme**: Pokemon card type prediction (classification problem)  
**Format**: Interactive Marimo notebooks with exercises  
**Duration**: 12-16 hours over 2-3 weeks  
**Outcome**: Production-ready ML engineering skills

---

## 📚 Status: ✅ COMPLETE - All 5 Modules Ready!

### ✅ All Modules Completed

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
- **Status**: ✅ Complete

#### Module 5: Deployment & Inference (2-3 hours)
- Model serialization and versioning
- Production inference API
- Input validation and error handling
- Monitoring and data drift detection
- **Status**: ✅ Complete with interactive UI

### 📦 Additional Resources
- ✅ **ML Cheatsheet** - Quick reference for all key concepts
- ✅ **Progress Tracker** - Track your learning journey
- ✅ **800 Pokemon Cards Dataset** - With intentional quality issues for learning

---

## 🚀 Quick Start

### Prerequisites
- **uv** - Fast Python package installer ([install here](https://docs.astral.sh/uv/))
- That's it! `uvx` will handle everything else

### Get Started

```bash
# Clone or navigate to the project
cd marimo-pokemon

# Install dependencies
uv sync

# Generate the dataset (if not already done)
uv run python data/generate_dataset.py

# Start learning! Use uvx to run marimo (no global install needed)
uvx marimo edit 01_data_engineering.py  # Begin with Module 1

# Or open the entire project as a workspace
uvx marimo edit ./
```

**Why uvx?**
- No need to install marimo globally
- Automatically uses the right version
- Works from any directory
- Perfect for ephemeral environments

---

## 📖 Course Structure

### Complete Learning Path (12-16 hours)

1. **Module 1**: Data Engineering → `uvx marimo edit 01_data_engineering.py`
2. **Exercises 1**: Practice validation and pipelines → `uvx marimo edit exercises_01.py`
3. **Module 2**: EDA & Features → `uvx marimo edit 02_eda_and_features.py`
4. **Exercises 2**: Feature engineering competition → `uvx marimo edit exercises_02.py`
5. **Module 3**: Model Training → `uvx marimo edit 03_model_training.py`
6. **Exercises 3**: CV and tuning → `uvx marimo edit exercises_03.py`
7. **Module 4**: Model Evaluation → `uvx marimo edit 04_model_evaluation.py`
8. **Module 5**: Deployment → `uvx marimo edit 05_inference_service.py`
9. **Reference**: ML Cheatsheet → `ml_cheatsheet.md`

### Pro Tip: Workspace Mode
Open all notebooks at once with:
```bash
uvx marimo edit ./
```
This gives you a file browser and lets you switch between modules easily!

---

## 🎯 What You'll Learn

By completing this course, you'll be able to:

- ✅ Build end-to-end ML systems from data to deployment
- ✅ Write production-quality ML code with proper validation
- ✅ Engineer features using domain knowledge
- ✅ Train and evaluate models systematically
- ✅ Deploy models with monitoring and error handling
- ✅ Debug ML issues like a professional
- ✅ Communicate effectively with stakeholders

**You'll be ready to contribute to an ML team on day one!**

---

## 📁 Project Structure

```
marimo-pokemon/
├── README.md                          # This file
├── prompt.md                          # Complete course specification
├── progress_tracker.md                # Track your learning
├── ml_cheatsheet.md                   # Quick reference guide
│
├── data/
│   ├── generate_dataset.py           # Dataset generator
│   ├── pokemon_cards.csv             # 800 Pokemon cards
│   └── clean/                        # Cleaned data (generated)
│
├── 01_data_engineering.py            # Module 1 main
├── exercises_01.py                    # Module 1 exercises
├── 02_eda_and_features.py            # Module 2 main
├── exercises_02.py                    # Module 2 exercises
├── 03_model_training.py              # Module 3 main
├── exercises_03.py                    # Module 3 exercises
├── 04_model_evaluation.py            # Module 4 main
├── 05_inference_service.py           # Module 5 main
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
1. **Week 1**: Modules 1-2 + exercises (6-8 hours)
2. **Week 2**: Modules 3-4 + exercises (6-8 hours)
3. **Week 3**: Module 5 + review cheatsheet (3-4 hours)

### Study Tips
- Run every code cell and experiment
- Answer socratic questions before moving on
- Complete exercises (don't skip!)
- Use progress_tracker.md to track completion
- Refer to ml_cheatsheet.md when stuck

---

## 🏆 Success Metrics

**You've completed the course when you can**:
- [ ] Load, validate, and clean data like a pro
- [ ] Engineer features from domain knowledge
- [ ] Train multiple models and pick the best
- [ ] Evaluate models beyond accuracy
- [ ] Deploy a model to production
- [ ] Debug ML systems systematically

**Ready for your first ML role!**

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

### Option 1: Start with Module 1
```bash
uvx marimo edit 01_data_engineering.py
```

### Option 2: Open workspace (recommended)
```bash
uvx marimo edit ./
```
Then select `01_data_engineering.py` from the file browser.

### Option 3: Run specific modules
```bash
uvx marimo run 01_data_engineering.py  # View-only mode
```

---

**Happy learning! 🎓**
