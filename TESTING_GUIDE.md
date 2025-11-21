# Testing Guide - Modules 1 & 2

## Quick Start

### 1. Open Module 1: Data Engineering

```bash
uvx marimo edit 01_data_engineering.py
```

**What to look for:**
- ✅ All cells run without errors
- ✅ Data loads correctly (should show 767 cleaned records)
- ✅ Pandera validation catches intentional data quality issues
- ✅ Pandas vs Polars benchmark runs and shows speedup
- ✅ Visualizations render correctly
- ✅ Code is clear and well-documented
- ✅ Industry context is helpful
- ✅ Socratic questions make you think

**Expected runtime:** 2-3 minutes for all cells

---

### 2. Open Module 1 Exercises

```bash
uvx marimo edit exercises_01.py
```

**What to look for:**
- ✅ Instructions are clear
- ✅ TODO comments guide you
- ✅ You understand what to implement
- ✅ Exercises are at appropriate difficulty
- ✅ Time estimates feel accurate

**Don't solve them yet** - just review to see if they make sense

---

### 3. Open Module 2: EDA & Feature Engineering

```bash
uvx marimo edit 02_eda_and_features.py
```

**What to look for:**
- ✅ All cells run without errors
- ✅ EDA visualizations are insightful
- ✅ Feature engineering creates 11 new features
- ✅ Data leakage explanation is clear
- ✅ Pipeline implementation is understandable
- ✅ Train/val/test split happens correctly
- ✅ Feature importance baseline runs
- ✅ Code quality is high

**Expected runtime:** 3-4 minutes for all cells

---

## Detailed Review Checklist

### Module 1: Data Engineering

#### Content Quality
- [ ] **Introduction**: Sets proper expectations for ML engineering?
- [ ] **Data Loading**: Error handling is clear?
- [ ] **Data Quality Issues**: All 5 types explained well?
- [ ] **Pandera Validation**: Schema concept is clear?
- [ ] **Data Cleaning**: Pipeline approach is logical?
- [ ] **Versioning**: Concept of data versioning is introduced?
- [ ] **Performance**: Pandas vs Polars comparison is fair?
- [ ] **Industry Context**: Real-world examples are relevant?
- [ ] **Socratic Questions**: Make you think?
- [ ] **Checkpoint Criteria**: Clear what "mastery" means?

#### Code Quality
- [ ] All code runs without errors
- [ ] Type hints present and helpful
- [ ] Docstrings are clear
- [ ] Comments explain "why" not "what"
- [ ] Variable names are descriptive
- [ ] Functions are modular and reusable

#### Learning Experience
- [ ] Appropriate difficulty (not too easy, not too hard)
- [ ] Logical flow (concepts build on each other)
- [ ] Engaging (not boring to read)
- [ ] Professional tone (respects your intelligence)
- [ ] Time estimate accurate (2-3 hours with exercises)

---

### Module 2: EDA & Feature Engineering

#### Content Quality
- [ ] **Introduction**: Motivates why feature engineering matters?
- [ ] **EDA Section**: Analysis is thorough and insightful?
- [ ] **Target Analysis**: Class imbalance discussed properly?
- [ ] **Feature Distributions**: Visualizations are helpful?
- [ ] **Correlations**: Discussion about what to do with them?
- [ ] **Features vs Target**: Shows features are predictive?
- [ ] **Feature Engineering**: 11 features make sense?
- [ ] **Domain Knowledge**: Pokemon knowledge used effectively?
- [ ] **Data Leakage**: Clear examples and warnings?
- [ ] **Pipelines**: Implementation is clean and correct?
- [ ] **Train/Val/Test Split**: Explanation of why this way?
- [ ] **Feature Importance**: Preview is interesting?

#### Code Quality
- [ ] All code runs without errors
- [ ] Visualizations render properly
- [ ] Feature engineering function is well-documented
- [ ] Pipeline prevents leakage correctly
- [ ] No deprecated sklearn warnings
- [ ] Clean separation of train/val/test

#### Learning Experience
- [ ] EDA insights are "aha moments"
- [ ] Feature engineering feels creative
- [ ] Data leakage warning is impactful
- [ ] Pipeline approach makes sense
- [ ] Ready to move to modeling
- [ ] Time estimate accurate (3-4 hours with exercises)

---

## Common Issues & Solutions

### Issue: Dataset not found
**Solution**: Run `uv run python data/generate_dataset.py` first

### Issue: Clean dataset not found (Module 2)
**Solution**: Run Module 1 first to generate the cleaned dataset

### Issue: Visualizations not showing
**Solution**: Make sure you're using `uvx marimo edit` not `uvx marimo run`

### Issue: XGBoost import error
**Solution**: Only needed for Module 3. For now, you can ignore it.
For Module 3, run: `brew install libomp` (Mac) or install OpenMP for your system

---

## Feedback Framework

### What's Working Well?
Consider:
- Content clarity
- Code quality
- Visualizations
- Industry context
- Difficulty level
- Engagement
- Practical value

### What Could Be Improved?
Consider:
- Confusing sections
- Missing explanations
- Too fast/slow pacing
- Code that doesn't work
- Unclear instructions
- Better examples needed
- More/less detail

### Specific Questions to Answer

1. **Overall Assessment**:
   - Are Modules 1-2 at the right difficulty for a software engineer new to ML?
   - Do they feel "professional" vs "tutorial-like"?

2. **Content Depth**:
   - Too much detail or not enough?
   - Industry context helpful or distracting?
   - Code examples clear?

3. **Learning Experience**:
   - Do you feel you're learning valuable skills?
   - Are "aha moments" happening?
   - Do socratic questions help?

4. **Practical Value**:
   - Could you apply these concepts to a real project?
   - Do you understand WHY not just HOW?
   - Would this prepare you for an ML team?

5. **Gaps or Missing Content**:
   - What topics are missing?
   - What would you add?
   - What could be removed?

6. **Exercise Quality** (after trying at least one):
   - Are TODOs helpful?
   - Is difficulty appropriate?
   - Are instructions clear?
   - Is the learning objective met?

---

## Quick Test Commands

Run all these to verify everything works:

```bash
# Generate dataset
uv run python data/generate_dataset.py

# Test Module 1 in command line (just imports)
uv run python -c "import pandas as pd; df = pd.read_csv('data/pokemon_cards.csv'); print(f'✅ Dataset has {len(df)} records')"

# Open Module 1 for interactive testing
uvx marimo edit 01_data_engineering.py

# Open Module 2 for interactive testing
uvx marimo edit 02_eda_and_features.py

# Open exercises
uvx marimo edit exercises_01.py

# Or open everything at once (workspace mode)
uvx marimo edit ./
```

---

## What to Share in Your Feedback

### Format (use whatever works for you):

**Module 1 Feedback:**
- Overall impression:
- What worked well:
- What needs improvement:
- Specific issues (if any):
- Ready to continue? (Yes/No)

**Module 2 Feedback:**
- Overall impression:
- What worked well:
- What needs improvement:
- Specific issues (if any):
- Ready to continue? (Yes/No)

**General Feedback:**
- Right difficulty level?
- Right tone/style?
- Should I continue building Modules 3-5?
- Any changes to the overall approach?

---

## Next Steps After Review

Once you've reviewed and provided feedback:

1. **If major changes needed**: I'll revise Modules 1-2 based on your feedback
2. **If minor changes needed**: I'll note them and incorporate in Modules 3-5
3. **If good to go**: I'll proceed with:
   - Module 2 exercises
   - Module 3: Model Training & Experimentation
   - Module 4: Model Evaluation & Validation
   - Module 5: Deployment & Inference
   - ML Cheatsheet
   - Setup instructions & README

Take your time! Quality feedback now will make Modules 3-5 much better.
