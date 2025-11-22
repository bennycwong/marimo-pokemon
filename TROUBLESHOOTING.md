# Troubleshooting Guide

Common issues and solutions for the ML Engineering Course.

---

## 🚀 Quick Fixes

### "I just want to start!"

```bash
cd marimo-pokemon
uv sync
uv run python data/generate_dataset.py
uvx marimo edit ./
```

If that doesn't work, keep reading!

---

## 📋 Before You Start

### Run the Setup Test

```bash
uv run python test_setup.py
```

This will check all dependencies and tell you exactly what's wrong.

---

## Common Issues

### 1. uv/uvx not found

**Error**:
```
bash: uv: command not found
bash: uvx: command not found
```

**Solution**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on Mac with Homebrew
brew install uv

# Verify installation
uv --version
```

**Still not working?**
- Restart your terminal
- Check if it's in your PATH: `echo $PATH`
- On Mac, you may need to add to shell profile: `export PATH="$HOME/.cargo/bin:$PATH"`

---

### 2. Dataset not found

**Error**:
```
FileNotFoundError: data/pokemon_cards.csv
```

**Solution**:
```bash
# Generate the dataset
uv run python data/generate_dataset.py

# Verify it exists
ls data/pokemon_cards.csv

# Should show: data/pokemon_cards.csv
```

---

### 3. XGBoost installation fails (Mac M1/M2)

**Error**:
```
ImportError: dlopen(..._xgboost.so): Library not loaded: /opt/homebrew/opt/libomp/lib/libomp.dylib
```

**Solution**:
```bash
# Install OpenMP library
brew install libomp

# If that doesn't work, reinstall XGBoost
uv pip install --force-reinstall xgboost
```

**Still not working?**
XGBoost is only needed for Module 3. You can proceed with other modules and come back to this.

---

### 4. Polars import error

**Error**:
```
ImportError: cannot import name 'polars'
```

**Solution**:
```bash
# Reinstall polars
uv sync --reinstall-package polars

# Or install manually
uv pip install polars
```

---

### 5. Marimo won't start

**Error**:
```
marimo: command not found
```

**Solution**:
Use `uvx` instead of `marimo` directly:

```bash
# ❌ Don't do this
marimo edit ./

# ✅ Do this
uvx marimo edit ./
```

`uvx` automatically finds and runs marimo without global installation.

---

### 6. Module import errors

**Error**:
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution**:
```bash
# Re-sync dependencies
uv sync

# If that doesn't work, clear cache and reinstall
rm -rf .venv
uv sync
```

---

### 7. "Clean" data not found

**Error**:
```
FileNotFoundError: data/clean/pokemon_cards_clean_latest.csv
```

**Solution**:
This file is generated when you run Module 1. You need to:
1. Open Module 1: `uvx marimo edit 01_data_engineering.py`
2. Run all cells to generate the clean data
3. Then other modules will work

**Or skip to specific modules**:
Modules 0 and 8 don't require the clean data and can be run independently.

---

### 8. MLflow import error

**Error**:
```
ModuleNotFoundError: No module named 'mlflow'
```

**Solution**:
```bash
# MLflow should be in dependencies
uv sync

# Verify it's installed
uv run python -c "import mlflow; print(mlflow.__version__)"

# If not, install manually
uv pip install mlflow
```

MLflow is needed for Module 3 (experiment tracking).

---

### 9. Pydantic validation errors

**Error**:
```
ValidationError: X validation errors for PokemonCard
```

**Solution**:
This is expected! It means your input validation is working. Check:
- Are all required fields present?
- Are values in valid ranges? (e.g., hp between 1-300)
- Are types correct? (int vs float vs string)

---

### 10. Port already in use (FastAPI)

**Error**:
```
OSError: [Errno 48] Address already in use
```

**Solution**:
```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
uvicorn app:app --port 8001
```

---

### 11. Git issues

**Error**:
```
fatal: not a git repository
```

**Solution**:
This is fine! The course doesn't require git. But if you want version control:

```bash
git init
git add .
git commit -m "Initial commit"
```

---

### 12. Jupyter notebook vs Marimo

**Error**:
"I'm used to Jupyter, this looks different!"

**Solution**:
Marimo is similar but better for production ML:
- Files are pure Python (git-friendly)
- Reactive execution (cells auto-update)
- No hidden state (reproducible)

Key differences:
- Use `uvx marimo edit` instead of `jupyter notebook`
- Use `mo.` for widgets instead of `ipywidgets`
- Save with Cmd/Ctrl+S (auto-formats your code!)

---

## Environment Issues

### Python version mismatch

**Error**:
```
This project requires Python >=3.13
```

**Solution**:
```bash
# Check your version
python3 --version

# Install Python 3.13+ using uv
uv python install 3.13

# Or use homebrew (Mac)
brew install python@3.13
```

---

### Virtual environment issues

**Error**:
"My virtual environment is messed up"

**Solution**:
```bash
# Delete and recreate
rm -rf .venv
uv sync

# This will recreate everything fresh
```

---

### Permission errors

**Error**:
```
PermissionError: [Errno 13] Permission denied
```

**Solution**:
```bash
# Don't use sudo with uv!
# Instead, check file permissions
ls -la

# If files are owned by root, fix it
sudo chown -R $USER:$USER .
```

---

## Performance Issues

### Slow notebook loading

**Issue**: Notebooks take forever to load

**Solution**:
- Don't load all 8 modules at once (use workspace mode to switch)
- Close unused notebooks
- Restart marimo: Stop (Ctrl+C) and re-run `uvx marimo edit ./`

---

### Out of memory

**Issue**: Kernel crashes with memory error

**Solution**:
- Use smaller dataset samples for practice
- Close other applications
- In Module 1, prefer polars over pandas for large data
- Use `del` to remove large variables you don't need

```python
# Free memory
del large_dataframe
import gc
gc.collect()
```

---

## Data Issues

### Dataset has wrong number of rows

**Issue**: Expected 800 rows, got different number

**Solution**:
```bash
# Regenerate the dataset
uv run python data/generate_dataset.py

# Verify
wc -l data/pokemon_cards.csv
# Should show: 801 (800 data + 1 header)
```

---

### Missing columns in dataset

**Issue**: Column 'price_usd' not found

**Solution**:
You may have an old dataset. Regenerate it:
```bash
rm data/pokemon_cards.csv
uv run python data/generate_dataset.py
```

The new dataset includes `price_usd` for the capstone project (Module 8).

---

## Module-Specific Issues

### Module 0: Business Context
No technical issues - this is all conceptual!

### Module 1: Data Engineering
- **Clean data not found**: Run all cells to generate `data/clean/`
- **Pandera validation fails**: This is expected for dirty data exercises

### Module 2: EDA & Features
- **Feature names don't match**: Make sure you ran Module 1 first
- **Data leakage detected**: Good! That's the exercise :)

### Module 3: Model Training
- **XGBoost not working**: See XGBoost troubleshooting above
- **MLflow not found**: See MLflow troubleshooting above
- **Training takes forever**: Use smaller sample for practice: `df.sample(100)`

### Module 4: Model Evaluation
- **No saved model**: Run Module 3 first to train and save model
- **Confusion matrix error**: Check class names match

### Module 5: Deployment
- **FastAPI not working**: Make sure `fastapi` and `uvicorn` are installed: `uv sync`
- **Pydantic errors**: This is expected - it means validation is working!

### Module 6: Production Monitoring
- **scipy not found**: Run `uv sync` to install
- **Drift detection fails**: Need both training and production data

### Module 7: Collaboration
- No technical dependencies - mostly conceptual!

### Module 8: Capstone
- **Missing price column**: Regenerate dataset (see above)
- **Overwhelmed**: Break it into 9 phases, one at a time!

---

## Still Stuck?

### Debug Checklist

1. ✅ Ran `uv sync`?
2. ✅ Generated dataset with `uv run python data/generate_dataset.py`?
3. ✅ Ran `uv run python test_setup.py` to verify setup?
4. ✅ Using `uvx marimo edit ./` from project root?
5. ✅ Ran previous modules to generate required data?
6. ✅ Read the error message carefully?

### Get Help

1. **Check the error message**: Most errors tell you exactly what's wrong
2. **Run test_setup.py**: `uv run python test_setup.py`
3. **Google the error**: You're probably not the first person to hit this
4. **Check course documentation**:
   - README.md - Setup instructions
   - COURSE_OUTLINE.md - Course structure
   - progress_tracker.md - Track your progress

### Report Issues

If you found a bug in the course materials:
1. Check if it's a known issue
2. Include:
   - Your OS (Mac, Linux, Windows)
   - Python version: `python --version`
   - uv version: `uv --version`
   - Error message (full traceback)
   - What you were trying to do
3. File an issue on GitHub (if applicable)

---

## Preventive Tips

### Do This ✅

- Run `uv sync` when you first clone the repo
- Generate dataset before starting: `uv run python data/generate_dataset.py`
- Save your work frequently (Cmd/Ctrl+S)
- Complete modules in order (0→1→2→3→4→5→6→7→8)
- Read error messages carefully - they usually tell you what's wrong
- Use `uvx marimo edit ./` for best experience

### Don't Do This ❌

- Don't use `pip install` directly - use `uv`
- Don't skip Module 1 - it generates required data
- Don't skip the exercises - that's where learning happens
- Don't jump to Module 8 without doing earlier modules
- Don't use `sudo` with uv commands
- Don't commit `.venv/` or `__pycache__/` to git (already in .gitignore)

---

## Emergency Reset

If everything is broken and you want to start fresh:

```bash
# 1. Save your work!
git add .
git commit -m "Backup before reset"

# 2. Clean everything
rm -rf .venv
rm -rf __pycache__
rm -rf data/clean
rm -rf models

# 3. Reinstall
uv sync

# 4. Regenerate data
uv run python data/generate_dataset.py

# 5. Test setup
uv run python test_setup.py

# 6. Start fresh
uvx marimo edit ./
```

---

## Platform-Specific Issues

### macOS

**M1/M2 Chips**:
- XGBoost may need OpenMP: `brew install libomp`
- Some packages need Rosetta: Install with `softwareupdate --install-rosetta`

**Permissions**:
- If you get permission errors, don't use `sudo`
- Fix ownership: `sudo chown -R $USER:$USER ~/.local`

### Linux

**Missing system libraries**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.13-dev python3.13-venv

# For XGBoost
sudo apt-get install libgomp1
```

### Windows

**Best option**: Use WSL2 (Windows Subsystem for Linux)
```bash
wsl --install
# Then follow Linux instructions above
```

**Native Windows**:
- Install Python 3.13 from python.org
- Install uv: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Some packages may have issues - WSL2 is recommended

---

## FAQ

**Q: Do I need to install marimo globally?**
A: No! Use `uvx marimo edit ./` - it runs marimo without global installation.

**Q: Can I use Jupyter instead of Marimo?**
A: The course is designed for Marimo. Jupyter would require conversion and you'd miss reactive features.

**Q: How much disk space do I need?**
A: ~500MB for all dependencies + data.

**Q: Can I run this on Google Colab?**
A: No, Marimo notebooks require a local environment. Use your computer or a cloud VM.

**Q: Do I need a GPU?**
A: No! All models run fine on CPU.

**Q: How long does setup take?**
A: 5-10 minutes (uv is very fast!)

**Q: Can I skip modules?**
A: Module 0 is critical for context. Modules 1-5 build on each other. Don't skip!

---

**Remember**: Most issues are fixed with `uv sync` and regenerating the dataset! 🎓
