"""
Test script to verify all dependencies and basic functionality.
"""

print("Testing ML Engineering Course Setup...")
print("=" * 60)

# Test 1: Import all required libraries
print("\n1. Testing library imports...")
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandera as pa
    import marimo
    print("   ✅ All core libraries imported successfully")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    exit(1)

# Test 2: Load dataset
print("\n2. Testing dataset...")
try:
    from pathlib import Path
    DATA_PATH = Path("data/pokemon_cards.csv")
    if not DATA_PATH.exists():
        print(f"   ❌ Dataset not found at {DATA_PATH}")
        print("   Run: uv run python data/generate_dataset.py")
        exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"   ✅ Dataset loaded: {len(df)} records, {df.shape[1]} columns")
except Exception as e:
    print(f"   ❌ Dataset error: {e}")
    exit(1)

# Test 3: Basic data operations
print("\n3. Testing data operations...")
try:
    # Basic stats
    assert df['type'].nunique() > 10, "Too few Pokemon types"
    assert len(df) > 700, "Dataset too small"

    # Feature engineering
    df['total_stats'] = df['hp'] + df['attack'] + df['defense']
    assert 'total_stats' in df.columns

    print("   ✅ Data operations working")
except Exception as e:
    print(f"   ❌ Data operation error: {e}")
    exit(1)

# Test 4: Train a simple model
print("\n4. Testing model training...")
try:
    feature_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    X = df[feature_cols].fillna(0)
    y = df['type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    score = model.score(X_test_scaled, y_test)

    print(f"   ✅ Model trained successfully (accuracy: {score:.2%})")
except Exception as e:
    print(f"   ❌ Model training error: {e}")
    exit(1)

# Test 5: Data validation
print("\n5. Testing data validation...")
try:
    schema = pa.DataFrameSchema({
        "hp": pa.Column(float, nullable=True),
        "attack": pa.Column(int, pa.Check.ge(0)),
        "type": pa.Column(str)
    }, strict=False)

    # This should work
    validated = schema.validate(df.head())
    print("   ✅ Data validation working")
except Exception as e:
    print(f"   ❌ Validation error: {e}")
    exit(1)

# Test 6: Visualization
print("\n6. Testing visualization...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    fig, ax = plt.subplots(figsize=(8, 6))
    df['type'].value_counts().head(10).plot(kind='bar', ax=ax)
    ax.set_title('Top 10 Pokemon Types')
    plt.close()

    print("   ✅ Visualization working")
except Exception as e:
    print(f"   ❌ Visualization error: {e}")
    exit(1)

# Test 7: Check marimo notebooks exist
print("\n7. Testing notebook files...")
try:
    notebooks = [
        '01_data_engineering.py',
        '02_eda_and_features.py',
        '03_model_training.py',
        '04_model_evaluation.py',
        '05_inference_service.py'
    ]

    for nb in notebooks:
        if not Path(nb).exists():
            print(f"   ❌ Missing notebook: {nb}")
            exit(1)

    print(f"   ✅ All {len(notebooks)} notebooks found")
except Exception as e:
    print(f"   ❌ Notebook check error: {e}")
    exit(1)

# Test 8: Check XGBoost (optional)
print("\n8. Testing XGBoost (optional)...")
try:
    import xgboost as xgb
    print("   ✅ XGBoost available")
except Exception as e:
    print("   ⚠️  XGBoost not available")
    print("   Note: XGBoost only needed for Module 3")
    print("   To install: brew install libomp (Mac) or apt-get install libgomp1 (Linux)")
    # Don't exit - XGBoost is optional for most modules

# Test 9: Check polars (performance library)
print("\n9. Testing Polars...")
try:
    import polars as pl
    df_polars = pl.from_pandas(df.head())
    print("   ✅ Polars working")
except Exception as e:
    print(f"   ❌ Polars error: {e}")

print("\n" + "=" * 60)
print("✅ SETUP TEST COMPLETE!")
print("\nYour environment is ready. Start learning with:")
print("  marimo edit 01_data_engineering.py")
print("=" * 60)
