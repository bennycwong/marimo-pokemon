"""
Module 1: Data Engineering Foundations
======================================

Professional ML Engineering Onboarding Project
Pokemon Card Type Classification

Learning Objectives:
- Understand data as code
- Master data quality fundamentals
- Build reproducible data pipelines
- Compare pandas vs polars performance

Duration: 2-3 hours
"""

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        # Module 1: Data Engineering Foundations

        **Welcome to your ML engineering journey!**

        In this module, you'll learn that **data engineering is the foundation of any ML system**.
        In industry, data engineering typically takes 60-80% of an ML engineer's time.

        ## What You'll Learn

        1. **Treat Data as Code**: Data deserves the same rigor as application code
        2. **Data Quality**: Identify and fix data quality issues systematically
        3. **Reproducibility**: Build pipelines that produce identical results
        4. **Performance**: Compare pandas vs polars for different scenarios

        ## Industry Context

        > "If I had an hour to solve a problem, I'd spend 55 minutes on data quality and 5 minutes on the model."
        > — Adapted from Einstein, but every ML engineer agrees

        **Why data engineering matters:**
        - Bad data → Bad models (no matter how sophisticated your algorithm)
        - Data bugs are hard to detect (model trains successfully, but learns wrong patterns)
        - Production failures are usually data issues, not model issues
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Section 1: Loading Data (The Right Way)

        Let's start by loading our Pokemon card dataset. But we won't just load it —
        we'll do it **professionally** with proper validation and error handling.
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from typing import Optional
    import warnings
    warnings.filterwarnings('ignore')
    return Optional, Path, np, pd, warnings


@app.cell
def __(Path, pd):
    # File path configuration
    DATA_DIR = Path("data")
    DATASET_PATH = DATA_DIR / "pokemon_cards.csv"

    def load_pokemon_data(filepath: Path) -> pd.DataFrame:
        """
        Load Pokemon card dataset with error handling.

        Args:
            filepath: Path to the CSV file

        Returns:
            DataFrame with Pokemon card data

        Raises:
            FileNotFoundError: If the data file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found at {filepath}")

        df = pd.read_csv(filepath)

        if df.empty:
            raise pd.errors.EmptyDataError("Dataset is empty")

        return df

    # Load the data
    df_raw = load_pokemon_data(DATASET_PATH)

    print(f"✅ Loaded {len(df_raw)} Pokemon cards")
    print(f"✅ Dataset shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    return DATA_DIR, DATASET_PATH, df_raw, load_pokemon_data


@app.cell
def __(df_raw, mo):
    mo.md(
        f"""
        ### 🔍 First Look at the Data

        We loaded **{len(df_raw)} records** with **{df_raw.shape[1]} features**.
        Let's examine the first few rows:
        """
    )
    return


@app.cell
def __(df_raw, mo):
    # Display first few rows
    mo.ui.table(df_raw.head(10), selection=None)
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### 📊 Initial Data Exploration

        Before we do anything else, let's understand what we're working with.
        This is called **Exploratory Data Analysis (EDA)**, though for now we're just getting oriented.
        """
    )
    return


@app.cell
def __(df_raw):
    # Basic dataset information
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    df_raw.info()
    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    print(df_raw.describe())
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### 🚨 Did You Notice Anything Wrong?

        Look at the `.info()` output above carefully:
        - `hp`: 784 non-null out of 800 → **16 missing values**
        - `price_usd`: 776 non-null out of 800 → **24 missing values**

        **This is realistic!** Real-world datasets are messy. Let's investigate further.

        ---
        ## Section 2: Data Quality Issues

        Now let's systematically check for common data quality problems:
        1. Missing values
        2. Duplicate records
        3. Inconsistent values (e.g., capitalization)
        4. Outliers
        5. Data type issues
        """
    )
    return


@app.cell
def __(df_raw):
    # Check for missing values
    missing_summary = df_raw.isnull().sum()
    missing_pct = (df_raw.isnull().sum() / len(df_raw) * 100).round(2)

    missing_df = pd.DataFrame({
        'Column': missing_summary.index,
        'Missing Count': missing_summary.values,
        'Missing %': missing_pct.values
    })

    print("=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)
    print(missing_df[missing_df['Missing Count'] > 0].to_string(index=False))
    return missing_df, missing_pct, missing_summary


@app.cell
def __(df_raw):
    # Check for duplicate card IDs (should be unique!)
    duplicate_ids = df_raw['card_id'].duplicated().sum()
    print(f"\n⚠️  Duplicate card IDs found: {duplicate_ids}")

    if duplicate_ids > 0:
        print("\nDuplicate records:")
        duplicate_records = df_raw[df_raw['card_id'].duplicated(keep=False)].sort_values('card_id')
        print(duplicate_records[['card_id', 'name', 'type', 'hp']].head(10))
    return duplicate_ids, duplicate_records


@app.cell
def __(df_raw):
    # Check for inconsistent capitalization in categorical columns
    print("\n" + "=" * 60)
    print("CATEGORICAL COLUMN VALUE COUNTS")
    print("=" * 60)

    print("\n--- Type Distribution ---")
    print(df_raw['type'].value_counts())

    print("\n⚠️  Notice the lowercase values? Data entry inconsistency!")
    return


@app.cell
def __(df_raw, np):
    # Check for outliers in price using IQR method
    Q1 = df_raw['price_usd'].quantile(0.25)
    Q3 = df_raw['price_usd'].quantile(0.75)
    IQR_val = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR_val
    upper_bound = Q3 + 1.5 * IQR_val

    outliers = df_raw[(df_raw['price_usd'] < lower_bound) | (df_raw['price_usd'] > upper_bound)]

    print("\n" + "=" * 60)
    print("OUTLIER DETECTION (Price)")
    print("=" * 60)
    print(f"Lower bound: ${lower_bound:.2f}")
    print(f"Upper bound: ${upper_bound:.2f}")
    print(f"Outliers found: {len(outliers)}")
    print("\nTop 10 most expensive cards (potential outliers):")
    print(df_raw.nlargest(10, 'price_usd')[['name', 'type', 'rarity', 'is_legendary', 'price_usd']])
    return IQR_val, Q1, Q3, lower_bound, outliers, upper_bound


@app.cell
def __(mo):
    mo.md(
        """
        ### 💡 Industry Insight: Data Quality Issues are Normal

        **What we found:**
        - ✅ Missing values in HP and price
        - ✅ Duplicate card IDs (data entry errors)
        - ✅ Inconsistent capitalization in `type` column
        - ✅ Price outliers (some cards 10x more expensive)

        **In production, you'll see:**
        - Database connection failures → missing data
        - Multiple data sources → inconsistent formats
        - Manual data entry → typos and duplicates
        - Edge cases → outliers that break models

        **The key question:** Are these errors, or valid data points?
        - Duplicates: Probably errors
        - Missing values: Depends (could be "unknown" vs "not collected")
        - Outliers: Rare cards ARE expensive (domain knowledge!)
        - Inconsistent case: Definitely an error

        ---
        ## Section 3: Data Validation with Pandera

        Manual checks are good for exploration, but in production we need **automated validation**.
        Let's use **Pandera** to define a schema that our data must conform to.
        """
    )
    return


@app.cell
def __():
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    return Check, Column, DataFrameSchema, pa


@app.cell
def __(Check, Column, DataFrameSchema):
    # Define expected schema for Pokemon card data
    pokemon_schema = DataFrameSchema(
        columns={
            "card_id": Column(str, unique=True, nullable=False),
            "name": Column(str, nullable=False),
            "type": Column(str, nullable=False, checks=[
                Check.str_matches(r"^[A-Z][a-z]+$",
                                error="Type must be properly capitalized (e.g., 'Fire', not 'fire')")
            ]),
            "hp": Column(int, nullable=True, checks=[
                Check.greater_than_or_equal_to(20),
                Check.less_than_or_equal_to(200)
            ]),
            "attack": Column(int, nullable=False, checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(200)
            ]),
            "defense": Column(int, nullable=False),
            "sp_attack": Column(int, nullable=False),
            "sp_defense": Column(int, nullable=False),
            "speed": Column(int, nullable=False),
            "generation": Column(int, nullable=False, checks=[
                Check.isin([1, 2, 3, 4, 5, 6, 7, 8, 9])
            ]),
            "is_legendary": Column(bool, nullable=False),
            "rarity": Column(str, nullable=False, checks=[
                Check.isin(['Common', 'Uncommon', 'Rare', 'Ultra Rare', 'Secret Rare'])
            ]),
            "price_usd": Column(float, nullable=True, checks=[
                Check.greater_than(0)
            ])
        },
        strict=False,  # Allow additional columns
        coerce=True    # Try to convert types automatically
    )

    print("✅ Schema defined")
    print("\nThis schema enforces:")
    print("- card_id must be unique")
    print("- type must be properly capitalized")
    print("- Stats must be in reasonable ranges")
    print("- generation must be 1-9")
    print("- rarity must be one of the expected values")
    return (pokemon_schema,)


@app.cell
def __(df_raw, pokemon_schema):
    # Try to validate raw data (it will fail!)
    print("=" * 60)
    print("VALIDATING RAW DATA")
    print("=" * 60)

    try:
        pokemon_schema.validate(df_raw, lazy=True)
        print("✅ Data is valid!")
    except pa.errors.SchemaErrors as e:
        print("❌ Validation failed!\n")
        print(f"Number of errors: {len(e.failure_cases)}")
        print("\nFirst 10 validation errors:")
        print(e.failure_cases.head(10))
    return (e,)


@app.cell
def __(mo):
    mo.md(
        """
        ### 🔧 Data Cleaning Pipeline

        Now that we know what's wrong, let's fix it systematically.
        We'll create a **data cleaning pipeline** that's:
        - Reproducible (same input → same output)
        - Documented (each step has a clear purpose)
        - Testable (we can verify it works)
        """
    )
    return


@app.cell
def __(pd):
    def clean_pokemon_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Pokemon card dataset.

        Steps:
        1. Remove duplicate card IDs (keep first occurrence)
        2. Fix inconsistent capitalization in type column
        3. Handle missing values appropriately
        4. Remove invalid records

        Args:
            df: Raw Pokemon DataFrame

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        # Step 1: Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['card_id'], keep='first')
        duplicates_removed = initial_count - len(df_clean)
        print(f"Step 1: Removed {duplicates_removed} duplicate records")

        # Step 2: Fix capitalization in type column
        df_clean['type'] = df_clean['type'].str.capitalize()
        print(f"Step 2: Standardized type capitalization")

        # Step 3: Handle missing values
        # For HP: Fill with median HP of the same type (reasonable imputation)
        for ptype in df_clean['type'].unique():
            type_mask = df_clean['type'] == ptype
            median_hp = df_clean.loc[type_mask, 'hp'].median()
            df_clean.loc[type_mask & df_clean['hp'].isna(), 'hp'] = median_hp

        print(f"Step 3a: Filled missing HP values with type-specific medians")

        # For price: Drop rows (can't reliably impute prices)
        price_missing = df_clean['price_usd'].isna().sum()
        df_clean = df_clean.dropna(subset=['price_usd'])
        print(f"Step 3b: Dropped {price_missing} rows with missing prices")

        # Step 4: Remove statistical outliers in price (>150, likely data entry errors)
        price_outliers = (df_clean['price_usd'] > 150).sum()
        df_clean = df_clean[df_clean['price_usd'] <= 150]
        print(f"Step 4: Removed {price_outliers} extreme price outliers (>$150)")

        # Convert HP to int
        df_clean['hp'] = df_clean['hp'].astype(int)

        print(f"\n✅ Cleaning complete: {len(df)} → {len(df_clean)} records")

        return df_clean

    df_clean = clean_pokemon_data(df_raw)
    return clean_pokemon_data, df_clean


@app.cell
def __(df_clean, mo):
    mo.md(
        f"""
        ### ✅ Cleaned Data Summary

        After cleaning, we have **{len(df_clean)} valid records**.

        Let's verify the data now passes validation:
        """
    )
    return


@app.cell
def __(df_clean, pokemon_schema):
    # Validate cleaned data
    try:
        validated_df = pokemon_schema.validate(df_clean)
        print("✅ Data is now valid!")
        print(f"✅ All {len(validated_df)} records passed validation")
    except pa.errors.SchemaErrors as e:
        print("❌ Still has validation errors:")
        print(e.failure_cases.head())
    return (validated_df,)


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Section 4: Data Versioning & Reproducibility

        **Key Question**: If you run this pipeline tomorrow, will you get the same result?

        In production, we need:
        - **Data versioning**: Track which version of data produced which model
        - **Pipeline versioning**: Track the code that processed the data
        - **Reproducibility**: Ability to recreate any past result

        ### Tools in Industry:
        - **DVC (Data Version Control)**: Git for data
        - **MLflow**: Track experiments with data versions
        - **Feast**: Feature store for versioned features
        - **Delta Lake**: Versioned data lake storage

        For this project, we'll use simple approaches that scale up.
        """
    )
    return


@app.cell
def __(Path, validated_df):
    # Save cleaned data with version information
    from datetime import datetime

    CLEAN_DATA_DIR = Path("data/clean")
    CLEAN_DATA_DIR.mkdir(exist_ok=True)

    # Version based on date
    version = datetime.now().strftime("%Y%m%d")
    output_path = CLEAN_DATA_DIR / f"pokemon_cards_clean_v{version}.csv"

    validated_df.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned data to: {output_path}")

    # Also save as "latest" for convenience
    latest_path = CLEAN_DATA_DIR / "pokemon_cards_clean_latest.csv"
    validated_df.to_csv(latest_path, index=False)
    print(f"✅ Saved as latest: {latest_path}")
    return CLEAN_DATA_DIR, datetime, latest_path, output_path, version


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Section 5: Performance Comparison - Pandas vs Polars

        You've heard about **Polars** - the fast new DataFrame library written in Rust.
        Let's compare it to pandas for our use case.

        ### When to use each:
        - **pandas**: <10GB data, lots of libraries expect pandas, exploratory work
        - **polars**: >10GB data, performance critical, parallel processing needed

        Let's run a performance comparison!
        """
    )
    return


@app.cell
def __():
    import polars as pl
    import time
    return pl, time


@app.cell
def __(DATASET_PATH, pd, pl, time, validated_df):
    # Pandas benchmark
    def pandas_pipeline():
        df = pd.read_csv(DATASET_PATH)
        df = df.drop_duplicates(subset=['card_id'])
        df['type'] = df['type'].str.capitalize()
        df = df.dropna(subset=['price_usd'])
        df = df[df['price_usd'] <= 150]
        return df

    # Polars benchmark
    def polars_pipeline():
        df = pl.read_csv(DATASET_PATH)
        df = df.unique(subset=['card_id'])
        df = df.with_columns(pl.col('type').str.to_titlecase())
        df = df.filter(pl.col('price_usd').is_not_null())
        df = df.filter(pl.col('price_usd') <= 150)
        return df

    # Benchmark pandas
    pandas_times = []
    for _ in range(5):
        start = time.time()
        df_pandas = pandas_pipeline()
        pandas_times.append(time.time() - start)

    pandas_avg = sum(pandas_times) / len(pandas_times)

    # Benchmark polars
    polars_times = []
    for _ in range(5):
        start = time.time()
        df_polars = polars_pipeline()
        polars_times.append(time.time() - start)

    polars_avg = sum(polars_times) / len(polars_times)

    speedup = pandas_avg / polars_avg

    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"Pandas average time: {pandas_avg*1000:.2f} ms")
    print(f"Polars average time: {polars_avg*1000:.2f} ms")
    print(f"\n🚀 Polars is {speedup:.2f}x faster!")
    print("\nNote: On larger datasets (>1GB), the difference is even more dramatic.")
    print(f"\nMemory usage:")
    print(f"  Pandas: {validated_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Polars: {df_polars.estimated_size() / 1024**2:.2f} MB")
    return (
        df_pandas,
        df_polars,
        pandas_avg,
        pandas_benchmark,
        pandas_pipeline,
        pandas_times,
        polars_avg,
        polars_benchmark,
        polars_pipeline,
        polars_times,
        speedup,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ### 🤔 Should You Switch to Polars?

        **Not necessarily!** Consider:

        **Stick with pandas if:**
        - Your data is <1GB
        - You're doing exploratory analysis
        - You need compatibility with scikit-learn, matplotlib, etc.
        - Your team knows pandas well

        **Use polars if:**
        - Data is >10GB
        - Performance is critical
        - You're building data pipelines
        - You can handle the learning curve

        **In this course**: We'll use pandas (industry standard, better ecosystem),
        but you now know polars exists for when you need it!

        ---
        ## Section 6: Key Takeaways & Socratic Questions

        ### ✅ What You Learned

        1. **Data quality matters more than algorithms**
        2. **Always validate your data** (use schemas like Pandera)
        3. **Build reproducible pipelines** (version everything)
        4. **Performance tools exist** (polars) but choose wisely
        5. **Document and test your data code** (it's real code!)

        ### 🤔 Socratic Questions (Test Your Understanding)

        Try to answer these before moving on:

        1. **"If I run your data pipeline twice on the same input, will I get identical output? Why or why not?"**
           - Think about: random operations, timestamps, external dependencies

        2. **"Your model performed great in training but fails in production. The data looks similar. What could be wrong?"**
           - Think about: data drift, schema changes, missing values handled differently

        3. **"Someone changed a column name upstream. How would you detect this before it breaks your model?"**
           - Think about: schema validation, automated tests, monitoring

        4. **"When would you choose polars over pandas? When would you stick with pandas?"**
           - Think about: data size, team skills, ecosystem compatibility

        ### 🏋️ Practice Exercise

        Before moving to Module 2, try this:

        **Exercise**: You receive a new Pokemon dataset with a different schema (missing columns, extra columns).
        Modify the cleaning pipeline to handle this gracefully.

        Hints:
        - Update the Pandera schema
        - Add checks for required columns
        - Handle missing features with defaults

        ---
        ## 🏢 Industry Context: Real-World Data Engineering

        ### How Companies Do This at Scale

        **Small Startup (<10 models)**:
        - Simple CSV files in S3/GCS
        - Python scripts for cleaning
        - Basic validation

        **Medium Company (10-50 models)**:
        - Data warehouse (Snowflake, BigQuery)
        - Airflow for orchestration
        - Great Expectations for validation
        - Basic feature store (Feast)

        **Large Company (>50 models)**:
        - Custom data platform
        - Real-time + batch pipelines
        - Feature stores (Tecton, internal)
        - Data quality monitoring
        - Automated alerting

        ### Common Pitfalls to Avoid

        ⚠️ **Don't**: Load data without validation
        ✅ **Do**: Define schemas and validate automatically

        ⚠️ **Don't**: Clean data in notebooks without saving the code
        ✅ **Do**: Write reusable cleaning functions with tests

        ⚠️ **Don't**: Ignore missing values
        ✅ **Do**: Understand why they're missing and handle appropriately

        ⚠️ **Don't**: Assume your data is correct
        ✅ **Do**: Be paranoid - validate everything

        ---
        ## 🎯 Module 1 Checkpoint

        You've completed Module 1 when you can:

        - [ ] Write a data validation test from scratch in <10 minutes
        - [ ] Explain why data versioning matters to a junior engineer
        - [ ] Debug a data quality issue by reading test failures
        - [ ] Choose the right tool (pandas/polars/SQL) for a given task

        **Next**: Module 2 - EDA & Feature Engineering

        In the next module, you'll learn how to:
        - Explore data to extract insights
        - Engineer features that improve model performance
        - Avoid data leakage (the silent killer)
        - Build scikit-learn preprocessing pipelines
        """
    )
    return


if __name__ == "__main__":
    app.run()
