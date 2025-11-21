"""
Module 1 Exercises: Data Engineering Foundations
================================================

These exercises reinforce the concepts from Module 1.
Complete each exercise and check your solutions.

Time estimate: 1-2 hours
"""

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Module 1 Exercises

    Complete these exercises to solidify your understanding of data engineering fundamentals.

    ## Exercise 1.1: Break the Data (20 min)

    **Goal**: Introduce 5 different data quality issues, then write tests to catch them.

    **Instructions**:
    1. Load the cleaned Pokemon dataset
    2. Introduce these issues deliberately:
       - Add 10 duplicate card_ids
       - Add 20 missing values in random columns
       - Add 5 invalid values (e.g., negative HP)
       - Add 3 outliers (price > 500)
       - Add inconsistent formatting in a categorical column
    3. Write Pandera schema tests to catch ALL issues
    4. Verify your tests work by running validation

    **Learning Objective**: Understanding what can go wrong helps you prevent it.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check

    # Load the clean data
    DATA_PATH = Path("data/clean/pokemon_cards_clean_latest.csv")
    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df)} clean records")
    return DataFrameSchema, Path, df, np, pa, pd


@app.cell
def _(df):
    # TODO: Create a copy and introduce data quality issues
    df_broken = df.copy()

    # TODO 1: Add 10 duplicate card_ids
    # Hint: Sample 10 random indices and duplicate those rows


    # TODO 2: Add 20 missing values in random columns (stats only)
    # Hint: Use df.loc[random_indices, 'column_name'] = np.nan


    # TODO 3: Add 5 invalid values (negative HP)
    # Hint: Sample 5 random indices and set hp to negative values


    # TODO 4: Add 3 price outliers (>500)
    # Hint: Sample 3 indices and multiply price by 100


    # TODO 5: Add inconsistent formatting in type column
    # Hint: Make some type values lowercase or UPPERCASE


    print(f"Broken dataset created: {len(df_broken)} records")
    return (df_broken,)


@app.cell
def _(DataFrameSchema):
    # TODO: Define a Pandera schema that will catch ALL the issues you introduced

    validation_schema = DataFrameSchema(
        columns={
            # TODO: Add column definitions with appropriate checks
            # Example:
            # "card_id": Column(str, unique=True, nullable=False),
            # Add the rest...
        },
        strict=False,
        coerce=True
    )

    print("Schema defined")
    return (validation_schema,)


@app.cell
def _(df_broken, pa, validation_schema):
    # TODO: Run validation and check that it catches all issues
    try:
        validation_schema.validate(df_broken, lazy=True)
        print("❌ Validation passed - your schema didn't catch the issues!")
    except pa.errors.SchemaErrors as e:
        print(f"✅ Validation failed as expected!")
        print(f"Caught {len(e.failure_cases)} errors")
        print("\nError summary:")
        print(e.failure_cases.groupby('check').size())
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🤔 Reflection Questions:

    1. Which type of data quality issue was hardest to detect? Why?
    2. How would you automate this validation in a production pipeline?
    3. What happens if the schema is too strict? Too lenient?

    ---
    ## Exercise 1.2: Refactor Messy Code (20 min)

    **Goal**: Take messy data loading code and make it production-ready.

    **Instructions**:
    1. Review the messy code below
    2. Refactor it to be:
       - Reproducible (no randomness, or controlled randomness)
       - Documented (type hints, docstrings)
       - Tested (add basic error handling)
       - Modular (break into functions)

    **Learning Objective**: Code quality matters for data code too!
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Messy Code (DON'T USE THIS!):

    ```python
    # Bad example - do NOT copy this!
    import pandas as pd

    data = pd.read_csv('data/pokemon_cards.csv')
    data = data.drop_duplicates()
    data = data.dropna()
    data['type'] = data['type'].str.lower()
    data = data[data.price_usd < 100]
    data.to_csv('output.csv')
    print('done')
    ```

    ### Problems with this code:
    1. No error handling (what if file doesn't exist?)
    2. No documentation (what does this code do?)
    3. Hardcoded paths
    4. No type hints
    5. Unclear transformation logic
    6. No validation
    7. Destructive operations (modifies data in place)
    8. No logging of what changed
    """)
    return


@app.cell
def _(Path, pd):
    # TODO: Refactor the messy code above using best practices

    def load_and_clean_data(
        input_path: Path,
        output_path: Path,
        max_price: float = 100.0
    ) -> pd.DataFrame:
        """
        Load and clean Pokemon card data.

        TODO: Complete this function with proper:
        - Error handling
        - Validation
        - Logging
        - Documentation

        Args:
            input_path: Path to input CSV file
            output_path: Path to save cleaned CSV file
            max_price: Maximum valid price threshold

        Returns:
            Cleaned DataFrame

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If data validation fails
        """
        # TODO: Implement the function
        pass

    # TODO: Test your function
    # input_path = Path("data/pokemon_cards.csv")
    # output_path = Path("data/clean/refactored_clean.csv")
    # df_refactored = load_and_clean_data(input_path, output_path)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🤔 Reflection Questions:

    1. How is your refactored version better than the messy version?
    2. What would you add if this went to production?
    3. How would you test this function automatically?

    ---
    ## Exercise 1.3: Performance Challenge (30 min)

    **Goal**: Optimize a slow pandas operation using polars.

    **Scenario**: You have a large dataset (we'll simulate 100K rows) and need to:
    1. Group by Pokemon type
    2. Calculate average stats per type
    3. Filter to types with avg HP > 70
    4. Sort by average attack
    5. Export to CSV

    **Instructions**:
    1. Implement this in pandas
    2. Benchmark the execution time
    3. Implement the same in polars
    4. Compare performance
    5. Determine when the switch to polars is worth it

    **Learning Objective**: Understand performance tradeoffs.
    """)
    return


@app.cell
def _(df, np, pd):
    # Generate a larger dataset for benchmarking
    # Repeat the dataset 125x to get ~100K rows
    df_large = pd.concat([df] * 125, ignore_index=True)

    # Add some randomness to make it realistic
    np.random.seed(42)
    df_large['hp'] = df_large['hp'] + np.random.randint(-5, 5, size=len(df_large))
    df_large['attack'] = df_large['attack'] + np.random.randint(-5, 5, size=len(df_large))

    print(f"Generated dataset with {len(df_large)} rows")
    print(f"Memory usage: {df_large.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return


@app.cell
def _():
    import time

    # TODO: Implement the transformation in pandas
    def pandas_aggregation(df):
        """
        Aggregate Pokemon stats by type.

        TODO: Implement this function to:
        1. Group by type
        2. Calculate mean for hp, attack, defense, sp_attack, sp_defense, speed
        3. Filter to types where avg hp > 70
        4. Sort by avg attack descending

        Returns:
            Aggregated DataFrame
        """
        # TODO: Your code here
        pass

    # TODO: Benchmark pandas
    # start = time.time()
    # result_pandas = pandas_aggregation(df_large)
    # pandas_time = time.time() - start
    # print(f"Pandas time: {pandas_time:.4f} seconds")
    return


@app.cell
def _(pl):
    # TODO: Implement the same transformation in polars
    def polars_aggregation(df):
        """
        Aggregate Pokemon stats by type using Polars.

        TODO: Implement the same logic as pandas version.

        Returns:
            Polars DataFrame with aggregated results
        """
        # Convert to polars if needed
        if not isinstance(df, pl.DataFrame):
            df_pl = pl.from_pandas(df)
        else:
            df_pl = df

        # TODO: Your polars code here
        pass

    # TODO: Benchmark polars
    # start = time.time()
    # result_polars = polars_aggregation(df_large)
    # polars_time = time.time() - start
    # print(f"Polars time: {polars_time:.4f} seconds")
    # print(f"Speedup: {pandas_time / polars_time:.2f}x")
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🤔 Reflection Questions:

    1. At what data size does polars become worth the complexity?
    2. What if your data is 10MB? 100MB? 10GB?
    3. What other factors matter besides performance (team knowledge, ecosystem, etc.)?

    ---
    ## 🎯 Module 1 Checkpoint: "New Pokemon Data Drop"

    **The Challenge**: You receive a new Pokemon card dataset from a different source.
    The schema is slightly different:
    - Column names use `snake_case` instead of matching your schema
    - Has extra columns you don't need
    - Missing some columns you expect
    - Different value encodings (e.g., "TRUE"/"FALSE" instead of True/False)

    **Your Task** (30-45 min):
    Create a robust pipeline that:
    1. Detects schema mismatches
    2. Maps columns to expected names
    3. Validates data quality
    4. Produces a dataset matching your standard schema
    5. Logs all transformations

    **Deliverable**:
    - A function `harmonize_external_data()` that handles this
    - Documentation of what transformations were applied
    - Test that it works on different input schemas

    This simulates a real production scenario where data sources change!
    """)
    return


@app.cell
def _(Path, pd):
    # First, let's create a simulated "external" dataset
    def create_external_dataset():
        """Create a dataset with different schema."""
        df_external = pd.DataFrame({
            'pokemon_id': ['EXT-001', 'EXT-002', 'EXT-003'],
            'pokemon_name': ['Charizard', 'Blastoise', 'Venusaur'],
            'primary_type': ['fire', 'water', 'grass'],
            'hit_points': [180, 170, 175],
            'atk': [130, 110, 105],
            'def': [110, 120, 115],
            'special_attack': [140, 130, 125],
            'special_defense': [115, 125, 120],
            'spd': [120, 100, 105],
            'gen': [1, 1, 1],
            'legendary': ['FALSE', 'FALSE', 'FALSE'],
            'rarity_level': ['Ultra Rare', 'Ultra Rare', 'Ultra Rare'],
            'market_price': [250.0, 200.0, 180.0],
            'extra_column_we_dont_need': ['ignore', 'this', 'data']
        })

        output_path = Path("data/pokemon_external.csv")
        df_external.to_csv(output_path, index=False)
        print(f"Created external dataset at {output_path}")
        return df_external

    df_external = create_external_dataset()
    return


@app.cell
def _(pd):
    # TODO: Create a harmonization function
    def harmonize_external_data(external_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform external Pokemon data to match our standard schema.

        Expected input columns (flexible):
        - pokemon_id, card_id, id → card_id
        - pokemon_name, name → name
        - primary_type, type → type
        - hit_points, hp → hp
        - atk, attack → attack
        - def, defense → defense
        - special_attack, sp_attack → sp_attack
        - special_defense, sp_defense → sp_defense
        - spd, speed → speed
        - gen, generation → generation
        - legendary, is_legendary → is_legendary (convert TRUE/FALSE to bool)
        - rarity_level, rarity → rarity
        - market_price, price_usd, price → price_usd

        Args:
            external_df: DataFrame from external source

        Returns:
            DataFrame matching our standard schema

        Raises:
            ValueError: If required columns are missing
        """
        # TODO: Implement column mapping
        # TODO: Handle data type conversions
        # TODO: Validate the result
        # TODO: Log all transformations
        pass

    # TODO: Test your function
    # df_harmonized = harmonize_external_data(df_external)
    # print(df_harmonized.head())
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 🎓 Exercise Solutions

    Don't peek until you've tried! Solutions are in `solutions_01.py`.

    ## 📝 Self-Assessment

    Before moving to Module 2, rate yourself:

    - [ ] I can write a data validation schema from scratch
    - [ ] I understand when to use pandas vs polars
    - [ ] I can debug data quality issues systematically
    - [ ] I can write production-quality data pipeline code
    - [ ] I can handle schema mismatches between data sources

    **If you checked all boxes**: You're ready for Module 2!

    **If not**: Review the sections you struggled with and try the exercises again.

    ---
    ## 💡 Additional Challenges (Optional)

    If you want more practice:

    1. **Challenge 1**: Add unit tests for the cleaning pipeline using pytest
    2. **Challenge 2**: Implement data drift detection between two versions
    3. **Challenge 3**: Create a monitoring dashboard for data quality metrics
    4. **Challenge 4**: Handle streaming data (one record at a time validation)

    These challenges will prepare you for real production scenarios!
    """)
    return


if __name__ == "__main__":
    app.run()
