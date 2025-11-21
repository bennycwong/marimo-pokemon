"""
Module 8: Capstone Project - Pokemon Card Price Prediction

An end-to-end ML engineering project applying all concepts from modules 0-7.

**Project Goal**: Build a production-ready Pokemon card price prediction model

**What You'll Do**:
1. Frame the business problem and success metrics
2. Engineer a clean, reproducible data pipeline
3. Perform systematic feature engineering
4. Train and evaluate multiple models
5. Document your work for team review
6. Design a monitoring strategy

**Success Criteria**: See CAPSTONE_RUBRIC.md
"""

import marimo

__generated_with = "0.9.30"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md("""
    # Capstone Project: Pokemon Card Price Prediction

    ## 🎯 Project Overview

    **Scenario**: You've been hired as the first ML engineer at PokéMarket, a Pokemon card
    trading platform. The CEO wants to add automatic price recommendations to help sellers
    price their cards competitively.

    **Your Mission**: Build a production-ready price prediction system.

    ---

    ## Module Guide

    This capstone integrates everything you've learned:

    | Module | Skills Applied |
    |--------|----------------|
    | 0: Business Context | Define success metrics, calculate ROI, set expectations |
    | 1: Data Engineering | Build reproducible pipelines, validate data quality |
    | 2: Feature Engineering | Create predictive features, prevent leakage |
    | 3: Model Training | Compare models systematically, tune hyperparameters |
    | 4: Evaluation | Choose right metrics, analyze errors, document results |
    | 5: Deployment | Design API, validate inputs, handle edge cases |
    | 6: Monitoring | Detect drift, plan incident response |
    | 7: Collaboration | Write clear docs, prepare for code review |

    ---

    ## 📋 Your Deliverables

    1. **Business Analysis** (Module 0)
       - Problem statement
       - Success metrics
       - ROI calculation

    2. **Data Pipeline** (Module 1)
       - Data validation with Pandera
       - Reproducible preprocessing
       - Quality checks

    3. **Feature Engineering** (Module 2)
       - Systematic feature creation
       - Leakage prevention
       - Feature documentation

    4. **Model Development** (Module 3)
       - Multiple model comparison
       - Hyperparameter tuning
       - Cross-validation

    5. **Evaluation Report** (Module 4)
       - Metric selection and results
       - Error analysis
       - Model card

    6. **Deployment Plan** (Module 5)
       - API design with FastAPI
       - Input validation
       - Performance requirements

    7. **Monitoring Strategy** (Module 6)
       - Key metrics to track
       - Alert thresholds
       - Incident response plan

    8. **Documentation** (Module 7)
       - README with setup instructions
       - PR description
       - Code review checklist

    ---

    ## 🚀 Getting Started

    Work through each phase below. Don't skip ahead - each builds on previous work!
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Phase 1: Business Context (Module 0)

    Before writing any code, understand the business problem.

    ### Your Tasks:

    #### 1.1 Problem Statement

    Answer these questions:
    - **Who** are the users? (Sellers? Buyers? Both?)
    - **What** decision does your model help them make?
    - **When** will the model be used? (Listing time? Real-time browsing?)
    - **Why** is ML better than a simple average price?
    - **How** good does it need to be? (What error is acceptable?)

    #### 1.2 Success Metrics

    Define metrics from 3 perspectives:

    **Model Metrics** (ML performance):
    - Primary: _______________ (e.g., RMSE, MAE, R²)
    - Why this metric? _______________
    - Target value: _______________

    **Business Metrics** (impact on company):
    - Primary: _______________ (e.g., % increase in listings, revenue impact)
    - How to measure: _______________
    - Target: _______________

    **User Metrics** (satisfaction):
    - Primary: _______________ (e.g., % of users who accept suggestion)
    - How to measure: _______________
    - Target: _______________

    #### 1.3 ROI Calculation

    Estimate the value:

    ```
    Current State:
    - How are prices determined now? _______________
    - What problems does this cause? _______________
    - Cost of current approach: $_______________/year

    ML Solution:
    - Development cost: $_______________ (your time @ $100k salary = ~$50/hour)
    - Infrastructure cost: $_______________/month (AWS, monitoring)
    - Expected value: $_______________ (more listings? Higher conversion?)

    ROI Year 1: (Value - Costs) / Costs = _______________
    ```

    #### 1.4 Stakeholder Communication

    Draft a 1-paragraph pitch to your CEO explaining:
    - What you'll build
    - Why it matters
    - What success looks like
    - Realistic timeline

    ---

    **💡 Deliverable**: Write your analysis in a markdown file: `capstone_business_analysis.md`
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Phase 2: Data Engineering (Module 1)

    Build a reliable data foundation.

    ### Your Tasks:

    #### 2.1 Load and Explore Data

    ```python
    import pandas as pd
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check

    # Load dataset
    df = pd.read_csv('data/pokemon_cards.csv')

    # Explore
    df.info()
    df.describe()
    df['price_usd'].hist()
    ```

    #### 2.2 Define Data Schema

    Create a Pandera schema to validate:
    - Column types
    - Value ranges (e.g., hp between 1-300)
    - Required fields
    - Unique constraints (e.g., card_id)

    ```python
    schema = DataFrameSchema({
        'card_id': Column(str, unique=True),
        'name': Column(str),
        'type': Column(str, Check.isin(['Fire', 'Water', ...])),
        'hp': Column(int, Check.in_range(1, 300)),
        # TODO: Add more columns
        'price_usd': Column(float, Check.greater_than(0)),
    })

    # Validate
    df = schema.validate(df)
    ```

    #### 2.3 Data Quality Checks

    Check for:
    - Missing values
    - Duplicates
    - Outliers in price (box plots, z-scores)
    - Unrealistic combinations (e.g., Common card with price > $100)

    #### 2.4 Create Reproducible Pipeline

    ```python
    def load_and_validate_data(path: str) -> pd.DataFrame:
        '''Load Pokemon card data with quality checks.'''
        df = pd.read_csv(path)
        df = schema.validate(df)
        # Add your quality checks
        return df

    # Test it
    df = load_and_validate_data('data/pokemon_cards.csv')
    ```

    #### 2.5 Train/Val/Test Split

    ```python
    from sklearn.model_selection import train_test_split

    # Hold out test set (20%)
    train_val, test = train_test_split(df, test_size=0.2, random_state=42)

    # Split train/val (80/20 of remaining)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    ```

    **⚠️ Critical**: Don't touch the test set until final evaluation!

    ---

    **💡 Deliverable**: Python file with `load_and_validate_data()` function and documented data quality checks
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Phase 3: Feature Engineering (Module 2)

    Create predictive features without leakage.

    ### Your Tasks:

    #### 3.1 Brainstorm Features

    List potential features:

    **Direct Features** (already in data):
    - [ ] HP, attack, defense, sp_attack, sp_defense, speed
    - [ ] Type (one-hot encoded)
    - [ ] Rarity (ordinal encoded)
    - [ ] Generation
    - [ ] is_legendary

    **Engineered Features** (derived):
    - [ ] total_stats = sum of all battle stats
    - [ ] attack_defense_ratio = attack / (defense + 1)
    - [ ] power_level = weighted sum of stats
    - [ ] _______________
    - [ ] _______________

    **Aggregated Features** (⚠️ check for leakage!):
    - [ ] count_by_type = count of this type in training data
    - [ ] avg_stats_by_rarity = average stats for this rarity
    - [ ] _______________

    #### 3.2 Check for Data Leakage

    For each feature, ask:
    1. **Will this feature be available at prediction time?**
    2. **Does it use information from the future?**
    3. **Does it use the target variable?**

    ❌ **Bad Feature**: `avg_price_by_type` (uses target!)
    ✅ **Good Feature**: `count_by_type` (only uses training data)

    #### 3.3 Implement Feature Pipeline

    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    # Define transformations
    numeric_features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    categorical_features = ['type', 'rarity']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Add engineered features
    def engineer_features(df):
        '''Add derived features.'''
        df = df.copy()
        df['total_stats'] = (df['hp'] + df['attack'] + df['defense'] +
                            df['sp_attack'] + df['sp_defense'] + df['speed'])
        # TODO: Add more features
        return df

    # Apply to train set only
    train_features = engineer_features(train)
    ```

    #### 3.4 Feature Documentation

    Document each feature:

    | Feature | Type | Description | Leakage Check | Rationale |
    |---------|------|-------------|---------------|-----------|
    | hp | Numeric | Hit points | ✅ Available at prediction time | Higher HP cards may be more valuable |
    | total_stats | Engineered | Sum of all stats | ✅ Derived from available data | Overall power affects price |
    | ... | ... | ... | ... | ... |

    ---

    **💡 Deliverable**: `engineer_features()` function and feature documentation table
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Phase 4: Model Training (Module 3)

    Systematically compare models and tune hyperparameters.

    ### Your Tasks:

    #### 4.1 Baseline Model

    Start simple:

    ```python
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score

    # Baseline: Predict mean price
    baseline_pred = train['price_usd'].mean()
    baseline_mae = mean_absolute_error(val['price_usd'],
                                      [baseline_pred] * len(val))

    print(f"Baseline MAE: ${baseline_mae:.2f}")

    # Simple linear regression
    lr = LinearRegression()
    # TODO: Fit on train, predict on val
    ```

    #### 4.2 Model Comparison

    Try multiple algorithms:

    ```python
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }

    results = {}
    for name, model in models.items():
        # Use cross-validation on train set
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring='neg_mean_absolute_error')
        results[name] = -cv_scores.mean()
        print(f"{name}: MAE = ${results[name]:.2f}")
    ```

    #### 4.3 Hyperparameter Tuning

    Tune the best model:

    ```python
    from sklearn.model_selection import RandomizedSearchCV

    param_dist = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='neg_mean_absolute_error',
        random_state=42
    )

    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    ```

    #### 4.4 Track Experiments

    Use MLflow to track all experiments:

    ```python
    import mlflow

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric('val_mae', val_mae)
        mlflow.log_metric('val_r2', val_r2)
        mlflow.sklearn.log_model(model, 'model')
    ```

    ---

    **💡 Deliverable**: Experiment tracking table comparing all models with MAE, R², training time
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Phase 5: Model Evaluation (Module 4)

    Rigorously evaluate your best model.

    ### Your Tasks:

    #### 5.1 Choose Metrics

    Explain why you chose each metric:

    - **MAE** (Mean Absolute Error): _______________
    - **RMSE**: _______________
    - **R²**: _______________
    - **MAPE** (Mean Absolute Percentage Error): _______________

    Which is most important for your business case?

    #### 5.2 Error Analysis

    Identify failure modes:

    ```python
    # Get predictions on validation set
    val_pred = best_model.predict(X_val)
    val['predicted_price'] = val_pred
    val['error'] = val['price_usd'] - val['predicted_price']
    val['abs_error'] = val['error'].abs()

    # Analyze worst predictions
    worst_10 = val.nlargest(10, 'abs_error')

    print("Worst predictions:")
    print(worst_10[['name', 'type', 'rarity', 'price_usd', 'predicted_price', 'error']])

    # Error by category
    val.groupby('rarity')['abs_error'].mean()
    val.groupby('type')['abs_error'].mean()

    # Visualize
    import matplotlib.pyplot as plt
    plt.scatter(val['price_usd'], val['predicted_price'])
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Prediction vs Actual')
    ```

    #### 5.3 Residual Analysis

    ```python
    # Check for patterns in residuals
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.hist(val['error'], bins=50)
    plt.title('Error Distribution')

    plt.subplot(132)
    plt.scatter(val['predicted_price'], val['error'])
    plt.xlabel('Predicted Price')
    plt.ylabel('Error')
    plt.title('Residual Plot')

    plt.subplot(133)
    val.boxplot(column='abs_error', by='rarity')
    plt.title('Error by Rarity')
    ```

    **Questions to answer**:
    - Are errors normally distributed?
    - Do errors increase with price (heteroscedasticity)?
    - Which categories have highest errors?
    - Why?

    #### 5.4 Create Model Card

    Document your model:

    ```markdown
    # Pokemon Card Price Prediction Model Card

    ## Model Details
    - **Name**: Pokemon Price Predictor v1.0
    - **Type**: Random Forest Regressor
    - **Date**: 2024-01-15
    - **Developer**: Your Name

    ## Intended Use
    - **Primary Use**: Price recommendations for sellers on PokéMarket
    - **Out-of-Scope**: Not for insurance valuation or legal disputes

    ## Training Data
    - **Source**: Pokemon card database
    - **Size**: 800 cards (640 train, 160 val)
    - **Date Range**: All generations

    ## Performance
    | Metric | Value |
    |--------|-------|
    | MAE | $X.XX |
    | RMSE | $X.XX |
    | R² | 0.XX |

    ## Limitations
    - Higher error on rare cards (< 5% of dataset)
    - Does not account for card condition
    - Price data may be outdated

    ## Ethical Considerations
    - May undervalue rare cards, disadvantaging sellers
    - Should not be sole factor in pricing decisions
    ```

    ---

    **💡 Deliverable**: Model card + error analysis report with visualizations
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Phase 6: Deployment Design (Module 5)

    Plan how to serve your model in production.

    ### Your Tasks:

    #### 6.1 Design API

    ```python
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field, validator
    import joblib

    # Load model
    model = joblib.load('model.pkl')

    # Define input schema
    class PokemonCard(BaseModel):
        name: str
        type: str
        hp: int = Field(..., ge=1, le=300)
        attack: int = Field(..., ge=1, le=300)
        defense: int = Field(..., ge=1, le=300)
        sp_attack: int = Field(..., ge=1, le=300)
        sp_defense: int = Field(..., ge=1, le=300)
        speed: int = Field(..., ge=1, le=300)
        generation: int = Field(..., ge=1, le=9)
        is_legendary: bool
        rarity: str

        @validator('type')
        def validate_type(cls, v):
            valid_types = ['Fire', 'Water', 'Grass', ...]
            if v not in valid_types:
                raise ValueError(f'Type must be one of {valid_types}')
            return v

        @validator('rarity')
        def validate_rarity(cls, v):
            valid = ['Common', 'Uncommon', 'Rare', 'Ultra Rare', 'Secret Rare']
            if v not in valid:
                raise ValueError(f'Rarity must be one of {valid}')
            return v

    app = FastAPI()

    @app.post('/predict')
    def predict_price(card: PokemonCard):
        '''Predict Pokemon card price.'''
        try:
            # Engineer features
            features = engineer_features(card.dict())
            # Predict
            prediction = model.predict([features])[0]
            return {
                'predicted_price_usd': round(prediction, 2),
                'confidence_interval': [round(prediction * 0.8, 2),
                                       round(prediction * 1.2, 2)]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    ```

    #### 6.2 Handle Edge Cases

    What happens when:
    - User submits a new Pokemon type not in training data?
    - All stats are at maximum (300)?
    - Card is both Common AND Legendary (impossible)?

    ```python
    # Test edge cases
    test_cases = [
        {'name': 'NewMon', 'type': 'Quantum', ...},  # Unknown type
        {'name': 'MaxMon', 'hp': 300, 'attack': 300, ...},  # Max stats
        {'name': 'WeirdMon', 'is_legendary': True, 'rarity': 'Common', ...}  # Impossible combo
    ]

    for card in test_cases:
        try:
            result = predict_price(PokemonCard(**card))
            print(f"✅ {card['name']}: ${result['predicted_price_usd']}")
        except Exception as e:
            print(f"❌ {card['name']}: {e}")
    ```

    #### 6.3 Performance Requirements

    Define SLOs (Service Level Objectives):

    | Metric | Target | Rationale |
    |--------|--------|-----------|
    | Latency (p95) | < 100ms | Users wait while listing |
    | Availability | 99.9% | 8 hours downtime/year acceptable |
    | Throughput | 100 req/sec | Expected peak traffic |

    #### 6.4 A/B Test Plan

    How will you validate the model helps users?

    ```
    Control Group (50% of users):
    - No price suggestion
    - Track: time to list, listing price, sale rate

    Treatment Group (50% of users):
    - Show ML price suggestion
    - Track: % who accept suggestion, listing price, sale rate

    Success Criteria:
    - 30%+ of users accept suggestion
    - Sale rate improves by 5%+
    - No decrease in average listing price

    Duration: 2 weeks
    Sample Size: 1000 listings per group
    ```

    ---

    **💡 Deliverable**: FastAPI code + edge case test results + A/B test plan
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Phase 7: Monitoring Strategy (Module 6)

    Plan how to detect issues in production.

    ### Your Tasks:

    #### 7.1 Key Metrics to Monitor

    **Model Performance**:
    - [ ] Prediction latency (p50, p95, p99)
    - [ ] Error rate (5xx responses)
    - [ ] MAE on recent predictions (once we get actual prices)

    **Data Quality**:
    - [ ] % of requests with missing fields
    - [ ] % of requests with out-of-range values
    - [ ] Distribution of input features (detect drift)

    **Business Metrics**:
    - [ ] % of users who see suggestion
    - [ ] % of users who accept suggestion
    - [ ] Average suggested price vs actual listing price

    #### 7.2 Design Data Drift Detection

    ```python
    import pandas as pd
    from scipy import stats

    def detect_drift(production_data, training_data, feature):
        '''Detect distribution shift using KS test.'''
        statistic, p_value = stats.ks_2samp(
            training_data[feature],
            production_data[feature]
        )

        if p_value < 0.05:
            print(f"⚠️ DRIFT DETECTED in {feature}")
            print(f"  KS statistic: {statistic:.4f}, p-value: {p_value:.4f}")
            return True
        return False

    # Run weekly
    for feature in numeric_features:
        detect_drift(production_df, train_df, feature)
    ```

    #### 7.3 Set Alert Thresholds

    | Metric | Warning | Critical | Action |
    |--------|---------|----------|--------|
    | Latency p95 | > 100ms | > 500ms | Scale up instances |
    | Error rate | > 1% | > 5% | Page on-call engineer |
    | MAE increase | > 20% | > 50% | Investigate data drift |
    | Drift detected | Any feature | 3+ features | Retrain model |

    #### 7.4 Create Incident Response Runbook

    **Scenario 1: MAE Suddenly Increases by 30%**

    ```markdown
    1. Check for data drift:
       - Run drift detection on last 7 days of data
       - Compare feature distributions to training data

    2. Inspect recent predictions:
       - Sample 100 recent predictions
       - Look for patterns in errors (type? rarity?)

    3. Check for data quality issues:
       - Any new/missing types?
       - Out-of-range values?

    4. Investigate external factors:
       - New card release?
       - Market event affecting prices?

    5. Mitigation:
       - If drift: Retrain with recent data
       - If data quality: Fix validation
       - If external: Update feature engineering
    ```

    **Scenario 2: Prediction Latency Spikes to 2 seconds**

    [Your turn - write the runbook]

    #### 7.5 Retraining Schedule

    Plan when to retrain:

    ```
    Trigger-Based Retraining:
    - MAE increases by 20%
    - Drift detected in 3+ features

    Time-Based Retraining:
    - Weekly: Retrain with last 7 days of data
    - Monthly: Full retrain with all data

    Process:
    1. Pull new data
    2. Run data quality checks
    3. Retrain model
    4. Validate on holdout set (MAE must be within 10% of current)
    5. A/B test new model (10% traffic for 24 hours)
    6. Full rollout if metrics improve
    ```

    ---

    **💡 Deliverable**: Monitoring dashboard mockup + alert rules + 2 incident runbooks
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Phase 8: Documentation & Collaboration (Module 7)

    Prepare your work for team review and handoff.

    ### Your Tasks:

    #### 8.1 Write README

    ```markdown
    # Pokemon Card Price Prediction

    ML system to provide price recommendations for Pokemon card sellers.

    ## Quick Start

    \\```bash
    # Install dependencies
    pip install -r requirements.txt

    # Train model
    python train_model.py

    # Start API server
    uvicorn api:app --reload

    # Test prediction
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"name": "Pikachu", "type": "Electric", ...}'
    \\```

    ## Project Structure

    \\```
    pokemon-price-prediction/
    ├── data/
    │   ├── pokemon_cards.csv
    │   └── generate_dataset.py
    ├── notebooks/
    │   ├── 01_eda.py
    │   ├── 02_feature_engineering.py
    │   └── 03_model_training.py
    ├── src/
    │   ├── data_validation.py
    │   ├── features.py
    │   └── model.py
    ├── api.py
    ├── train_model.py
    ├── requirements.txt
    └── README.md
    \\```

    ## Model Performance

    | Metric | Validation | Test |
    |--------|------------|------|
    | MAE | $X.XX | $X.XX |
    | RMSE | $X.XX | $X.XX |
    | R² | 0.XX | 0.XX |

    ## API Usage

    [Document your endpoints]

    ## Monitoring

    [Link to dashboard]

    ## Contributing

    See CONTRIBUTING.md for code review guidelines.
    ```

    #### 8.2 Write Pull Request Description

    Imagine you're submitting this for review:

    ```markdown
    ## PR: Add Pokemon Card Price Prediction Model

    ### Summary

    Implements end-to-end price prediction system for PokéMarket.

    ### Changes

    - **Data Pipeline** (`src/data_validation.py`): Pandera schemas + quality checks
    - **Feature Engineering** (`src/features.py`): 8 engineered features
    - **Model Training** (`train_model.py`): Random Forest with hyperparameter tuning
    - **API** (`api.py`): FastAPI endpoint with Pydantic validation
    - **Monitoring** (`src/monitoring.py`): Drift detection + alert rules
    - **Tests** (`tests/`): Unit tests for all modules

    ### Performance

    | Model | MAE | RMSE | R² |
    |-------|-----|------|-----|
    | Baseline (mean) | $15.43 | $25.12 | 0.00 |
    | **Random Forest** | **$2.34** | **$4.12** | **0.89** |

    ### Error Analysis

    - Average error: $2.34 (12% of median price)
    - Highest errors on Secret Rare cards (< 2% of data)
    - No systematic bias by type or generation

    ### Testing

    - [x] All unit tests pass
    - [x] Manual testing of edge cases
    - [x] Load testing: 500 req/s at p95 latency 45ms
    - [x] Validated no data leakage

    ### Deployment Plan

    1. Deploy to staging for QA testing
    2. A/B test with 10% traffic for 1 week
    3. Full rollout if metrics improve

    ### Risks & Mitigations

    **Risk**: Model may undervalue rare cards
    **Mitigation**: Show confidence interval to users

    **Risk**: Price data may be stale
    **Mitigation**: Weekly retraining + drift detection

    ### Questions for Reviewers

    1. Is the feature set comprehensive enough?
    2. Should we add card condition as a feature?
    3. What should we do about new Pokemon types?

    ### Checklist

    - [x] Code follows style guide
    - [x] Added tests
    - [x] Updated documentation
    - [x] No data leakage
    - [x] Model card created
    - [x] Monitoring implemented
    ```

    #### 8.3 Create requirements.txt

    ```
    pandas==2.1.0
    scikit-learn==1.3.0
    xgboost==2.0.0
    pandera==0.17.0
    fastapi==0.103.0
    pydantic==2.3.0
    mlflow==2.7.0
    matplotlib==3.7.0
    seaborn==0.12.0
    ```

    #### 8.4 Add Type Hints and Docstrings

    Ensure all functions have:
    ```python
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        '''
        Add derived features to Pokemon card data.

        Args:
            df: DataFrame with columns [hp, attack, defense, ...]

        Returns:
            DataFrame with additional feature columns

        Raises:
            ValueError: If required columns are missing
        '''
        pass
    ```

    ---

    **💡 Deliverable**: README + PR description + type-hinted code
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Phase 9: Final Evaluation

    **Only after completing all phases above!**

    ### 9.1 Test Set Evaluation

    Now you can finally use the test set:

    ```python
    # Load test set (you split this in Phase 2)
    X_test = engineer_features(test)
    y_test = test['price_usd']

    # Predict
    test_pred = best_model.predict(X_test)

    # Evaluate
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    print("Final Test Set Results:")
    print(f"MAE: ${mean_absolute_error(y_test, test_pred):.2f}")
    print(f"RMSE: ${mean_squared_error(y_test, test_pred, squared=False):.2f}")
    print(f"R²: {r2_score(y_test, test_pred):.4f}")
    ```

    ### 9.2 Compare to Validation Results

    | Metric | Validation | Test | Difference |
    |--------|------------|------|------------|
    | MAE | $X.XX | $X.XX | X.X% |
    | RMSE | $X.XX | $X.XX | X.X% |
    | R² | 0.XX | 0.XX | X.X% |

    **If test results are significantly worse than validation**:
    - You may have overfit to the validation set
    - Consider using nested cross-validation

    ### 9.3 Self-Assessment

    Use the rubric in `CAPSTONE_RUBRIC.md` to grade your work.

    Rate yourself (1-5) on:
    - Business understanding
    - Data engineering
    - Feature engineering
    - Model development
    - Evaluation rigor
    - Deployment readiness
    - Monitoring strategy
    - Documentation quality

    ### 9.4 Reflection

    Answer these questions:

    1. **What was the hardest part of this project?**

       _Your answer_

    2. **What would you do differently if you had more time?**

       _Your answer_

    3. **What surprised you?**

       _Your answer_

    4. **What ML concept finally "clicked" for you?**

       _Your answer_

    5. **How would you explain this project to a non-technical stakeholder?**

       _Your answer (practice this!!)_

    ---

    **💡 Final Deliverable**: Complete GitHub repo with all code, docs, and self-assessment
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## 🎓 You're Done!

    Congratulations! You've completed an end-to-end ML engineering project.

    ### What You've Learned

    ✅ Frame business problems before writing code
    ✅ Build reliable, reproducible data pipelines
    ✅ Engineer features without data leakage
    ✅ Train and evaluate models systematically
    ✅ Design production-ready ML systems
    ✅ Plan monitoring and incident response
    ✅ Document your work for team collaboration

    ### Next Steps

    1. **Share your work**: Add this to your portfolio
    2. **Get feedback**: Submit your PR description for review
    3. **Extend the project**: Add new features, try different models
    4. **Apply to real problems**: Use this workflow at work

    ### Real-World Applications

    This workflow applies to:
    - Recommendation systems
    - Fraud detection
    - Demand forecasting
    - Customer churn prediction
    - Image classification
    - NLP tasks
    - And more!

    ### Keep Learning

    - Advanced: AutoML, neural networks, deep learning
    - Production: Kubernetes, model serving frameworks
    - MLOps: CI/CD for ML, feature stores, model registries
    - Domain: Computer vision, NLP, time series, reinforcement learning

    ---

    **You're now equipped to be a strong ML engineer at any company. Go build something awesome! 🚀**
    """)
    return


if __name__ == "__main__":
    app.run()
