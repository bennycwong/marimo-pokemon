"""
Module 5 Exercises: Deployment & Inference
===========================================

Hands-on practice with production ML deployment.

Duration: 2-3 hours
Difficulty: Intermediate to Advanced
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
    # Module 5 Exercises: Deployment & Inference

    ## Learning Objectives
    - Build production-ready APIs with FastAPI
    - Handle edge cases gracefully
    - Validate inputs with Pydantic
    - Optimize inference latency
    - Design A/B tests

    ## Exercises Overview

    1. **Exercise 1**: Build a FastAPI endpoint with proper validation
    2. **Exercise 2**: Handle 10 edge cases
    3. **Exercise 3**: Optimize latency from 500ms to <100ms
    4. **Exercise 4**: Design an A/B test plan
    5. **Exercise 5**: Create a health check endpoint

    **Estimated time**: 2-3 hours

    ---
    """)
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import pickle
    import time
    from typing import Dict, List, Optional, Any

    # Load the trained model
    MODEL_PATH = Path("models/pokemon_classifier_v1.pkl")

    if MODEL_PATH.exists():
        with open(MODEL_PATH, 'rb') as f:
            production_model = pickle.load(f)
        print("✅ Model loaded successfully")
    else:
        print("⚠️ Model not found. Run Module 5 first to train and save the model.")
        production_model = None

    return (
        Any,
        Dict,
        MODEL_PATH,
        Optional,
        Path,
        pd,
        pickle,
        production_model,
    )


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Exercise 1: Build FastAPI Endpoint with Validation

    **Objective**: Create a production-ready API endpoint with proper input validation.

    ### Requirements:
    1. Use FastAPI and Pydantic for input validation
    2. Validate all Pokemon stats (hp, attack, defense, etc.)
    3. Ensure stats are within valid ranges (1-300)
    4. Return prediction with confidence score
    5. Handle errors gracefully with appropriate status codes

    ### TODO:
    Fill in the code below to create a working API endpoint.
    """)
    return


@app.cell
def __():
    from pydantic import BaseModel, Field, validator
    from fastapi import FastAPI, HTTPException

    # TODO: Define the input schema
    class PokemonCardInput(BaseModel):
        '''Input schema for Pokemon card prediction.'''
        # TODO: Add fields with validation
        # hp: int = Field(..., ge=1, le=300, description="Hit points")
        # attack: int = Field(...)
        # ... add all required fields
        pass

        # TODO: Add custom validators if needed
        # @validator('*')
        # def validate_stats(cls, v):
        #     ... your validation logic
        #     return v

    # TODO: Create FastAPI app
    # app = FastAPI(title="Pokemon Type Classifier API")

    # TODO: Create prediction endpoint
    # @app.post("/predict")
    # def predict_pokemon_type(card: PokemonCardInput):
    #     try:
    #         # Extract features
    #         features = ...
    #         # Make prediction
    #         prediction = production_model.predict([features])[0]
    #         probabilities = production_model.predict_proba([features])[0]
    #         confidence = float(max(probabilities))
    #
    #         return {
    #             'success': True,
    #             'predicted_type': prediction,
    #             'confidence': confidence,
    #             'all_probabilities': dict(zip(production_model.classes_, probabilities))
    #         }
    #     except Exception as e:
    #         raise HTTPException(status_code=500, detail=str(e))

    print("TODO: Implement the FastAPI endpoint above")
    return BaseModel, FastAPI, Field, HTTPException, PokemonCardInput, validator


@app.cell
def __(mo):
    mo.md("""
    **💡 Testing Your API**

    Once implemented, you can test your API with:

    ```python
    # Test valid input
    test_card = {
        'hp': 90,
        'attack': 85,
        'defense': 75,
        'sp_attack': 110,
        'sp_defense': 95,
        'speed': 80,
        'generation': 1,
        'is_legendary': False
    }

    result = predict_pokemon_type(PokemonCardInput(**test_card))
    print(result)
    ```

    **Success Criteria**:
    - ✅ API accepts valid inputs
    - ✅ API rejects invalid inputs with clear error messages
    - ✅ Returns prediction with confidence score
    - ✅ Handles errors gracefully
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Exercise 2: Handle Edge Cases

    **Objective**: Your API must handle all these edge cases gracefully.

    ### Test Cases (10 scenarios):

    Test each case and document how your API handles it:
    """)
    return


@app.cell
def __():
    # Edge case test scenarios
    edge_cases = {
        'case_1_all_zeros': {
            'name': 'All stats are 0',
            'input': {'hp': 0, 'attack': 0, 'defense': 0, 'sp_attack': 0,
                     'sp_defense': 0, 'speed': 0, 'generation': 1, 'is_legendary': False},
            'expected': 'Should reject (stats must be > 0)'
        },
        'case_2_all_max': {
            'name': 'All stats at maximum',
            'input': {'hp': 300, 'attack': 300, 'defense': 300, 'sp_attack': 300,
                     'sp_defense': 300, 'speed': 300, 'generation': 1, 'is_legendary': True},
            'expected': 'Should accept and predict (rare but valid)'
        },
        'case_3_negative': {
            'name': 'Negative stats',
            'input': {'hp': -10, 'attack': 50, 'defense': 50, 'sp_attack': 50,
                     'sp_defense': 50, 'speed': 50, 'generation': 1, 'is_legendary': False},
            'expected': 'Should reject (stats cannot be negative)'
        },
        'case_4_missing_field': {
            'name': 'Missing required field',
            'input': {'hp': 90, 'attack': 85},  # Missing other fields
            'expected': 'Should reject with clear message about missing fields'
        },
        'case_5_wrong_type': {
            'name': 'Wrong data type',
            'input': {'hp': 'ninety', 'attack': 85, 'defense': 75, 'sp_attack': 110,
                     'sp_defense': 95, 'speed': 80, 'generation': 1, 'is_legendary': False},
            'expected': 'Should reject with type error'
        },
        'case_6_future_generation': {
            'name': 'Generation 100 (doesn\'t exist)',
            'input': {'hp': 90, 'attack': 85, 'defense': 75, 'sp_attack': 110,
                     'sp_defense': 95, 'speed': 80, 'generation': 100, 'is_legendary': False},
            'expected': 'Should accept (model will extrapolate) or reject with warning'
        },
        'case_7_extreme_imbalance': {
            'name': 'Extreme stat imbalance',
            'input': {'hp': 10, 'attack': 300, 'defense': 10, 'sp_attack': 10,
                     'sp_defense': 10, 'speed': 10, 'generation': 1, 'is_legendary': False},
            'expected': 'Should accept (unusual but valid)'
        },
        'case_8_extra_fields': {
            'name': 'Extra unexpected fields',
            'input': {'hp': 90, 'attack': 85, 'defense': 75, 'sp_attack': 110,
                     'sp_defense': 95, 'speed': 80, 'generation': 1, 'is_legendary': False,
                     'favorite_food': 'pizza', 'color': 'red'},
            'expected': 'Should ignore extra fields or reject depending on strict mode'
        },
        'case_9_null_values': {
            'name': 'Null/None values',
            'input': {'hp': None, 'attack': 85, 'defense': 75, 'sp_attack': 110,
                     'sp_defense': 95, 'speed': 80, 'generation': 1, 'is_legendary': False},
            'expected': 'Should reject (nulls not allowed)'
        },
        'case_10_float_values': {
            'name': 'Float instead of int',
            'input': {'hp': 90.5, 'attack': 85.3, 'defense': 75, 'sp_attack': 110,
                     'sp_defense': 95, 'speed': 80, 'generation': 1, 'is_legendary': False},
            'expected': 'Should either round to int or reject'
        }
    }

    # TODO: Test each case with your API
    # for case_name, case_data in edge_cases.items():
    #     print(f"\nTesting: {case_data['name']}")
    #     try:
    #         result = your_api_predict(case_data['input'])
    #         print(f"  ✅ Result: {result}")
    #     except Exception as e:
    #         print(f"  ❌ Error: {e}")
    #     print(f"  Expected: {case_data['expected']}")

    print(f"📝 {len(edge_cases)} edge cases defined. Test them with your API!")
    return (edge_cases,)


@app.cell
def __(mo):
    mo.md("""
    **Your Results Table**

    Document your findings:

    | Case | Input | Your Result | Pass/Fail | Notes |
    |------|-------|-------------|-----------|-------|
    | 1. All zeros | stats=0 | | | |
    | 2. All max | stats=300 | | | |
    | 3. Negative | hp=-10 | | | |
    | 4. Missing field | Only hp, attack | | | |
    | 5. Wrong type | hp="ninety" | | | |
    | 6. Future gen | gen=100 | | | |
    | 7. Imbalanced | attack=300, rest=10 | | | |
    | 8. Extra fields | favorite_food="pizza" | | | |
    | 9. Null values | hp=None | | | |
    | 10. Float values | hp=90.5 | | | |

    **Success Criteria**:
    - ✅ All 10 cases handled gracefully (accept or reject with clear message)
    - ✅ No unexpected crashes
    - ✅ Error messages are helpful to users
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Exercise 3: Optimize Inference Latency

    **Objective**: Reduce prediction latency from 500ms to <100ms

    ### Current Slow Implementation:

    ```python
    def predict_slow(card_data):
        # Load model from disk every time (SLOW!)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Create DataFrame (SLOW for single prediction!)
        df = pd.DataFrame([card_data])

        # Feature engineering (repeated work!)
        df['total_stats'] = df['hp'] + df['attack'] + ...

        # Predict
        return model.predict(df)[0]
    ```

    ### TODO: Optimize this implementation

    **Optimization strategies**:
    1. Load model once at startup (not per request)
    2. Use numpy arrays instead of DataFrame for single predictions
    3. Pre-compute what you can
    4. Cache common predictions (if applicable)
    5. Batch predictions if possible

    ### Benchmark your optimization:
    """)
    return


@app.cell
def __(production_model):
    import time as _time

    # TODO: Implement optimized prediction function
    def predict_optimized(card_features):
        '''
        Optimized prediction function.

        Args:
            card_features: List or numpy array of features

        Returns:
            str: Predicted Pokemon type
        '''
        # TODO: Your optimized implementation
        # Hint: Use production_model directly (already loaded)
        # Hint: Use numpy arrays, not DataFrames
        # Hint: Pre-compute derived features
        pass

    # Benchmark
    test_features = [90, 85, 75, 110, 95, 80, 1, 0]  # Example features

    if production_model:
        # Warm up
        _ = production_model.predict([test_features])

        # Benchmark
        start = _time.time()
        for _ in range(1000):
            # TODO: Call your optimized function
            # prediction = predict_optimized(test_features)
            pass
        elapsed = _time.time() - start

        avg_latency = (elapsed / 1000) * 1000  # milliseconds
        print(f"Average latency: {avg_latency:.2f}ms per prediction")
        print(f"Throughput: {1000 / elapsed:.0f} predictions/second")

        if avg_latency < 100:
            print("✅ Target achieved! (<100ms)")
        else:
            print(f"❌ Still too slow. Target: <100ms, Current: {avg_latency:.2f}ms")
    else:
        print("⚠️ Load the model first")
    return avg_latency, elapsed, predict_optimized, test_features


@app.cell
def __(mo):
    mo.md("""
    **Success Criteria**:
    - ✅ Average latency < 100ms
    - ✅ Throughput > 100 predictions/second
    - ✅ Same predictions as original (correctness)

    **Bonus**: Can you get it under 10ms? 1ms?
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Exercise 4: Design an A/B Test

    **Objective**: Plan how to validate your model in production.

    ### Scenario:
    You're deploying a new Pokemon type classifier to production. You need to prove
    it's better than the current rule-based system before full rollout.

    ### TODO: Design your A/B test

    Fill in this template:
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### A/B Test Plan Template

    **1. Hypothesis**
    - **Null hypothesis (H0)**: _[Your answer: e.g., "ML model performs same as rule-based system"]_
    - **Alternative hypothesis (H1)**: _[Your answer: e.g., "ML model has 10% higher accuracy"]_

    **2. Metrics**

    | Metric Type | Metric Name | How to Measure | Target |
    |-------------|-------------|----------------|--------|
    | Primary | _[e.g., Prediction Accuracy]_ | _[e.g., % correct on labeled data]_ | _[e.g., 85%]_ |
    | Secondary | _[e.g., User acceptance rate]_ | _[e.g., % users who accept prediction]_ | _[e.g., 70%]_ |
    | Guardrail | _[e.g., Latency]_ | _[e.g., p95 response time]_ | _[e.g., <100ms]_ |

    **3. Experiment Design**

    - **Traffic split**: _[Your answer: e.g., 90% control, 10% treatment]_
    - **Duration**: _[Your answer: e.g., 2 weeks]_
    - **Sample size needed**: _[Your answer: calculations]_
    - **Randomization unit**: _[Your answer: e.g., user_id, session_id, request_id]_

    **4. Success Criteria**

    We will roll out to 100% if:
    - [ ] Primary metric improves by ___% (with p<0.05)
    - [ ] No degradation in secondary metrics
    - [ ] All guardrail metrics within bounds
    - [ ] No increase in error rate

    **5. Risks & Mitigations**

    | Risk | Likelihood | Impact | Mitigation |
    |------|------------|--------|------------|
    | _[e.g., Model crashes under load]_ | _[High/Med/Low]_ | _[High/Med/Low]_ | _[e.g., Load testing before launch]_ |
    | _[e.g., Predictions wrong for rare types]_ | _[Med]_ | _[Med]_ | _[e.g., Monitor per-type accuracy]_ |

    **6. Rollback Plan**

    If things go wrong:
    1. _[Step 1: e.g., Disable treatment group immediately]_
    2. _[Step 2: e.g., Route all traffic to control]_
    3. _[Step 3: e.g., Investigate logs and metrics]_
    4. _[Step 4: e.g., Fix issue and re-test in staging]_

    **7. Analysis Plan**

    After experiment ends:
    1. _[Statistical test to use: e.g., t-test, chi-square]_
    2. _[Confidence level: e.g., 95%]_
    3. _[Segmentation analysis: e.g., by Pokemon type, generation]_
    4. _[Documentation: e.g., Write report with recommendations]_
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    **Success Criteria**:
    - ✅ Clear hypothesis stated
    - ✅ Metrics cover model, business, and guardrails
    - ✅ Sample size and duration calculated
    - ✅ Risks identified with mitigations
    - ✅ Rollback plan in place

    **Questions to consider**:
    - What if results are inconclusive?
    - How do you handle seasonality (e.g., new Pokemon releases)?
    - What if different user segments behave differently?
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Exercise 5: Health Check Endpoint

    **Objective**: Create a health check endpoint to monitor model availability.

    ### Requirements:
    1. Check if model is loaded
    2. Check if model can make predictions
    3. Return response time
    4. Return model version
    5. Check dependencies (optional)

    ### TODO: Implement health check
    """)
    return


@app.cell
def __(production_model):
    import time as _time_health

    # TODO: Implement health check endpoint
    def health_check():
        '''
        Health check for the prediction service.

        Returns:
            dict: Health status and diagnostics
        '''
        health_status = {
            'status': 'unknown',
            'checks': {},
            'timestamp': _time_health.time()
        }

        # TODO: Check 1 - Model loaded
        # health_status['checks']['model_loaded'] = production_model is not None

        # TODO: Check 2 - Can predict
        # try:
        #     test_input = [90, 85, 75, 110, 95, 80, 1, 0]
        #     start = _time_health.time()
        #     _ = production_model.predict([test_input])
        #     latency = (_time_health.time() - start) * 1000
        #     health_status['checks']['prediction_works'] = True
        #     health_status['checks']['prediction_latency_ms'] = latency
        # except Exception as e:
        #     health_status['checks']['prediction_works'] = False
        #     health_status['checks']['error'] = str(e)

        # TODO: Check 3 - Model version
        # health_status['model_version'] = '1.0.0'  # Load from metadata

        # TODO: Set overall status
        # all_checks_pass = all(health_status['checks'].values())
        # health_status['status'] = 'healthy' if all_checks_pass else 'unhealthy'

        return health_status

    # Test health check
    if production_model:
        result = health_check()
        print("Health check result:")
        print(result)
    else:
        print("⚠️ Model not loaded")
    return (health_check,)


@app.cell
def __(mo):
    mo.md("""
    **Example healthy response**:

    ```json
    {
      "status": "healthy",
      "checks": {
        "model_loaded": true,
        "prediction_works": true,
        "prediction_latency_ms": 2.3
      },
      "model_version": "1.0.0",
      "timestamp": 1234567890.123
    }
    ```

    **Success Criteria**:
    - ✅ Returns 'healthy' when everything works
    - ✅ Returns 'unhealthy' with details when something fails
    - ✅ Can be called by monitoring systems (Kubernetes, load balancers)
    - ✅ Completes quickly (< 50ms)
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Bonus Exercise: Load Testing

    **Objective**: Test your API under realistic load.

    ### Scenario:
    Your API needs to handle:
    - 100 requests/second (normal traffic)
    - 500 requests/second (peak traffic)
    - Latency p95 < 100ms

    ### TODO: Write a load test

    ```python
    import concurrent.futures
    import time

    def load_test(n_requests=1000, n_workers=10):
        '''Simulate concurrent requests.'''

        def make_request():
            # TODO: Call your predict function
            start = time.time()
            # result = predict(...)
            latency = (time.time() - start) * 1000
            return latency

        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            latencies = list(executor.map(lambda _: make_request(), range(n_requests)))

        # Analyze results
        import numpy as np
        print(f"Total requests: {n_requests}")
        print(f"Mean latency: {np.mean(latencies):.2f}ms")
        print(f"p50 latency: {np.percentile(latencies, 50):.2f}ms")
        print(f"p95 latency: {np.percentile(latencies, 95):.2f}ms")
        print(f"p99 latency: {np.percentile(latencies, 99):.2f}ms")

        return latencies

    # Run load test
    # latencies = load_test(n_requests=1000, n_workers=50)
    ```

    **Success Criteria**:
    - ✅ Handles 100 req/s without errors
    - ✅ p95 latency < 100ms under load
    - ✅ No memory leaks (memory stable over time)
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---
    ## Summary & Next Steps

    You've practiced:
    - ✅ Building production-ready APIs
    - ✅ Handling edge cases gracefully
    - ✅ Optimizing inference latency
    - ✅ Designing A/B tests
    - ✅ Creating health checks

    ### Self-Assessment

    Rate yourself (1-5) on each skill:

    | Skill | Rating | Evidence |
    |-------|--------|----------|
    | API design | __ / 5 | Can build FastAPI endpoints with validation |
    | Error handling | __ / 5 | Handled all 10 edge cases |
    | Performance optimization | __ / 5 | Achieved <100ms latency |
    | Experiment design | __ / 5 | Complete A/B test plan |
    | Production readiness | __ / 5 | Health checks and monitoring |

    **Target**: 4+ on all skills

    ### Next Steps

    - **Module 6**: Learn production monitoring and debugging
    - **Module 7**: Team collaboration and code reviews
    - **Module 8**: Complete end-to-end capstone project

    **Real-world application**: These skills apply to any production ML system!
    """)
    return


if __name__ == "__main__":
    app.run()
