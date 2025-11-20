"""
Module 5: Deployment, Inference & Monitoring
============================================

Professional ML Engineering Onboarding Project
Pokemon Card Type Classification

Learning Objectives:
- Serialize and version models
- Build inference APIs
- Handle production edge cases
- Monitor model health
- Optimize inference latency

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
        # Module 5: Deployment & Inference

        **"Models in notebooks aren't models in production"** — Every ML engineer

        You've built a great model. Now the hard part: **making it work in production!**

        ## What You'll Learn

        1. **Model Serialization**: Save and load models correctly
        2. **Inference API**: Build production-ready prediction service
        3. **Input Validation**: Handle bad inputs gracefully
        4. **Latency Optimization**: Make predictions fast
        5. **Monitoring**: Detect when things go wrong

        ## Why Deployment is Hard

        **Research models vs Production models**:
        - Research: Accuracy matters most
        - Production: Reliability, latency, cost, maintainability matter

        **Common production failures**:
        - Model expects features in wrong order
        - Input data has different schema
        - Latency too high under load
        - Model degrades over time (data drift)
        - No monitoring, issues go undetected

        Let's learn to deploy like a pro!
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import warnings
    import pickle
    import json
    from typing import Dict, List, Optional, Any
    import time
    warnings.filterwarnings('ignore')
    return (
        Any,
        Dict,
        List,
        Optional,
        Path,
        json,
        np,
        pd,
        pickle,
        plt,
        time,
        warnings,
    )


@app.cell
def __(Path, pd):
    # Train our final production model
    DATA_PATH = Path("data/clean/pokemon_cards_clean_latest.csv")
    df = pd.read_csv(DATA_PATH)

    # Feature engineering
    df['total_stats'] = df['hp'] + df['attack'] + df['defense'] + df['sp_attack'] + df['sp_defense'] + df['speed']
    df['physical_bias'] = (df['attack'] + df['defense']) - (df['sp_attack'] + df['sp_defense'])

    feature_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
                    'total_stats', 'physical_bias', 'generation', 'is_legendary']
    X = df[feature_cols]
    y = df['type']

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train final model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    production_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ])

    production_model.fit(X_train, y_train)

    print(f"✅ Production model trained")
    print(f"   Features: {feature_cols}")
    print(f"   Classes: {sorted(y.unique())}")
    return (
        DATA_PATH,
        Pipeline,
        RandomForestClassifier,
        StandardScaler,
        X,
        X_test,
        X_train,
        df,
        feature_cols,
        production_model,
        train_test_split,
        y,
        y_test,
        y_train,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Section 1: Model Serialization (Saving Your Work)

        **Key question**: How do we save a trained model for later use?

        ### Common Approaches

        1. **Pickle** (Python standard): Simple but Python-only
        2. **Joblib** (scikit-learn): Better for large numpy arrays
        3. **ONNX**: Cross-platform format
        4. **TensorFlow SavedModel** / **PyTorch torchscript**: Framework-specific

        **What to save**:
        - ✅ Trained model
        - ✅ Preprocessing pipeline (scaler, encoders)
        - ✅ Feature names and order
        - ✅ Model version and metadata

        Let's save our model properly:
        """
    )
    return


@app.cell
def __(Path, feature_cols, json, pickle, production_model):
    # Create models directory
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)

    # Model version
    MODEL_VERSION = "v1.0.0"

    # Save model
    model_path = MODELS_DIR / f"pokemon_classifier_{MODEL_VERSION}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(production_model, f)

    # Save metadata
    metadata = {
        'version': MODEL_VERSION,
        'model_type': 'RandomForestClassifier',
        'feature_names': feature_cols,
        'n_features': len(feature_cols),
        'classes': sorted(production_model.classes_.tolist()),
        'training_date': '2025-01-19',
        'sklearn_version': '1.4.0'
    }

    metadata_path = MODELS_DIR / f"pokemon_classifier_{MODEL_VERSION}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Metadata saved to: {metadata_path}")
    print(f"\nModel size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    return (
        MODEL_VERSION,
        MODELS_DIR,
        f,
        metadata,
        metadata_path,
        model_path,
    )


@app.cell
def __(metadata_path, model_path, pickle):
    # Load model (simulating production environment)
    def load_model(model_path, metadata_path):
        """Load model and metadata for inference."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"✅ Loaded model version: {metadata['version']}")
        print(f"   Expected features: {metadata['feature_names']}")

        return model, metadata

    loaded_model, loaded_metadata = load_model(model_path, metadata_path)
    return f, load_model, loaded_metadata, loaded_model


@app.cell
def __(mo):
    mo.md(
        """
        ### 💡 Model Serialization Best Practices

        **Always include**:
        - Model version (semantic versioning: major.minor.patch)
        - Feature names and expected order
        - Training date
        - Library versions (scikit-learn, python, etc.)
        - Performance metrics

        **Common mistakes**:
        - ❌ Forgetting to save preprocessing steps
        - ❌ Not versioning models
        - ❌ Saving model but not metadata
        - ❌ Using pickle across Python versions (can break!)

        **Production tip**: Use a model registry (MLflow Model Registry, Sagemaker Model Registry, etc.)

        ---
        ## Section 2: Inference API (Making Predictions)

        **Goal**: Build a simple prediction service.

        In production, you'd use FastAPI, Flask, or BentoML. For this project,
        we'll build the core prediction logic with proper validation.
        """
    )
    return


@app.cell
def __(Any, Dict, List, Optional, loaded_metadata, loaded_model, pd):
    class PokemonTypePredictor:
        """Production-ready Pokemon type prediction service."""

        def __init__(self, model, metadata):
            self.model = model
            self.metadata = metadata
            self.feature_names = metadata['feature_names']
            self.expected_classes = metadata['classes']

        def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
            """
            Validate input data.

            Returns:
                (is_valid, error_message)
            """
            # Check required features
            missing_features = set(self.feature_names) - set(input_data.keys())
            if missing_features:
                return False, f"Missing features: {missing_features}"

            # Check feature types and ranges
            try:
                for feature in self.feature_names:
                    value = input_data[feature]

                    # Check type
                    if feature == 'is_legendary':
                        if not isinstance(value, (bool, int)):
                            return False, f"{feature} must be boolean or int"
                    else:
                        if not isinstance(value, (int, float)):
                            return False, f"{feature} must be numeric"

                    # Check ranges (example)
                    if feature in ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']:
                        if not (0 <= value <= 200):
                            return False, f"{feature} must be between 0 and 200"

                    if feature == 'generation':
                        if not (1 <= value <= 9):
                            return False, f"generation must be between 1 and 9"

            except Exception as e:
                return False, f"Validation error: {str(e)}"

            return True, None

        def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            """
            Make prediction with validation and error handling.

            Args:
                input_data: Dictionary with feature values

            Returns:
                Dictionary with prediction and metadata
            """
            # Validate input
            is_valid, error_msg = self.validate_input(input_data)
            if not is_valid:
                return {
                    'success': False,
                    'error': error_msg,
                    'prediction': None,
                    'confidence': None
                }

            try:
                # Prepare input (ensure correct order)
                input_df = pd.DataFrame([input_data])[self.feature_names]

                # Make prediction
                prediction = self.model.predict(input_df)[0]
                probabilities = self.model.predict_proba(input_df)[0]
                confidence = float(max(probabilities))

                # Get top 3 predictions
                top_3_idx = probabilities.argsort()[-3:][::-1]
                top_3_predictions = [
                    {
                        'type': self.model.classes_[idx],
                        'probability': float(probabilities[idx])
                    }
                    for idx in top_3_idx
                ]

                return {
                    'success': True,
                    'prediction': prediction,
                    'confidence': confidence,
                    'top_3': top_3_predictions,
                    'model_version': self.metadata['version']
                }

            except Exception as e:
                return {
                    'success': False,
                    'error': f"Prediction error: {str(e)}",
                    'prediction': None,
                    'confidence': None
                }

    # Create predictor service
    predictor = PokemonTypePredictor(loaded_model, loaded_metadata)

    print("✅ Prediction service initialized")
    return (PokemonTypePredictor, predictor)


@app.cell
def __(predictor):
    # Test the prediction service
    test_input = {
        'hp': 180,
        'attack': 130,
        'defense': 110,
        'sp_attack': 140,
        'sp_defense': 115,
        'speed': 120,
        'total_stats': 795,  # Sum of above
        'physical_bias': -15,  # (attack + defense) - (sp_attack + sp_defense)
        'generation': 1,
        'is_legendary': True
    }

    result = predictor.predict(test_input)

    print("Test Prediction:")
    print(json.dumps(result, indent=2))
    return result, test_input


@app.cell
def __(mo, predictor):
    # Create interactive UI for testing
    hp_slider = mo.ui.slider(start=20, stop=200, value=100, step=1, label="HP")
    attack_slider = mo.ui.slider(start=20, stop=200, value=100, step=1, label="Attack")
    defense_slider = mo.ui.slider(start=20, stop=200, value=100, step=1, label="Defense")
    sp_attack_slider = mo.ui.slider(start=20, stop=200, value=100, step=1, label="Sp. Attack")
    sp_defense_slider = mo.ui.slider(start=20, stop=200, value=100, step=1, label="Sp. Defense")
    speed_slider = mo.ui.slider(start=20, stop=200, value=100, step=1, label="Speed")
    generation_slider = mo.ui.slider(start=1, stop=9, value=1, step=1, label="Generation")
    legendary_checkbox = mo.ui.checkbox(label="Is Legendary?")

    mo.md(
        f"""
        ## 🎮 Interactive Pokemon Type Predictor

        Adjust the stats below to see predictions:

        {mo.hstack([hp_slider, attack_slider, defense_slider])}
        {mo.hstack([sp_attack_slider, sp_defense_slider, speed_slider])}
        {mo.hstack([generation_slider, legendary_checkbox])}
        """
    )
    return (
        attack_slider,
        defense_slider,
        generation_slider,
        hp_slider,
        legendary_checkbox,
        sp_attack_slider,
        sp_defense_slider,
        speed_slider,
    )


@app.cell
def __(
    attack_slider,
    defense_slider,
    generation_slider,
    hp_slider,
    legendary_checkbox,
    mo,
    predictor,
    sp_attack_slider,
    sp_defense_slider,
    speed_slider,
):
    # Make prediction based on UI inputs
    ui_input = {
        'hp': hp_slider.value,
        'attack': attack_slider.value,
        'defense': defense_slider.value,
        'sp_attack': sp_attack_slider.value,
        'sp_defense': sp_defense_slider.value,
        'speed': speed_slider.value,
        'total_stats': (hp_slider.value + attack_slider.value + defense_slider.value +
                       sp_attack_slider.value + sp_defense_slider.value + speed_slider.value),
        'physical_bias': ((attack_slider.value + defense_slider.value) -
                         (sp_attack_slider.value + sp_defense_slider.value)),
        'generation': generation_slider.value,
        'is_legendary': legendary_checkbox.value
    }

    ui_result = predictor.predict(ui_input)

    if ui_result['success']:
        mo.md(
            f"""
            ### Prediction Result

            **Predicted Type**: {ui_result['prediction']} ({ui_result['confidence']:.1%} confidence)

            **Top 3 Predictions**:
            1. {ui_result['top_3'][0]['type']}: {ui_result['top_3'][0]['probability']:.1%}
            2. {ui_result['top_3'][1]['type']}: {ui_result['top_3'][1]['probability']:.1%}
            3. {ui_result['top_3'][2]['type']}: {ui_result['top_3'][2]['probability']:.1%}
            """
        )
    else:
        mo.md(f"**Error**: {ui_result['error']}")
    return ui_input, ui_result


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Section 3: Production Considerations

        ### Latency Optimization

        **Production requirements**: Predictions must be fast!
        - Web API: <200ms total (including network)
        - Real-time: <10ms for model inference
        - Batch: Throughput matters more than latency

        **Optimization strategies**:
        1. **Model simplification**: Fewer trees, smaller depth
        2. **Feature selection**: Remove low-importance features
        3. **Model quantization**: Reduce precision (float32 → float16)
        4. **Batch predictions**: Predict multiple samples at once
        5. **Caching**: Cache frequent predictions
        6. **Model distillation**: Train smaller model to mimic large one

        ---
        ## Section 4: Monitoring & Observability

        **Key insight**: Models degrade over time without you noticing!

        **What to monitor**:
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### 📊 Production Monitoring Checklist

        **Input Monitoring**:
        - [ ] Feature distributions (detect drift)
        - [ ] Missing values rate
        - [ ] Out-of-range values
        - [ ] Request volume

        **Output Monitoring**:
        - [ ] Prediction distribution
        - [ ] Confidence distribution
        - [ ] Per-class prediction rate
        - [ ] Response latency (p50, p95, p99)

        **Performance Monitoring**:
        - [ ] Accuracy (when ground truth available)
        - [ ] Precision/Recall per class
        - [ ] Confusion patterns
        - [ ] Error rate

        **Infrastructure Monitoring**:
        - [ ] CPU/Memory usage
        - [ ] Request queue depth
        - [ ] Error rate (500s)
        - [ ] Availability/uptime

        ### 🚨 Alerting Thresholds

        **Critical alerts** (page someone!):
        - Prediction error rate > 5%
        - Latency p95 > 500ms
        - Service down

        **Warning alerts** (investigate next day):
        - Feature distribution drift > 20%
        - Prediction distribution change > 10%
        - Confidence drop > 5%

        ---
        ## Section 5: Data Drift Detection

        **Data drift** = Distribution of input features changes over time

        **Why it matters**: Model trained on old data won't work on new data!
        """
    )
    return


@app.cell
def __(X_train, np):
    # Simulate data drift detection
    def detect_drift_simple(reference_data, new_data, threshold=0.1):
        """
        Simple drift detection using statistical distance.

        Args:
            reference_data: Training data statistics
            new_data: Production data statistics
            threshold: Drift threshold

        Returns:
            Dict with drift metrics per feature
        """
        drift_scores = {}

        for col in reference_data.columns:
            # Compare means and stds
            ref_mean = reference_data[col].mean()
            ref_std = reference_data[col].std()

            new_mean = new_data[col].mean()
            new_std = new_data[col].std()

            # Simple normalized difference
            mean_diff = abs(new_mean - ref_mean) / (ref_std + 1e-6)
            std_diff = abs(new_std - ref_std) / (ref_std + 1e-6)

            drift_score = (mean_diff + std_diff) / 2

            drift_scores[col] = {
                'drift_score': drift_score,
                'is_drifted': drift_score > threshold,
                'ref_mean': ref_mean,
                'new_mean': new_mean
            }

        return drift_scores

    # Simulate "new" data with drift
    X_new = X_train.copy()
    X_new['total_stats'] = X_new['total_stats'] * 1.2  # Simulate power creep
    X_new['generation'] = X_new['generation'] + 2  # New generations

    drift_results = detect_drift_simple(X_train, X_new[:100], threshold=0.1)

    print("Drift Detection Results:")
    print("=" * 60)
    for feature, metrics in drift_results.items():
        status = "🚨 DRIFT" if metrics['is_drifted'] else "✅ OK"
        print(f"{feature:20s}: {status} (score: {metrics['drift_score']:.3f})")
    return X_new, detect_drift_simple, drift_results, feature, metrics, status


@app.cell
def __(mo):
    mo.md(
        """
        ### 💡 Handling Data Drift

        **When drift is detected**:
        1. **Investigate**: What changed in the data?
        2. **Assess impact**: Is model performance affected?
        3. **Decision**:
           - Minor drift: Continue monitoring
           - Moderate drift: Retrain with new data
           - Severe drift: Retrain + add new features

        **Prevention strategies**:
        - Regular retraining schedule
        - Online learning (continuous updates)
        - Robust features (less sensitive to drift)
        - Ensemble models (more stable)

        ---
        ## Key Takeaways

        ### ✅ What You Learned

        1. **Model serialization**: Save models + metadata properly
        2. **Input validation**: Handle edge cases gracefully
        3. **Monitoring**: Track inputs, outputs, and performance
        4. **Data drift**: Detect and respond to distribution changes
        5. **Production mindset**: Reliability > accuracy

        ### 🤔 Socratic Questions

        1. **"You deployed a model. A week later, accuracy drops from 85% to 60%. What do you investigate first?"**
           - Data drift? Code bug? Upstream data issue? Ground truth labels correct?

        2. **"Training takes 1 hour. Inference must complete in <100ms. How do you approach this?"**
           - Simplify model, feature selection, batch prediction, caching, distillation

        3. **"A user sends malformed input that crashes your API. Who's responsible - you or the user?"**
           - You! Production code must handle all inputs gracefully.

        ---
        ## 🏢 Industry Context

        ### How Companies Deploy Models

        **Small Startup**:
        - Docker container
        - FastAPI/Flask
        - Manual deployment
        - Basic monitoring (CloudWatch, Datadog)

        **Medium Company**:
        - Kubernetes
        - BentoML/Seldon
        - CI/CD pipelines
        - Structured monitoring (Prometheus, Grafana)

        **Large Company**:
        - Custom ML platform
        - A/B testing framework
        - Canary deployments
        - Comprehensive observability

        ### Common Deployment Patterns

        1. **Batch inference**: Run predictions offline, store results
        2. **Online inference**: Real-time predictions via API
        3. **Edge inference**: Model runs on device (mobile, IoT)
        4. **Streaming inference**: Process stream of events (Kafka, Kinesis)

        ---
        ## 🎯 Module 5 Checkpoint

        You've completed Module 5 when you can:

        - [ ] Serialize and load models correctly
        - [ ] Build inference API with validation
        - [ ] Handle production edge cases
        - [ ] Monitor model in production
        - [ ] Detect and respond to data drift

        ---
        ## 🎓 Course Complete!

        **Congratulations!** You've completed all 5 modules of the
        Professional ML Engineering Onboarding Project.

        **What you've learned**:
        - ✅ Data engineering and validation
        - ✅ Feature engineering with domain knowledge
        - ✅ Systematic model training and experimentation
        - ✅ Comprehensive model evaluation
        - ✅ Production deployment and monitoring

        **You're now ready to**:
        - Join an ML team and contribute from day one
        - Build end-to-end ML systems
        - Debug production ML issues
        - Communicate with stakeholders effectively

        **Next steps**:
        - Complete the capstone project
        - Apply these skills to your own projects
        - Continue learning advanced topics

        **Welcome to the world of ML engineering!** 🚀
        """
    )
    return


if __name__ == "__main__":
    app.run()
