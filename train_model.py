"""
Train a Pokemon type classifier and save it for production use.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from datetime import datetime

print("Training Pokemon Type Classifier...")
print("=" * 60)

# 1. Load data
print("\n1. Loading dataset...")
DATA_PATH = Path("data/pokemon_cards.csv")
if not DATA_PATH.exists():
    print(f"❌ Dataset not found at {DATA_PATH}")
    print("Run: uv run python data/generate_dataset.py")
    exit(1)

df = pd.read_csv(DATA_PATH)
print(f"   ✅ Loaded {len(df)} Pokemon cards")

# 2. Prepare features and target
print("\n2. Preparing features...")
feature_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
X = df[feature_cols].fillna(0)
y = df['type']

# Add some engineered features
X['total_stats'] = X['hp'] + X['attack'] + X['defense'] + X['sp_attack'] + X['sp_defense'] + X['speed']
X['attack_defense_ratio'] = X['attack'] / (X['defense'] + 1)  # Avoid division by zero
X['hp_per_stat'] = X['hp'] / (X['total_stats'] + 1)

print(f"   Features: {list(X.columns)}")
print(f"   Target classes: {y.nunique()} types")
print(f"   Class distribution (top 5):\n{y.value_counts().head()}")

# 3. Split data
print("\n3. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

# 4. Create pipeline with scaling and model
print("\n4. Training model...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)
print("   ✅ Model trained")

# 5. Evaluate
print("\n5. Evaluating model...")
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"   Train accuracy: {train_score:.2%}")
print(f"   Test accuracy: {test_score:.2%}")

# 6. Save model
print("\n6. Saving model...")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = models_dir / f"pokemon_classifier_{timestamp}.joblib"
latest_path = models_dir / "pokemon_classifier_latest.joblib"

# Save with metadata
model_artifact = {
    'pipeline': pipeline,
    'feature_names': list(X.columns),
    'target_classes': sorted(y.unique()),
    'train_score': train_score,
    'test_score': test_score,
    'trained_at': timestamp,
    'n_samples': len(df),
}

joblib.dump(model_artifact, model_path)
joblib.dump(model_artifact, latest_path)

print(f"   ✅ Model saved to: {model_path}")
print(f"   ✅ Latest model: {latest_path}")

# 7. Test prediction
print("\n7. Testing sample prediction...")
sample_pokemon = {
    'hp': 100,
    'attack': 120,
    'defense': 80,
    'sp_attack': 90,
    'sp_defense': 75,
    'speed': 85,
}

# Engineer features for sample
sample_df = pd.DataFrame([sample_pokemon])
sample_df['total_stats'] = sample_df['hp'] + sample_df['attack'] + sample_df['defense'] + sample_df['sp_attack'] + sample_df['sp_defense'] + sample_df['speed']
sample_df['attack_defense_ratio'] = sample_df['attack'] / (sample_df['defense'] + 1)
sample_df['hp_per_stat'] = sample_df['hp'] / (sample_df['total_stats'] + 1)

prediction = pipeline.predict(sample_df)[0]
probabilities = pipeline.predict_proba(sample_df)[0]
top_3_idx = np.argsort(probabilities)[-3:][::-1]
top_3_classes = [pipeline.classes_[i] for i in top_3_idx]
top_3_probs = [probabilities[i] for i in top_3_idx]

print(f"   Sample Pokemon stats: {sample_pokemon}")
print(f"   Predicted type: {prediction}")
print(f"   Top 3 predictions:")
for cls, prob in zip(top_3_classes, top_3_probs):
    print(f"     - {cls}: {prob:.2%}")

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print(f"\nModel location: {latest_path}")
print(f"Feature count: {len(X.columns)}")
print(f"Classes: {len(pipeline.classes_)}")
print(f"Test accuracy: {test_score:.2%}")
print("\nReady for inference! Start server with:")
print("  uv run python inference_server.py")
print("=" * 60)
