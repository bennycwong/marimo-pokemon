# Pokemon Type Classifier - Inference System

A complete production-ready ML inference system with trained model, REST API, and beautiful web UI.

## 🎯 What's Included

### 1. Trained Model (`models/pokemon_classifier_latest.joblib`)
- **Type**: Random Forest Classifier
- **Task**: Pokemon type classification (18 types)
- **Features**: 9 engineered features from base stats
- **Performance**: 46.78% test accuracy on 18-class problem
- **Training data**: 16,000 Pokemon cards

### 2. REST API Server (`inference_server.py`)
- **Framework**: FastAPI
- **Features**:
  - Single and batch predictions
  - Input validation with Pydantic
  - Model versioning and metadata
  - Health checks for monitoring
  - CORS enabled for browser requests
  - Comprehensive error handling
  - Structured logging

### 3. Web UI (`pokemon_predictor_ui.html`)
- **Design**: Beautiful Pokemon card-themed interface
- **Features**:
  - Interactive sliders for all 6 base stats
  - **Randomizer button** - Generate realistic Pokemon with 8 different archetypes (Tank, Physical Attacker, Special Attacker, Speedster, Balanced, Glass Cannon, Legendary, Defensive Wall)
  - Real-time prediction updates
  - Type-specific color coding (18 Pokemon types)
  - Confidence visualization
  - Top 3 predictions display
  - Responsive design
  - Loading states and error handling

---

## 🚀 Quick Start

### Step 1: Train the Model

```bash
uv run python train_model.py
```

This will:
- Load the Pokemon cards dataset
- Engineer features
- Train a Random Forest model
- Save to `models/pokemon_classifier_latest.joblib`
- Display accuracy metrics

**Output example**:
```
✅ MODEL TRAINING COMPLETE!

Model location: models/pokemon_classifier_latest.joblib
Feature count: 9
Classes: 18
Test accuracy: 46.78%
```

### Step 2: Start the Inference Server

```bash
uv run python inference_server.py
```

Or with uvicorn:
```bash
uv run uvicorn inference_server:app --host 127.0.0.1 --port 8001
```

Server will be available at:
- **API**: http://localhost:8001
- **Docs**: http://localhost:8001/docs
- **Health**: http://localhost:8001/health

### Step 3: Open the Web UI

Simply open `pokemon_predictor_ui.html` in your browser:

```bash
open pokemon_predictor_ui.html
```

The UI will automatically connect to the server on port 8001.

---

## 📡 API Reference

### 1. Single Prediction

**Endpoint**: `POST /predict`

**Request**:
```json
{
  "hp": 100,
  "attack": 120,
  "defense": 80,
  "sp_attack": 90,
  "sp_defense": 75,
  "speed": 85
}
```

**Response**:
```json
{
  "predicted_type": "Fire",
  "confidence": 0.53,
  "top_3_predictions": [
    {"type": "Fire", "confidence": 0.53},
    {"type": "Water", "confidence": 0.16},
    {"type": "Fairy", "confidence": 0.11}
  ],
  "model_version": "20251121_170637",
  "predicted_at": "2025-11-21T17:11:52.336040"
}
```

**Validation**:
- All stats must be integers
- Range: 1-300 for each stat
- All fields required

### 2. Batch Prediction

**Endpoint**: `POST /predict/batch`

**Request**:
```json
{
  "cards": [
    {
      "hp": 100,
      "attack": 120,
      "defense": 80,
      "sp_attack": 90,
      "sp_defense": 75,
      "speed": 85
    },
    {
      "hp": 150,
      "attack": 60,
      "defense": 120,
      "sp_attack": 70,
      "sp_defense": 100,
      "speed": 50
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "predicted_type": "Fire",
      "confidence": 0.53,
      "top_3_predictions": [...],
      "model_version": "20251121_170637",
      "predicted_at": "2025-11-21T17:11:52.336040"
    },
    ...
  ],
  "batch_size": 2,
  "model_version": "20251121_170637"
}
```

**Limits**:
- Minimum: 1 card
- Maximum: 100 cards per batch

### 3. Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "20251121_170637",
  "uptime_seconds": 123.45
}
```

### 4. Model Metadata

**Endpoint**: `GET /model/metadata`

**Response**:
```json
{
  "model_version": "20251121_170637",
  "feature_names": ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed", "total_stats", "attack_defense_ratio", "hp_per_stat"],
  "target_classes": ["Bug", "Dark", "Dragon", "Electric", "Fairy", "Fighting", "Fire", "Flying", "Ghost", "Grass", "Ground", "Ice", "Normal", "Poison", "Psychic", "Rock", "Steel", "Water"],
  "train_score": 0.9286,
  "test_score": 0.4678,
  "trained_at": "20251121_170637",
  "n_samples": 16000
}
```

---

## 🎨 Pokemon Type Colors

The UI uses authentic Pokemon type colors:

| Type | Color |
|------|-------|
| Normal | #A8A878 |
| Fire | #F08030 |
| Water | #6890F0 |
| Electric | #F8D030 |
| Grass | #78C850 |
| Ice | #98D8D8 |
| Fighting | #C03028 |
| Poison | #A040A0 |
| Ground | #E0C068 |
| Flying | #A890F0 |
| Psychic | #F85888 |
| Bug | #A8B820 |
| Rock | #B8A038 |
| Ghost | #705898 |
| Dragon | #7038F8 |
| Dark | #705848 |
| Steel | #B8B8D0 |
| Fairy | #EE99AC |

---

## 🧪 Testing

### Test with curl

```bash
# Single prediction
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"hp": 100, "attack": 120, "defense": 80, "sp_attack": 90, "sp_defense": 75, "speed": 85}'

# Health check
curl http://127.0.0.1:8001/health
```

### Test with Python script

```bash
# Run comprehensive tests
python test_inference_server.py
```

This tests:
- All API endpoints
- Edge cases (min/max stats, extreme builds)
- Batch predictions
- Input validation
- Error handling

---

## 📊 Model Details

### Features Used

**Base Stats** (6):
- HP (Hit Points)
- Attack
- Defense
- Special Attack
- Special Defense
- Speed

**Engineered Features** (3):
- `total_stats`: Sum of all base stats
- `attack_defense_ratio`: Attack / (Defense + 1)
- `hp_per_stat`: HP / (Total Stats + 1)

### Model Architecture

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    ))
])
```

### Performance

- **Train Accuracy**: 92.86%
- **Test Accuracy**: 46.78%
- **Classes**: 18 Pokemon types
- **Training Samples**: 12,800
- **Test Samples**: 3,200

The model shows some overfitting (train 93% vs test 47%), which is common for 18-class classification problems with stat-based features. In production, you might want to:
- Add more diverse features (moves, abilities, generation)
- Increase training data
- Use ensemble methods
- Fine-tune regularization

---

## 🔧 Production Considerations

### Current Implementation
✅ Input validation with Pydantic
✅ Health checks for monitoring
✅ Model versioning and metadata
✅ Error handling and logging
✅ CORS enabled for web access
✅ Batch prediction support

### For Production Deployment

1. **Security**:
   - Replace `allow_origins=["*"]` with specific domains
   - Add API key authentication
   - Use HTTPS/TLS
   - Rate limiting

2. **Scaling**:
   - Use multiple workers: `uvicorn app:app --workers 4`
   - Add load balancer (nginx, AWS ELB)
   - Deploy on container platform (Docker + Kubernetes)

3. **Monitoring**:
   - Prometheus metrics
   - Application Performance Monitoring (APM)
   - Log aggregation (ELK stack, Datadog)
   - Model performance tracking (MLflow, Weights & Biases)

4. **Model Updates**:
   - A/B testing framework
   - Canary deployments
   - Model registry (MLflow, BentoML)
   - Automated retraining pipeline

---

## 📁 File Structure

```
marimo-pokemon/
├── models/
│   ├── pokemon_classifier_20251121_170637.joblib   # Timestamped model
│   └── pokemon_classifier_latest.joblib             # Symlink to latest
│
├── train_model.py                                   # Model training script
├── inference_server.py                              # FastAPI inference server
├── pokemon_predictor_ui.html                        # Web UI
├── test_inference_server.py                         # API test suite
└── INFERENCE_README.md                             # This file
```

---

## 🐛 Troubleshooting

### Issue: Port 8001 already in use

```bash
# Find and kill process on port 8001
lsof -ti :8001 | xargs kill -9

# Or use a different port
uvicorn inference_server:app --port 8002
```

### Issue: Model not found

```bash
# Train the model first
uv run python train_model.py
```

### Issue: UI can't connect to server

1. Check server is running: `curl http://127.0.0.1:8001/health`
2. Check CORS is enabled (should see CORS headers in browser console)
3. Verify port in `pokemon_predictor_ui.html` matches server port

### Issue: Dependencies missing

```bash
# Reinstall dependencies
uv sync

# Or install specific packages
uv pip install fastapi uvicorn pydantic
```

---

## 🎲 Randomizer Feature

The UI includes a **Random** button that generates realistic Pokemon stats based on 8 different archetypes:

### Archetypes

1. **Tank** - High HP and defenses, low speed
   - Total stats: ~450-550
   - Example: Snorlax, Blissey

2. **Physical Attacker** - High attack, moderate speed
   - Total stats: ~480-580
   - Example: Machamp, Tyranitar

3. **Special Attacker** - High special attack, moderate speed
   - Total stats: ~480-570
   - Example: Alakazam, Gengar

4. **Speedster** - Very high speed, moderate offenses
   - Total stats: ~460-550
   - Example: Jolteon, Crobat

5. **Balanced** - All stats around 80-100
   - Total stats: ~480-600
   - Example: Mew, Celebi

6. **Glass Cannon** - Extreme offense, very low defense
   - Total stats: ~450-560
   - Example: Alakazam, Gengar (offensive builds)

7. **Legendary** - High stats all around
   - Total stats: ~600-720
   - Example: Mewtwo, Rayquaza, Arceus

8. **Defensive Wall** - Extreme defenses, low speed
   - Total stats: ~450-580
   - Example: Shuckle, Steelix

Each archetype generates stats within realistic ranges for actual Pokemon, ensuring the predictions are meaningful!

---

## 🎯 Usage Examples

### Example 1: Tank Pokemon (High HP, Low Speed)

**Input**:
- HP: 250, Attack: 50, Defense: 200, Sp. Attack: 50, Sp. Defense: 200, Speed: 30

**Likely Types**: Water, Normal, Rock (defensive types)

### Example 2: Glass Cannon (High Attack, Low Defense)

**Input**:
- HP: 60, Attack: 250, Defense: 40, Sp. Attack: 250, Sp. Defense: 40, Speed: 150

**Likely Types**: Dragon, Electric, Ghost (offensive types)

### Example 3: Balanced Fighter

**Input**:
- HP: 100, Attack: 120, Defense: 80, Sp. Attack: 90, Sp. Defense: 75, Speed: 85

**Likely Types**: Fire, Fighting, Normal (balanced types)

### Example 4: Speedy Attacker

**Input**:
- HP: 80, Attack: 90, Defense: 70, Sp. Attack: 100, Sp. Defense: 80, Speed: 200

**Likely Types**: Electric, Flying, Psychic (fast types)

---

## 📝 API Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

You can test all endpoints directly from the browser!

---

## 🎓 Learning Value

This inference system demonstrates:

1. **ML Engineering**: Complete pipeline from training to deployment
2. **API Design**: RESTful API with proper validation and error handling
3. **Web Development**: Modern, responsive UI with real-time updates
4. **Production Patterns**: Health checks, versioning, monitoring, CORS
5. **Documentation**: Comprehensive API docs and examples

Perfect for portfolio projects and understanding production ML systems!

---

## 🚀 Next Steps

To improve this system:

1. **Model Improvements**:
   - Add more features (type weaknesses, abilities)
   - Try other algorithms (XGBoost, Neural Networks)
   - Implement ensemble methods
   - Add regression for stats prediction

2. **API Enhancements**:
   - Add authentication (API keys, OAuth)
   - Implement rate limiting
   - Add caching layer (Redis)
   - Version the API (v1, v2)

3. **UI Features**:
   - Add preset Pokemon templates
   - Show feature importance
   - Display confusion matrix
   - Add comparison mode (multiple Pokemon)
   - Save/load configurations

4. **Deployment**:
   - Dockerize the application
   - Deploy to cloud (AWS, GCP, Azure)
   - Set up CI/CD pipeline
   - Add load testing

---

**Built with**: FastAPI, scikit-learn, Pydantic, Modern HTML/CSS/JS

**Model**: Random Forest Classifier (18 Pokemon types)

**Ready for production!** 🚀
