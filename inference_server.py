"""
Production-ready FastAPI inference server for Pokemon type classification.

Features:
- Input validation with Pydantic
- Model versioning and metadata
- Health checks
- Error handling
- Logging
- Batch predictions
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pokemon Type Classifier API",
    description="Production ML inference service for predicting Pokemon card types",
    version="1.0.0",
)

# Add CORS middleware to allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODEL_CACHE = {}


class PokemonCardInput(BaseModel):
    """Input schema for Pokemon card with validation."""
    hp: int = Field(..., ge=1, le=300, description="Hit points (1-300)")
    attack: int = Field(..., ge=1, le=300, description="Attack stat (1-300)")
    defense: int = Field(..., ge=1, le=300, description="Defense stat (1-300)")
    sp_attack: int = Field(..., ge=1, le=300, description="Special attack stat (1-300)")
    sp_defense: int = Field(..., ge=1, le=300, description="Special defense stat (1-300)")
    speed: int = Field(..., ge=1, le=300, description="Speed stat (1-300)")

    @validator('*')
    def check_positive(cls, v):
        if v < 0:
            raise ValueError('All stats must be positive')
        return v

    class Config:
        schema_extra = {
            "example": {
                "hp": 100,
                "attack": 120,
                "defense": 80,
                "sp_attack": 90,
                "sp_defense": 75,
                "speed": 85,
            }
        }


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""
    cards: List[PokemonCardInput] = Field(..., min_items=1, max_items=100)


class TypePrediction(BaseModel):
    """Single type prediction with confidence."""
    type: str
    confidence: float


class PredictionOutput(BaseModel):
    """Output schema for single prediction."""
    predicted_type: str
    confidence: float
    top_3_predictions: List[TypePrediction]
    model_version: str
    predicted_at: str


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionOutput]
    batch_size: int
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float


class ModelMetadata(BaseModel):
    """Model metadata response."""
    model_version: str
    feature_names: List[str]
    target_classes: List[str]
    train_score: float
    test_score: float
    trained_at: str
    n_samples: int


# Server start time for uptime calculation
SERVER_START_TIME = datetime.now()


def load_model(model_path: str = "models/pokemon_classifier_latest.joblib") -> Dict[str, Any]:
    """Load model from disk with caching."""
    if model_path in MODEL_CACHE:
        logger.info(f"Using cached model: {model_path}")
        return MODEL_CACHE[model_path]

    path = Path(model_path)
    if not path.exists():
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")
    model_artifact = joblib.load(path)
    MODEL_CACHE[model_path] = model_artifact
    logger.info(f"Model loaded successfully: {model_artifact['trained_at']}")

    return model_artifact


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features to match training data."""
    df['total_stats'] = df['hp'] + df['attack'] + df['defense'] + df['sp_attack'] + df['sp_defense'] + df['speed']
    df['attack_defense_ratio'] = df['attack'] / (df['defense'] + 1)
    df['hp_per_stat'] = df['hp'] / (df['total_stats'] + 1)
    return df


def predict_single(card: PokemonCardInput, model_artifact: Dict[str, Any]) -> PredictionOutput:
    """Make prediction for a single Pokemon card."""
    try:
        # Convert to DataFrame
        card_dict = card.dict()
        df = pd.DataFrame([card_dict])

        # Engineer features
        df = engineer_features(df)

        # Ensure column order matches training
        df = df[model_artifact['feature_names']]

        # Make prediction
        pipeline = model_artifact['pipeline']
        prediction = pipeline.predict(df)[0]
        probabilities = pipeline.predict_proba(df)[0]

        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3 = [
            TypePrediction(type=pipeline.classes_[idx], confidence=float(probabilities[idx]))
            for idx in top_3_idx
        ]

        return PredictionOutput(
            predicted_type=prediction,
            confidence=float(max(probabilities)),
            top_3_predictions=top_3,
            model_version=model_artifact['trained_at'],
            predicted_at=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    try:
        load_model()
        logger.info("Server started successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Don't crash the server, let health check report the issue


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Pokemon Type Classifier API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "health": "/health",
            "metadata": "/model/metadata",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        model_artifact = load_model()
        model_loaded = True
        model_version = model_artifact['trained_at']
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        model_loaded = False
        model_version = None

    uptime = (datetime.now() - SERVER_START_TIME).total_seconds()

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
        uptime_seconds=uptime,
    )


@app.get("/model/metadata", response_model=ModelMetadata, tags=["Model"])
async def get_model_metadata():
    """Get model metadata and training information."""
    try:
        model_artifact = load_model()
        return ModelMetadata(
            model_version=model_artifact['trained_at'],
            feature_names=model_artifact['feature_names'],
            target_classes=model_artifact['target_classes'],
            train_score=model_artifact['train_score'],
            test_score=model_artifact['test_score'],
            trained_at=model_artifact['trained_at'],
            n_samples=model_artifact['n_samples'],
        )
    except Exception as e:
        logger.error(f"Failed to get model metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model metadata: {str(e)}"
        )


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict(card: PokemonCardInput):
    """
    Predict Pokemon type for a single card.

    Returns the predicted type, confidence score, and top 3 predictions.
    """
    try:
        model_artifact = load_model()
        return predict_single(card, model_artifact)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Please train a model first."
        )
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Predictions"])
async def predict_batch(batch: BatchPredictionInput):
    """
    Predict Pokemon types for multiple cards in batch.

    Maximum batch size: 100 cards.
    """
    try:
        model_artifact = load_model()
        predictions = [predict_single(card, model_artifact) for card in batch.cards]

        return BatchPredictionOutput(
            predictions=predictions,
            batch_size=len(predictions),
            model_version=model_artifact['trained_at'],
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Please train a model first."
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__,
        }
    )


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Starting Pokemon Type Classifier Inference Server")
    print("=" * 60)
    print("\nServer will be available at:")
    print("  - Local: http://localhost:8000")
    print("  - API Docs: http://localhost:8000/docs")
    print("  - Health: http://localhost:8000/health")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60)

    uvicorn.run(
        "inference_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
