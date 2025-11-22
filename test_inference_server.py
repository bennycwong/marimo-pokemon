"""
Test script for the inference server.
Demonstrates all endpoints with sample requests.
"""
import requests
import json
from typing import Dict, Any

# Server configuration
BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_response(response: requests.Response):
    """Pretty print API response."""
    print(f"\nStatus Code: {response.status_code}")
    if response.status_code == 200:
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")


def test_root():
    """Test root endpoint."""
    print_section("1. Testing Root Endpoint")
    response = requests.get(f"{BASE_URL}/")
    print_response(response)


def test_health_check():
    """Test health check endpoint."""
    print_section("2. Testing Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print_response(response)


def test_model_metadata():
    """Test model metadata endpoint."""
    print_section("3. Testing Model Metadata")
    response = requests.get(f"{BASE_URL}/model/metadata")
    print_response(response)


def test_single_prediction():
    """Test single prediction endpoint."""
    print_section("4. Testing Single Prediction")

    # Sample: Strong attacking Pokemon (likely Fire or Fighting)
    pokemon_card = {
        "hp": 100,
        "attack": 120,
        "defense": 80,
        "sp_attack": 90,
        "sp_defense": 75,
        "speed": 85
    }

    print(f"\nInput Pokemon: {pokemon_card}")
    response = requests.post(f"{BASE_URL}/predict", json=pokemon_card)
    print_response(response)


def test_edge_cases():
    """Test edge case predictions."""
    print_section("5. Testing Edge Cases")

    edge_cases = [
        {
            "name": "Minimum stats",
            "data": {
                "hp": 1,
                "attack": 1,
                "defense": 1,
                "sp_attack": 1,
                "sp_defense": 1,
                "speed": 1
            }
        },
        {
            "name": "Maximum stats",
            "data": {
                "hp": 300,
                "attack": 300,
                "defense": 300,
                "sp_attack": 300,
                "sp_defense": 300,
                "speed": 300
            }
        },
        {
            "name": "High HP, low attack (Tank - likely Water or Normal)",
            "data": {
                "hp": 250,
                "attack": 50,
                "defense": 200,
                "sp_attack": 50,
                "sp_defense": 200,
                "speed": 30
            }
        },
        {
            "name": "Glass cannon (High attack, low defense - likely Dragon or Electric)",
            "data": {
                "hp": 60,
                "attack": 250,
                "defense": 40,
                "sp_attack": 250,
                "sp_defense": 40,
                "speed": 150
            }
        },
        {
            "name": "Speedy (High speed - likely Electric or Flying)",
            "data": {
                "hp": 80,
                "attack": 90,
                "defense": 70,
                "sp_attack": 100,
                "sp_defense": 80,
                "speed": 200
            }
        }
    ]

    for i, case in enumerate(edge_cases, 1):
        print(f"\n--- Edge Case {i}: {case['name']} ---")
        print(f"Input: {case['data']}")
        response = requests.post(f"{BASE_URL}/predict", json=case['data'])
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Type: {result['predicted_type']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("Top 3:")
            for pred in result['top_3_predictions']:
                print(f"  - {pred['type']}: {pred['confidence']:.2%}")
        else:
            print(f"Error: {response.text}")


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print_section("6. Testing Batch Prediction")

    batch_cards = {
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
            },
            {
                "hp": 70,
                "attack": 200,
                "defense": 50,
                "sp_attack": 180,
                "sp_defense": 60,
                "speed": 150
            }
        ]
    }

    print(f"\nSending batch of {len(batch_cards['cards'])} cards...")
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_cards)

    if response.status_code == 200:
        result = response.json()
        print(f"\nBatch Size: {result['batch_size']}")
        print(f"Model Version: {result['model_version']}")
        print("\nPredictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"\n  Card {i}:")
            print(f"    Type: {pred['predicted_type']}")
            print(f"    Confidence: {pred['confidence']:.2%}")
    else:
        print_response(response)


def test_invalid_input():
    """Test validation with invalid input."""
    print_section("7. Testing Input Validation (Expected Errors)")

    invalid_cases = [
        {
            "name": "Negative stats",
            "data": {
                "hp": -10,
                "attack": 100,
                "defense": 80,
                "sp_attack": 90,
                "sp_defense": 75,
                "speed": 85
            }
        },
        {
            "name": "Stats too high",
            "data": {
                "hp": 500,
                "attack": 100,
                "defense": 80,
                "sp_attack": 90,
                "sp_defense": 75,
                "speed": 85
            }
        },
        {
            "name": "Missing field",
            "data": {
                "hp": 100,
                "attack": 100,
                "defense": 80,
                # Missing sp_attack, sp_defense, speed
            }
        }
    ]

    for i, case in enumerate(invalid_cases, 1):
        print(f"\n--- Invalid Case {i}: {case['name']} ---")
        print(f"Input: {case['data']}")
        response = requests.post(f"{BASE_URL}/predict", json=case['data'])
        print(f"Status Code: {response.status_code} (Expected: 422)")
        if response.status_code != 200:
            error_detail = response.json().get('detail', 'Unknown error')
            print(f"Validation Error: {error_detail[0] if isinstance(error_detail, list) else error_detail}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  POKEMON TYPE CLASSIFIER - INFERENCE SERVER TESTS")
    print("=" * 60)
    print("\nMake sure the server is running:")
    print("  uv run python inference_server.py")
    print("\nOr in another terminal:")
    print("  uvicorn inference_server:app --reload")
    print("\nPress Enter to start tests...")
    input()

    try:
        test_root()
        test_health_check()
        test_model_metadata()
        test_single_prediction()
        test_edge_cases()
        test_batch_prediction()
        test_invalid_input()

        print("\n" + "=" * 60)
        print("  ✅ ALL TESTS COMPLETED")
        print("=" * 60)
        print("\nAPI Documentation available at:")
        print("  http://localhost:8000/docs")
        print("\nYou can also test interactively at:")
        print("  http://localhost:8000/docs#/")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server")
        print("\nMake sure the server is running:")
        print("  uv run python inference_server.py")
        print("\nOr:")
        print("  uvicorn inference_server:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    main()
