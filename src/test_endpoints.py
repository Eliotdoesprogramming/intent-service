from fastapi.testclient import TestClient
from api.endpoints import app
import mlflow
import polars as pl
from ml.train import train_intent_classifier  # Import your training function

client = TestClient(app)

def test_predict_endpoint():
    # First, create some test data and train a model
    intents = {
        "intent": ["greeting", "farewell", "help_request"],
        "text": ["hello there", "goodbye", "can you help me"]
    }
    df = pl.DataFrame(intents)
    
    # Train the model and get run_id
    run_id = train_intent_classifier(df)
    
    # Test cases
    test_cases = [
        {
            "text": "hi there",
            "expected_intents": ["greeting"]  # We expect greeting to have highest confidence
        },
        {
            "text": "bye bye",
            "expected_intents": ["farewell"]
        },
        {
            "text": "I need assistance",
            "expected_intents": ["help_request"]
        }
    ]
    
    # Test each case
    for test_case in test_cases:
        response = client.post(
            f"/models/{run_id}/predict",
            params={"text": test_case["text"]}
        )
        
        # Assert response is successful
        assert response.status_code == 200
        
        # Get prediction results
        prediction = response.json()
        
        # Assert we got a dictionary of predictions
        assert isinstance(prediction, dict)
        
        # Get the intent with highest confidence
        top_intent = max(prediction.items(), key=lambda x: x[1])[0]
        
        # Assert the top intent is in our expected intents
        assert top_intent in test_case["expected_intents"]
        
        # Assert all confidence scores are between 0 and 1
        assert all(0 <= score <= 1 for score in prediction.values())

def test_predict_invalid_model():
    # Test with non-existent model ID
    response = client.post(
        "/models/99999/predict",
        params={"text": "hello"}
    )
    
    # Assert we get a 404 error
    assert response.status_code == 404
    assert "Model not found" in response.json()["detail"]
    

if __name__ == "__main__":
    test_predict_endpoint()
    test_predict_invalid_model()