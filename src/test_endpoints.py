from fastapi.testclient import TestClient
from api.endpoints import app
import polars as pl
from ml.train import package_model, train_intent_classifier
from schema.schema import TrainingConfig
import base64
from unittest.mock import patch
import io

client = TestClient(app)

def get_default_training_config():
    """Helper function to create a default training config for tests"""
    return TrainingConfig(
        num_epochs=2,  # Reduced for faster testing
        batch_size=32,
        learning_rate=5e-5,
        model_name='distilbert-base-uncased',
        max_length=128,
        warmup_steps=0,
        weight_decay=0.01,
        early_stopping_patience=None
    )

def test_predict_endpoint():
    # First, create some test data and train a model
    intents = {
        "intent": ["greeting", "farewell", "help_request"],
        "text": ["hello there", "goodbye", "can you help me"]
    }
    df = pl.DataFrame(intents)
    
    # Train the model and get run_id
    model, intents_list, tokenizer = train_intent_classifier(df)
    run_id = package_model(model, intents_list, tokenizer)
    
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
    id = "99999"
    response = client.post(
        f"/models/{id}/predict",
        params={"text": "hello"}
    )
    
    # Assert we get a 404 error
    assert response.status_code == 400
    assert f"Model ID {id} not found" in response.json()["detail"]
    


def test_train_endpoint_with_url():
    # Create a mock CSV content
    mock_csv_content = """intent,text
greeting,hello there
greeting,hi how are you
greeting,good morning
greeting,hey friend
farewell,goodbye for now
farewell,see you later
farewell,bye bye
farewell,have a great day
help_request,can you assist me
help_request,i need help with something
help_request,could you help me out
help_request,having trouble with this"""

    # Mock the URL request to return our local CSV
    with patch('requests.get') as mock_get:
        # Configure the mock response
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.text = mock_csv_content.encode()
        mock_response.iter_lines = lambda: io.StringIO(mock_csv_content).readlines()
        
        # Create test request with URL
        request = {
            "name": "test_model",
            "description": "Test model",
            "intents": ["greeting", "farewell", "help_request"],
            "dataset_source": {
                "source_type": "url",
                "url": "https://example.com/dataset.csv"
            },
            "training_config": {
                "num_epochs": 2,  # Reduced for testing
                "batch_size": 16,
                "learning_rate": 1e-4,
                "model_name": "distilbert-base-uncased",
                "max_length": 64,
                "warmup_steps": 0,
                "weight_decay": 0.01,
                "early_stopping_patience": 2
            }
        }
        
        response = client.post("/train", json=request)
        assert response.status_code == 200
        assert "model_id" in response.json()
        assert response.json()["status"] == "success"
        
        # Verify the URL was "called"
        mock_get.assert_called_once_with("https://example.com/dataset.csv")

def test_train_endpoint_with_upload():
    # Create test request with file upload
    csv_content = """intent,text
greeting,hello there
greeting,hi how are you
greeting,good morning
greeting,hey friend
farewell,goodbye for now
farewell,see you later
farewell,bye bye
farewell,have a great day
help_request,can you assist me
help_request,i need help with something
help_request,could you help me out
help_request,having trouble with this"""
    
    encoded_content = base64.b64encode(csv_content.encode()).decode()
    
    request = {
        "name": "test_model",
        "description": "Test model",
        "intents": ["greeting", "farewell", "help_request"],
        "dataset_source": {
            "source_type": "upload",
            "file_content": encoded_content
        },
        "training_config": {
            "num_epochs": 1,  # Minimal for testing
            "batch_size": 8,
            "learning_rate": 2e-5,
            "model_name": "distilbert-base-uncased",
            "max_length": 32,
            "warmup_steps": 0,
            "weight_decay": 0.01,
            "early_stopping_patience": None
        }
    }
    
    response = client.post("/train", json=request)
    assert response.status_code == 200
    assert "model_id" in response.json()
    assert response.json()["status"] == "success"

if __name__ == "__main__":
    test_predict_endpoint()
    test_predict_invalid_model()
    test_train_endpoint_with_url()
    test_train_endpoint_with_upload()