import pytest
from fastapi.testclient import TestClient
from api.endpoints import app
import polars as pl
from ml.train import package_model, train_intent_classifier
from schema.schema import TrainingConfig
import base64
from unittest.mock import patch
import io
import logging
import time

logger = logging.getLogger(__name__)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def default_training_config():
    """Fixture for default training configuration"""
    config = TrainingConfig(
        num_epochs=2,  # Reduced for faster testing
        batch_size=32,
        learning_rate=5e-5,
        base_model_name='distilbert-base-uncased',
        max_length=128,
        warmup_steps=0,
        weight_decay=0.01,
        early_stopping_patience=None
    )
    return config  # Return the config object directly

@pytest.fixture
def sample_training_data():
    """Fixture for sample training data"""
    intents = {
        "intent": [
            "greeting", "greeting", "greeting", "greeting", "greeting",
            "farewell", "farewell", "farewell", "farewell", "farewell", 
            "help_request", "help_request", "help_request", "help_request", "help_request"
        ],
        "text": [
            "hello there", "hi", "hey", "good morning", "greetings",
            "goodbye", "bye", "see you later", "farewell", "take care",
            "can you help me", "i need assistance", "help please", "could you assist me", "need some help"
        ]
    }
    return pl.DataFrame(intents)

@pytest.fixture
def trained_model(sample_training_data):
    """Fixture for trained model and its components"""
    model, intents_list, tokenizer = train_intent_classifier(sample_training_data)
    run_id = package_model(model, intents_list, tokenizer)
    return {
        "run_id": run_id,
        "model": model,
        "intents_list": intents_list,
        "tokenizer": tokenizer
    }

@pytest.fixture
def mock_csv_content():
    """Fixture for mock CSV content"""
    return """intent,text
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

def test_predict_endpoint(client, trained_model):
    run_id = trained_model["run_id"]
    test_cases = [
        {
            "text": "hi there",
            "expected_intents": ["greeting"]
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
    
    for test_case in test_cases:
        start_time = time.time()
        response = client.post(
            f"/model/{run_id}/predict",
            params={"text": test_case["text"]}
        )
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Predict endpoint execution time: {execution_time:.4f} seconds")
        
        assert response.status_code == 200
        prediction = response.json()
        assert isinstance(prediction, dict)
        
        top_intent = max(prediction.items(), key=lambda x: x[1])[0]
        assert top_intent in test_case["expected_intents"]
        assert all(0 <= score <= 1 for score in prediction.values())

def test_predict_invalid_model(client):
    id = "99999"
    response = client.post(
        f"/model/{id}/predict",
        params={"text": "hello"}
    )
    assert response.status_code == 400
    assert f"Model ID {id} not found" in response.json()["detail"]

def test_train_endpoint_with_url(client, mock_csv_content,default_training_config):
    with patch('requests.get') as mock_get:
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.text = mock_csv_content.encode()
        mock_response.iter_lines = lambda: io.StringIO(mock_csv_content).readlines()
        
        request = {
            "name": "test_model",
            "description": "Test model",
            "intents": ["greeting", "farewell", "help_request"],
            "dataset_source": {
                "source_type": "url",
                "url": "https://example.com/dataset.csv"
            },
            "training_config": default_training_config.dict()
        }
        
        response = client.post("/model/train", json=request)
        assert response.status_code == 200
        assert "model_id" in response.json()
        assert response.json()["status"] == "success"
        mock_get.assert_called_once_with("https://example.com/dataset.csv")

def test_train_endpoint_with_upload(client, mock_csv_content, default_training_config):
    encoded_content = base64.b64encode(mock_csv_content.encode()).decode()
    print(default_training_config.model_dump())
    request = {
        "name": "test_model",
        "description": "Test model",
        "intents": ["greeting", "farewell", "help_request"],
        "dataset_source": {
            "source_type": "upload",
            "file_content": encoded_content
        },
        "training_config": default_training_config.dict()
    }
    
    response = client.post("/model/train", json=request)
    assert response.status_code == 200
    assert "model_id" in response.json()
    assert response.json()["status"] == "success"

def test_create_model(client, trained_model):
    run_id = trained_model["run_id"]
    model_request = {
        "mlflow_run_id": run_id,
        "name": "test_intent_model",
        "description": "Test intent classification model",
        "intents": ["greeting", "farewell"],
        "dataset": {
            "id": 1,
            "collection_name": "test_collection",
            "description": "Test dataset"
        },
        "extra_data": {
            "test_accuracy": 0.95,
            "created_date": "2024-03-21"
        }
    }
    
    # Test successful model registration
    response = client.post("/model/register", json=model_request)
    model_version = response.json()["version"]
    assert response.status_code == 200
    result = response.json()
    assert result["name"] == "test_intent_model"
    assert result["status"] == "success"
    
    # Test registering non-existent model
    invalid_request = model_request.copy()
    invalid_request["mlflow_run_id"] = "99999"
    response = client.post("/model/register", json=invalid_request)
    assert response.status_code == 404
    assert "No model found with run ID" in response.json()["detail"]
    
    # Test registering with duplicate name
    # should create a new version of the model
    duplicate_request = model_request.copy()
    duplicate_request["id"] = run_id
    response = client.post("/model/register", json=duplicate_request)
    assert response.status_code == 200
    assert response.json()["version"] != model_version