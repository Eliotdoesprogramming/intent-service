import asyncio
import base64
import io
import json
import logging
import time
from unittest.mock import patch

import polars as pl
import pytest
from fastapi.testclient import TestClient

from api.endpoints import app
from ml.mlflow import package_model
from ml.train import train_intent_classifier
from schema.schema import TrainingConfig

logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create a test client that can handle streaming responses"""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def default_training_config():
    """Fixture for default training configuration"""
    config = TrainingConfig(
        num_epochs=2,  # Reduced for faster testing
        batch_size=32,
        learning_rate=5e-5,
        base_model_name="distilbert-base-uncased",
        max_length=128,
        warmup_steps=0,
        weight_decay=0.01,
        early_stopping_patience=None,
    )
    return config  # Return the config object directly


@pytest.fixture
def sample_training_data():
    """Fixture for sample training data"""
    intents = {
        "intent": [
            "greeting",
            "greeting",
            "greeting",
            "greeting",
            "greeting",
            "farewell",
            "farewell",
            "farewell",
            "farewell",
            "farewell",
            "help_request",
            "help_request",
            "help_request",
            "help_request",
            "help_request",
        ],
        "text": [
            "hello there",
            "hi",
            "hey",
            "good morning",
            "greetings",
            "goodbye",
            "bye",
            "see you later",
            "farewell",
            "take care",
            "can you help me",
            "i need assistance",
            "help please",
            "could you assist me",
            "need some help",
        ],
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
        "tokenizer": tokenizer,
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


@pytest.mark.short
def test_predict_endpoint(client, trained_model):
    run_id = trained_model["run_id"]
    test_cases = [
        {"text": "hi there", "expected_intents": ["greeting"]},
        {"text": "bye bye", "expected_intents": ["farewell"]},
        {"text": "I need assistance", "expected_intents": ["help_request"]},
    ]

    for test_case in test_cases:
        start_time = time.time()
        response = client.post(
            f"/model/{run_id}/predict", params={"text": test_case["text"]}
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


@pytest.mark.short
def test_predict_invalid_model(client):
    id = "99999"
    response = client.post(f"/model/{id}/predict", params={"text": "hello"})
    assert response.status_code == 404


@pytest.mark.long
def test_train_endpoint_with_url(client, mock_csv_content, default_training_config):
    with patch("requests.get") as mock_get:
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
                "url": "https://example.com/dataset.csv",
            },
            "training_config": default_training_config.model_dump(),
        }

        response = client.post("/model/train", json=request)
        assert response.status_code == 200
        assert "model_id" in response.json()
        assert response.json()["status"] == "success"
        mock_get.assert_called_once_with("https://example.com/dataset.csv")


@pytest.mark.long
def test_train_endpoint_with_upload(client, mock_csv_content, default_training_config):
    encoded_content = base64.b64encode(mock_csv_content.encode()).decode()
    request = {
        "name": "test_model",
        "description": "Test model",
        "intents": ["greeting", "farewell", "help_request"],
        "dataset_source": {"source_type": "upload", "file_content": encoded_content},
        "training_config": default_training_config.model_dump(),
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
            "description": "Test dataset",
        },
        "extra_data": {"test_accuracy": 0.95, "created_date": "2024-03-21"},
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


def test_get_model_info(client, trained_model):
    """Test the get_model_info endpoint with both registered and unregistered models"""

    # First register a model to test registered model path
    run_id = trained_model["run_id"]
    model_request = {
        "mlflow_run_id": run_id,
        "name": "test_model_info",
        "description": "Test model for info endpoint",
        "tags": {"test_tag": "test_value", "environment": "testing"},
    }

    # Register the model first
    register_response = client.post("/model/register", json=model_request)
    assert register_response.status_code == 200

    # Test getting info for registered model
    response = client.get("/model/test_model_info")
    assert response.status_code == 200
    model_info = response.json()

    # Verify registered model information
    assert model_info["name"] == "test_model_info"
    assert model_info["description"] == "Test model for info endpoint"
    assert "version" in model_info
    assert "creation_timestamp" in model_info
    assert "last_updated_timestamp" in model_info
    assert "intents" in model_info
    assert isinstance(model_info["intents"], list)
    assert model_info["tags"]["test_tag"] == "test_value"
    assert model_info["tags"]["environment"] == "testing"
    assert "run_info" in model_info

    # Test getting info using run ID directly
    response = client.get(f"/model/{run_id}")
    assert response.status_code == 200
    run_info = response.json()

    # Verify run information
    assert run_info["run_id"] == run_id
    assert "status" in run_info
    assert "metrics" in run_info
    assert "params" in run_info
    assert "intents" in run_info
    assert isinstance(run_info["intents"], list)

    # Test invalid model ID
    invalid_id = "nonexistent_model_12345"
    response = client.get(f"/model/{invalid_id}")
    assert response.status_code == 404
    assert f"No model found with ID: {invalid_id}" in response.json()["detail"]


def test_search_models(client, trained_model):
    """Test the model search endpoint with various search criteria"""

    # First register a few models with different tags and intents
    run_id = trained_model["run_id"]

    # Register first model
    model1_request = {
        "mlflow_run_id": run_id,
        "name": "prod_model_v1",
        "description": "Production model version 1",
        "tags": {"environment": "production", "version": "1.0", "team": "nlp"},
    }

    # Register second model
    model2_request = {
        "mlflow_run_id": run_id,
        "name": "test_bert_v1",
        "description": "Test BERT model",
        "tags": {"environment": "testing", "version": "1.0", "model_type": "bert"},
    }

    # Register both models
    response1 = client.post("/model/register", json=model1_request)
    assert response1.status_code == 200
    response2 = client.post("/model/register", json=model2_request)
    assert response2.status_code == 200

    # Test 1: Search by tag
    search_request = {"tags": {"environment": "production"}}
    response = client.post("/model/search", json=search_request)
    assert response.status_code == 200
    results = response.json()
    assert len(results) >= 1
    assert any(model["name"] == "prod_model_v1" for model in results)
    assert all(model["tags"].get("environment") == "production" for model in results)

    # Test 2: Search by name pattern
    search_request = {"name_contains": "bert"}
    response = client.post("/model/search", json=search_request)
    assert response.status_code == 200
    results = response.json()
    assert len(results) >= 1
    assert any(model["name"] == "test_bert_v1" for model in results)

    # Test 3: Search by intents
    search_request = {"intents": ["greeting", "farewell"]}
    response = client.post("/model/search", json=search_request)
    assert response.status_code == 200
    results = response.json()
    assert len(results) >= 1
    assert all(
        set(["greeting", "farewell"]).issubset(set(model["intents"]))
        for model in results
    )

    # Test 4: Combined search criteria
    search_request = {
        "tags": {"environment": "testing"},
        "name_contains": "bert",
        "intents": ["greeting"],
        "limit": 1,
    }
    response = client.post("/model/search", json=search_request)
    assert response.status_code == 200
    results = response.json()
    assert len(results) <= 1  # Respects the limit
    if results:
        assert results[0]["name"] == "test_bert_v1"
        assert results[0]["tags"]["environment"] == "testing"
        assert "greeting" in results[0]["intents"]

    # Test 5: Search with no results
    search_request = {"tags": {"environment": "nonexistent"}}
    response = client.post("/model/search", json=search_request)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 0

    # Test 6: Invalid search request (should still return 200 with empty results)
    search_request = {"tags": None, "intents": None, "name_contains": None}
    response = client.post("/model/search", json=search_request)
    assert response.status_code == 200


def test_get_models(client, trained_model):
    """Test the get_models endpoint for listing registered models"""

    # First register a model to ensure there's something to list
    run_id = trained_model["run_id"]
    model_request = {
        "mlflow_run_id": run_id,
        "name": "test_list_model",
        "description": "Test model for listing endpoint",
        "tags": {"environment": "testing", "version": "1.0"},
    }

    # Register the model
    register_response = client.post("/model/register", json=model_request)
    assert register_response.status_code == 200

    # Test getting all models
    response = client.get("/model")
    assert response.status_code == 200
    models = response.json()

    # Verify response structure
    assert isinstance(models, list)
    assert len(models) >= 1  # Should have at least our registered model

    # Find our test model in the results
    test_model = next((m for m in models if m["name"] == "test_list_model"), None)
    assert test_model is not None

    # Verify model information structure
    assert "name" in test_model
    assert "description" in test_model
    assert "version" in test_model
    assert "creation_timestamp" in test_model
    assert "last_updated_timestamp" in test_model
    assert "intents" in test_model
    assert "tags" in test_model
    assert isinstance(test_model["intents"], list)
    assert isinstance(test_model["tags"], dict)

    # Verify our test model's specific values
    assert test_model["name"] == "test_list_model"
    assert test_model["description"] == "Test model for listing endpoint"
    assert test_model["tags"]["environment"] == "testing"
    assert test_model["tags"]["version"] == "1.0"


@pytest.mark.asyncio
async def test_train_endpoint_streaming(
    client, mock_csv_content, default_training_config
):
    """Test the streaming training endpoint with normal flow"""
    encoded_content = base64.b64encode(mock_csv_content.encode()).decode()
    request = {
        "name": "test_model",
        "description": "Test model",
        "intents": ["greeting", "farewell", "help_request"],
        "dataset_source": {"source_type": "upload", "file_content": encoded_content},
        "training_config": default_training_config.model_dump(),
    }

    with client.stream("POST", "/model/train/stream", json=request) as response:
        assert response.status_code == 200

        # Track the events we expect to see
        seen_events = {
            "initializing": False,
            "training": False,
            "packaging": False,
            "complete": False,
        }
        run_id = None

        for line in response.iter_lines():
            if line:
                # SSE format is "data: {json}"
                line_text = line.decode("utf-8") if isinstance(line, bytes) else line
                assert line_text.startswith("data: ")
                data = json.loads(line_text.split("data: ")[1])

                assert "status" in data
                assert "message" in data

                if data["status"] == "initializing":
                    seen_events["initializing"] = True
                elif data["status"] == "training":
                    seen_events["training"] = True
                    if "current_epoch" in data:
                        assert isinstance(data["current_epoch"], int)
                    if "epoch_progress" in data:
                        assert 0 <= data["epoch_progress"] <= 1
                elif data["status"] == "packaging":
                    seen_events["packaging"] = True
                elif data["status"] == "complete":
                    seen_events["complete"] = True
                    assert "run_id" in data
                    run_id = data["run_id"]
                elif data["status"] == "error":
                    pytest.fail(f"Received error: {data['message']}")

        # Verify we saw all expected events
        assert all(seen_events.values()), "Did not receive all expected event types"
        assert run_id is not None, "Did not receive run_id in complete event"


@pytest.mark.asyncio
async def test_train_endpoint_streaming_invalid_data(client, default_training_config):
    """Test the streaming endpoint with invalid data"""
    # Test with missing file content
    request = {
        "name": "test_model",
        "description": "Test model",
        "dataset_source": {"source_type": "upload", "file_content": ""},
        "training_config": default_training_config.model_dump(),
    }

    with client.stream("POST", "/model/train/stream", json=request) as response:
        assert response.status_code == 200

        for line in response.iter_lines():
            if line:
                line_text = line.decode("utf-8") if isinstance(line, bytes) else line
                data = json.loads(line_text.split("data: ")[1])
                if data["status"] == "error":
                    assert "File content must be provided" in data["message"]
                    break
        else:
            pytest.fail("Did not receive error message")


@pytest.mark.asyncio
async def test_train_endpoint_streaming_client_disconnect(
    client, mock_csv_content, default_training_config
):
    """Test that training stops when client disconnects"""
    encoded_content = base64.b64encode(mock_csv_content.encode()).decode()
    request = {
        "name": "test_model",
        "description": "Test model",
        "dataset_source": {"source_type": "upload", "file_content": encoded_content},
        "training_config": default_training_config.model_dump(),
    }

    training_started = False

    with client.stream("POST", "/model/train/stream", json=request) as response:
        assert response.status_code == 200

        for line in response.iter_lines():
            if line:
                line_text = line.decode("utf-8") if isinstance(line, bytes) else line
                data = json.loads(line_text.split("data: ")[1])
                if data["status"] == "training":
                    training_started = True
                    break  # Disconnect after training starts

    assert training_started, "Training never started"

    # Give a moment for cleanup
    await asyncio.sleep(1)

    # Verify the process was cleaned up by starting a new training
    with client.stream("POST", "/model/train/stream", json=request) as response:
        assert response.status_code == 200
        # Should be able to start a new training session


@pytest.mark.asyncio
async def test_train_endpoint_streaming_invalid_intents(
    client, mock_csv_content, default_training_config
):
    """Test streaming endpoint with intents that don't match the dataset"""
    encoded_content = base64.b64encode(mock_csv_content.encode()).decode()
    request = {
        "name": "test_model",
        "description": "Test model",
        "intents": ["invalid_intent"],  # Intent not in dataset
        "dataset_source": {"source_type": "upload", "file_content": encoded_content},
        "training_config": default_training_config.model_dump(),
    }

    with client.stream("POST", "/model/train/stream", json=request) as response:
        assert response.status_code == 200

        for line in response.iter_lines():
            if line:
                line_text = line.decode("utf-8") if isinstance(line, bytes) else line
                data = json.loads(line_text.split("data: ")[1])
                if data["status"] == "error":
                    break
        else:
            pytest.fail("Did not receive error message")
