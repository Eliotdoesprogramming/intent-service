import warnings
import tempfile
import polars as pl
from pydantic import warnings as pydantic_warnings
from fastapi.testclient import TestClient
from intent_service.api.endpoints import app
from intent_service.ml.train import train_intent_classifier
from pathlib import Path
warnings.filterwarnings("ignore", category=pydantic_warnings.PydanticDeprecatedSince20)

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
    

async def test_train_model_endpoint():
    # Create a temporary CSV file for testing
    test_data = {
        "intent": ["greeting", "farewell", "help_request"],
        "text": ["hello there", "goodbye", "can you help me"]
    }
    df = pl.DataFrame(test_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df.write_csv(tmp.name)
        tmp_path = Path(tmp.name)

        # Test file upload
        with open(tmp_path, 'rb') as f:
            files = {'file': ('test.csv', f, 'text/csv')}
            response = client.post("/train", files=files)
            
            assert response.status_code == 200
            assert isinstance(response.json(), str)  # Should return a run_id
            
        # Test URL-based training (mocked with TestClient)
        # First, create a route that serves our test file
        @app.get("/test-data")
        async def get_test_data():
            return df.to_csv()
            
        # Now use that route to test URL-based training
        response = client.post(
            "/train",
            params={"dataset_url": "http://testserver/test-data"}
        )
        
        assert response.status_code == 200
        assert isinstance(response.json(), str)  # Should return a run_id

        # Test invalid data
        invalid_data = {
            "wrong_column": ["greeting"],
            "text": ["hello there"]
        }
        invalid_df = pl.DataFrame(invalid_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            invalid_df.write_csv(tmp.name)
            with open(tmp.name, 'rb') as f:
                files = {'file': ('invalid.csv', f, 'text/csv')}
                response = client.post("/train", files=files)
                
                assert response.status_code == 400
                assert "Dataset must contain columns" in response.json()["detail"]

        # Test missing both URL and file
        response = client.post("/train")
        assert response.status_code == 400
        assert "Either dataset_url or file must be provided" in response.json()["detail"]

        # Clean up
        tmp_path.unlink()

def test_invalid_url():
    # Test with invalid URL
    response = client.post(
        "/train",
        params={"dataset_url": "http://invalid-url-that-does-not-exist.com/data.csv"}
    )
    assert response.status_code == 500

if __name__ == "__main__":
    test_predict_endpoint()
    test_predict_invalid_model()
    test_train_model_endpoint()
    test_invalid_url()