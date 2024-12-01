import mlflow
import pandas as pd
import polars as pl
import pytest
from unittest.mock import patch, MagicMock

from ml import train_intent_classifier
from ml.mlflow import log_dataset, package_model
from ml.huggingface import upload_model_to_hub


@pytest.fixture
def intents_data():
    """Fixture providing sample intent classification data"""
    intents = {
        "intent": [
            "greeting",
            "greeting",
            "greeting",
            "greeting",
            "greeting",
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
            "help_request",
            "help_request",
            "help_request",
            "help_request",
            "help_request",
            "product_inquiry",
            "product_inquiry",
            "product_inquiry",
            "product_inquiry",
            "product_inquiry",
            "product_inquiry",
            "product_inquiry",
            "product_inquiry",
            "product_inquiry",
            "product_inquiry",
            "complaint",
            "complaint",
            "complaint",
            "complaint",
            "complaint",
            "complaint",
            "complaint",
            "complaint",
            "complaint",
            "complaint",
        ],
        "text": [
            "hello there",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "howdy",
            "hi there",
            "greetings",
            "hello",
            "goodbye",
            "bye",
            "see you later",
            "have a nice day",
            "take care",
            "bye bye",
            "farewell",
            "until next time",
            "good night",
            "thanks bye",
            "i need help",
            "can you assist me",
            "having trouble with",
            "support needed",
            "help please",
            "can someone help",
            "assistance required",
            "need support",
            "got a problem",
            "help me out",
            "how much does it cost",
            "what's the price",
            "tell me about this product",
            "product details please",
            "is this item available",
            "do you have this in stock",
            "what are the features",
            "product specifications",
            "when will it ship",
            "what colors are available",
            "this isn't working",
            "i'm having issues with",
            "not satisfied",
            "product is defective",
            "poor service",
            "this is unacceptable",
            "want to file a complaint",
            "disappointed with",
            "doesn't work as advertised",
            "broken on arrival",
        ],
    }
    return pl.DataFrame(intents)


def test_intent_classifier(intents_data):
    """Test the intent classifier training and prediction pipeline"""
    # Train the model and get the run_id
    model, intents_list, tokenizer = train_intent_classifier(intents_data)
    run_id = package_model(model, intents_list, tokenizer)
    print(f"Model trained successfully. Run ID: {run_id}")

    # Load the model from MLflow
    loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/intent_model")

    # Test some example phrases - using pandas DataFrame
    test_data = pd.DataFrame({
        "text": [
            "hi there, how are you?",
            "what is the cost of shipping?",
            "my order arrived damaged",
            "bye for now",
            "can you help me find something?",
        ]
    })

    # Get predictions
    predictions = loaded_model.predict(test_data)

    # Print results
    print("\nTest Results:")
    for text, prediction in zip(test_data["text"], predictions):
        # Sort intents by confidence score
        sorted_intents = sorted(prediction.items(), key=lambda x: x[1], reverse=True)
        top_intent = sorted_intents[0]

        print(f"\nText: {text}")
        print(f"Predicted intent: {top_intent[0]} (confidence: {top_intent[1]:.3f})")
        print("All scores:")
        for intent, score in sorted_intents:
            print(f"  {intent}: {score:.3f}")


def test_dataset_logging(intents_data):
    """Test that the dataset logging functionality works correctly"""
    # Create training config with dataset logging enabled
    with mlflow.start_run():
        log_dataset(intents_data)
        run_id = mlflow.active_run().info.run_id

    # Verify that the dataset was logged
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id)
    dataset = run_data.inputs.dataset_inputs[0].dataset
    assert dataset.name == "training_data"


if __name__ == "__main__":
    pytest.main([__file__])
