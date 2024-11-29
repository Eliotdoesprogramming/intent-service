from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from api.client import IntentServiceClient
from cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def setup_api_client():
    """Reset the API client before each test."""
    with patch("cli.get_api_client") as mock:
        yield mock


@pytest.fixture
def mock_api_client(setup_api_client):
    """Create a mock API client and inject it into the CLI."""
    client = MagicMock(spec=IntentServiceClient)
    setup_api_client.return_value = client
    return client


def test_predict_command(mock_api_client):
    # Mock prediction response
    mock_api_client.predict.return_value = {"greeting": 0.8, "farewell": 0.2}

    # Test successful prediction
    result = runner.invoke(app, ["predict", "test_model", "hello there"])
    assert result.exit_code == 0
    assert "Intent Predictions" in result.stdout
    mock_api_client.predict.assert_called_once_with(
        model_id="test_model", text="hello there"
    )

    # Test prediction with API error
    mock_api_client.predict.side_effect = Exception("API Error")
    result = runner.invoke(app, ["predict", "invalid_model", "hello"])
    assert result.exit_code == 1
    assert "Error making prediction" in result.stdout


def test_train_command(mock_api_client, tmp_path):
    # Create a temporary CSV file
    data_path = tmp_path / "test_data.csv"
    data_path.write_text("intent,text\ngreeting,hello\nfarewell,goodbye")

    # Mock training response
    mock_api_client.train.return_value = "test_run_id"

    # Test successful training
    result = runner.invoke(
        app,
        [
            "train",
            str(data_path),
            "--experiment-name",
            "test_experiment",
            "--num-epochs",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "Successfully trained model" in result.stdout
    mock_api_client.train.assert_called_once()

    # Test training with invalid dataset
    nonexistent_path = tmp_path / "nonexistent.csv"
    result = runner.invoke(app, ["train", str(nonexistent_path)])
    assert result.exit_code == 1
    expected_error = f"Error: Dataset file not found: {str(nonexistent_path).strip()}"
    assert expected_error in result.stdout.replace("\n", "")


def test_info_command(mock_api_client):
    # Mock model info response
    mock_api_client.get_model_info.return_value = {
        "name": "test_model",
        "version": "1",
        "description": "Test model",
        "tags": {"version": "1.0.0"},
        "intents": ["greeting", "farewell"],
        "run_info": {
            "run_id": "test_run",
            "status": "FINISHED",
            "metrics": {"accuracy": 0.95},
            "params": {"epochs": "3"},
        },
    }

    # Test info retrieval
    result = runner.invoke(app, ["info", "test_model"])
    assert result.exit_code == 0
    assert "Model Information" in result.stdout
    mock_api_client.get_model_info.assert_called_once_with("test_model")

    # Test info retrieval with API error
    mock_api_client.get_model_info.side_effect = Exception("API Error")
    result = runner.invoke(app, ["info", "invalid_model"])
    assert result.exit_code == 1
    assert "Error retrieving model info" in result.stdout


def test_register_command(mock_api_client):
    # Mock registration response
    mock_api_client.register_model.return_value = {
        "name": "test_model",
        "version": "1",
        "status": "success",
    }

    # Test successful registration
    result = runner.invoke(
        app,
        [
            "register",
            "test_run_id",
            "test_model",
            "--description",
            "Test model",
            "--tags",
            '{"version": "1.0.0"}',
        ],
    )
    assert result.exit_code == 0
    assert "Successfully registered model" in result.stdout
    mock_api_client.register_model.assert_called_once()

    # Test registration with API error
    mock_api_client.register_model.side_effect = Exception("API Error")
    result = runner.invoke(app, ["register", "invalid_run_id", "test_model"])
    assert result.exit_code == 1
    assert "Error registering model" in result.stdout


def test_search_command(mock_api_client):
    # Mock search response
    mock_api_client.search_models.return_value = [
        {
            "name": "test_model",
            "version": "1",
            "description": "Test model",
            "tags": {"version": "1.0.0"},
            "intents": ["greeting", "farewell"],
        }
    ]

    # Test search with no filters
    result = runner.invoke(app, ["search"])
    assert result.exit_code == 0
    assert "Search Results" in result.stdout
    mock_api_client.search_models.assert_called_with({})

    # Test search with filters
    result = runner.invoke(
        app,
        [
            "search",
            "--name-contains",
            "test",
            "--tags",
            '{"version": "1.0.0"}',
            "--intents",
            "greeting,farewell",
        ],
    )
    assert result.exit_code == 0
    mock_api_client.search_models.assert_called_with({
        "name_contains": "test",
        "tags": {"version": "1.0.0"},
        "intents": ["greeting", "farewell"],
    })

    # Test search with no results
    mock_api_client.search_models.return_value = []
    result = runner.invoke(app, ["search"])
    assert result.exit_code == 0
    assert "No models found" in result.stdout

    # Test search with API error
    mock_api_client.search_models.side_effect = Exception("API Error")
    result = runner.invoke(app, ["search"])
    assert result.exit_code == 1
    assert "Error searching models" in result.stdout


def test_serve_command():
    with patch("cli.get_api") as mock_api:
        app_mock, uvicorn_mock = MagicMock(), MagicMock()
        mock_api.return_value = (app_mock, uvicorn_mock)

        # Test development mode
        result = runner.invoke(app, ["serve", "--port", "8000"])
        assert result.exit_code == 0
        assert "Starting API server" in result.stdout

        # Test production mode
        result = runner.invoke(
            app, ["serve", "--environment", "prod", "--workers", "4"]
        )
        assert result.exit_code == 0
        assert "Starting API server" in result.stdout

        # Verify uvicorn configuration
        uvicorn_mock.run.assert_called()


def test_api_url_configuration():
    """Test different ways of configuring the API URL."""
    # Test default URL
    client = IntentServiceClient()
    assert client.base_url == "http://localhost:8000"

    # Test command-line option
    with patch("cli.IntentServiceClient") as mock_client_class:
        mock_client = MagicMock(spec=IntentServiceClient)
        mock_client_class.return_value = mock_client
        mock_client.get_model_info.return_value = {
            "name": "test_model",
            "version": "1",
        }

        result = runner.invoke(
            app, ["--api-url", "http://cli.example.com", "info", "test-model"]
        )
        assert result.exit_code == 0
        mock_client_class.assert_called_once_with(base_url="http://cli.example.com")

    # Test environment variable
    with patch("cli.IntentServiceClient") as mock_client_class:
        mock_client = MagicMock(spec=IntentServiceClient)
        mock_client_class.return_value = mock_client
        mock_client.get_model_info.return_value = {
            "name": "test_model",
            "version": "1",
        }

        with patch.dict("os.environ", {"INTENT_SERVICE_URL": "http://env.example.com"}):
            result = runner.invoke(app, ["info", "test-model"])
            assert result.exit_code == 0
            mock_client_class.assert_called_once_with(base_url="http://env.example.com")
