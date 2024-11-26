from unittest.mock import MagicMock, PropertyMock, patch

import mlflow
import polars as pl
import pytest
from typer.testing import CliRunner

from cli import app

runner = CliRunner()


@pytest.fixture
def mock_mlflow():
    with patch("cli.get_mlflow") as mock:
        mlflow_mock = MagicMock()
        mock.return_value = mlflow_mock
        yield mlflow_mock


@pytest.fixture
def mock_train_utils():
    with patch("cli.get_train_utils") as mock:
        package_model_mock = MagicMock(return_value="test_run_id")
        train_classifier_mock = MagicMock(
            return_value=(MagicMock(), ["greeting", "farewell"], MagicMock())
        )
        mock.return_value = (package_model_mock, train_classifier_mock)
        yield package_model_mock, train_classifier_mock


@pytest.fixture
def mock_api():
    with patch("cli.get_api") as mock:
        app_mock = MagicMock()
        uvicorn_mock = MagicMock()
        mock.return_value = (app_mock, uvicorn_mock)
        yield app_mock, uvicorn_mock


def test_register_command(mock_mlflow):
    # Mock the model loading and registration
    loaded_model = MagicMock()
    loaded_model._model_impl.python_model.intent_labels = ["greeting", "farewell"]
    mock_mlflow.pyfunc.load_model.return_value = loaded_model

    mock_mlflow.register_model.return_value = MagicMock(version="1")

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

    # Test registration with invalid run ID
    mock_mlflow.pyfunc.load_model.side_effect = mlflow.exceptions.MlflowException(
        "Not found"
    )
    result = runner.invoke(app, ["register", "invalid_run_id", "test_model"])
    assert result.exit_code == 1


def test_search_command(mock_mlflow):
    # Mock the search results
    mock_client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    # Create a more complete mock model with all required attributes
    mock_model = MagicMock(
        name="test_model",
        description="Test model",
        creation_timestamp=1234567890,
        last_updated_timestamp=1234567890,
    )
    # Mock tags as a property to ensure proper access
    type(mock_model).tags = PropertyMock(
        return_value={"intent_greeting": "true", "version": "1.0.0"}
    )

    mock_client.search_registered_models.return_value = [mock_model]

    # Mock version with more complete attributes
    mock_version = MagicMock(
        version="1", current_stage="None", status="READY", run_id="test_run_id"
    )
    mock_client.get_latest_versions.return_value = [mock_version]

    # Remove the stderr debug print since it's not captured
    result = runner.invoke(app, ["search"], catch_exceptions=True)
    print(f"Command output: {result.stdout}")
    if result.exception:
        print(f"Exception: {result.exception}")
    assert result.exit_code == 0

    # Add specific assertions about what should be in the output
    # This will help identify what's not rendering correctly
    assert "test_model" in result.stdout
    assert "Test model" in result.stdout
    assert "1.0.0" in result.stdout


def test_train_command(mock_mlflow, mock_train_utils, tmp_path):
    # Create a temporary CSV file
    data_path = tmp_path / "test_data.csv"
    test_data = pl.DataFrame({
        "intent": ["greeting", "farewell"],
        "text": ["hello", "goodbye"],
    })
    test_data.write_csv(data_path)

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

    # Test training with invalid dataset
    result = runner.invoke(app, ["train", str(tmp_path / "nonexistent.csv")])
    assert result.exit_code == 1
    assert "Error loading dataset" in result.stdout


def test_info_command(mock_mlflow):
    # Mock model info retrieval
    mock_client = mock_mlflow.tracking.MlflowClient()
    mock_client.get_registered_model.return_value = MagicMock(
        name="test_model",
        description="Test model",
        tags={"intent_greeting": "true"},
        creation_timestamp=1234567890,
        last_updated_timestamp=1234567890,
    )
    mock_client.get_latest_versions.return_value = [
        MagicMock(version="1", run_id="test_run")
    ]
    mock_client.get_run.return_value = MagicMock(
        info=MagicMock(
            run_id="test_run",
            status="FINISHED",
            start_time=1234567890,
            end_time=1234567890,
        ),
        data=MagicMock(metrics={"accuracy": 0.95}, params={"epochs": "3"}),
    )

    # Test info retrieval by model name
    result = runner.invoke(app, ["info", "test_model"])
    assert result.exit_code == 0
    assert "Model Information" in result.stdout

    # Test info retrieval by run ID
    result = runner.invoke(app, ["info", "test_run"])
    assert result.exit_code == 0
    assert "Model Information" in result.stdout


def test_predict_command(mock_mlflow):
    # Mock model loading and prediction
    mock_model = MagicMock()
    mock_model.predict.return_value = [{"greeting": 0.8, "farewell": 0.2}]
    mock_mlflow.pyfunc.load_model.return_value = mock_model

    # Test prediction with registered model
    result = runner.invoke(app, ["predict", "test_model", "hello there"])
    assert result.exit_code == 0
    assert "Intent Predictions" in result.stdout

    # Test prediction with invalid model
    mock_mlflow.pyfunc.load_model.side_effect = mlflow.exceptions.MlflowException(
        "Not found"
    )
    result = runner.invoke(app, ["predict", "invalid_model", "hello"])
    assert result.exit_code == 1
    assert "Error making prediction" in result.stdout


def test_serve_command(mock_api):
    _app_mock, uvicorn_mock = mock_api

    # Test development mode
    result = runner.invoke(app, ["serve", "--port", "8000"])
    assert result.exit_code == 0
    assert "Starting API server" in result.stdout

    # Test production mode
    result = runner.invoke(app, ["serve", "--environment", "prod", "--workers", "4"])
    assert result.exit_code == 0
    assert "Starting API server" in result.stdout

    # Verify uvicorn configuration
    uvicorn_mock.run.assert_called()
