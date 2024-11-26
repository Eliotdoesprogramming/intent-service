import base64
import io

import mlflow
import pandas as pd
import polars as pl
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from ml.train import package_model, train_intent_classifier
from schema import (
    ModelSearchRequest,
    RegisterModelRequest,
    TrainingConfig,
    TrainingRequest,
    TrainingResponse,
)

app = FastAPI()


@app.get("/")
async def redirect_to_docs():
    """
    Redirect to API documentation.

    Automatically redirects all root path requests to the Swagger UI documentation page,
    providing a user-friendly interface to explore and test the API.

    Returns:
        RedirectResponse: A 302 redirect to the /docs endpoint
    """
    return RedirectResponse(url="/docs")


@app.get("/model")
def get_models(request: ModelSearchRequest = None):
    """
    List all registered models and their details from MLflow.

    Parameters:
        request (ModelSearchRequest, optional): Search criteria for filtering models

    Returns:
        list: List of model information dictionaries containing:
            - name: Model name
            - version: Latest version number
            - description: Model description
            - creation_timestamp: When model was created
            - last_updated_timestamp: When model was last modified
            - intents: List of supported intent labels
            - tags: Additional metadata tags
            - run_info: Information about the training run

    Raises:
        HTTPException(500): If there are errors accessing MLflow
    """
    try:
        client = mlflow.tracking.MlflowClient()
        models = []

        # Get all registered models
        registered_models = client.search_registered_models()

        for model in registered_models:
            model_info = {
                "name": model.name,
                "description": model.description,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
            }

            # Get latest version
            latest_versions = client.get_latest_versions(model.name, stages=["None"])
            if latest_versions:
                latest = latest_versions[0]
                model_info["version"] = latest.version

                # Get run info if available
                if latest.run_id:
                    run = client.get_run(latest.run_id)
                    model_info["run_info"] = {
                        "run_id": run.info.run_id,
                        "status": run.info.status,
                        "metrics": run.data.metrics,
                        "params": run.data.params,
                    }

            # Get intent labels from tags
            intent_tags = {
                k: v for k, v in model.tags.items() if k.startswith("intent_")
            }
            model_info["intents"] = [
                k.replace("intent_", "") for k in intent_tags.keys()
            ]

            # Add remaining tags
            model_info["tags"] = {
                k: v for k, v in model.tags.items() if not k.startswith("intent_")
            }

            models.append(model_info)

        return models

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


# Endpoint to list available models
@app.get("/model/{model_id}")
def get_model_info(model_id: str):
    """
    Retrieve detailed information about a specific model from MLflow.

    This endpoint attempts to find the model in two ways:
    1. As a registered model in the MLflow Model Registry
    2. As a specific run ID if not found in registry

    Parameters:
        model_id (str): Either a registered model name or MLflow run ID

    Returns:
        - dict: Model information including:
            - name: Model name (if registered)
            - version: Latest version number (if registered)
            - description: Model description
            - creation_timestamp: When the model was created
            - last_updated_timestamp: When the model was last modified
            - intents: List of supported intent labels
            - tags: Additional metadata tags
            - run_info: Information about the training run

    Raises:
        - HTTPException(404): If no model is found with the specified ID
        - HTTPException(500): If there are errors accessing MLflow
    """
    try:
        client = mlflow.tracking.MlflowClient()
        model_info = {}

        # First try to get as registered model
        try:
            registered_model = client.get_registered_model(model_id)
            latest_version = client.get_latest_versions(model_id, stages=["None"])[0]

            # Basic model info
            model_info.update({
                "name": registered_model.name,
                "version": latest_version.version,
                "description": registered_model.description,
                "creation_timestamp": registered_model.creation_timestamp,
                "last_updated_timestamp": registered_model.last_updated_timestamp,
            })

            # Get all tags
            tags = registered_model.tags if registered_model.tags else {}

            # Extract intent labels from tags
            intents = [
                tag.replace("intent_", "")
                for tag in tags.keys()
                if tag.startswith("intent_")
            ]

            model_info["intents"] = intents
            model_info["tags"] = {
                k: v for k, v in tags.items() if not k.startswith("intent_")
            }

            # Add run info if available
            if latest_version.run_id:
                run = client.get_run(latest_version.run_id)
                model_info["run_info"] = {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                }

        except mlflow.exceptions.MlflowException:
            # Try to get as run ID instead
            try:
                run = client.get_run(model_id)
                model_info.update({
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                })

                # Load model to get intents
                model = mlflow.pyfunc.load_model(f"runs:/{model_id}/intent_model")
                model_info["intents"] = model._model_impl.python_model.intent_labels

            except mlflow.exceptions.MlflowException:
                raise HTTPException(
                    status_code=404, detail=f"No model found with ID: {model_id}"
                )

        return model_info

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving model info: {str(e)}"
        )


@app.post("/model/search")
def search_models(model_search_request: ModelSearchRequest):
    """
    Search for registered models based on tags, intents, and name patterns.

    Args:
        model_search_request (ModelSearchRequest): Search criteria including:
            - tags: Optional dict of tag key-value pairs to match
            - intents: Optional list of required intent labels
            - name_contains: Optional substring to match in model names
            - limit: Maximum number of results to return (default: 100)

    Returns:
        list: List of matching model information dictionaries containing:
            - name: Model name
            - version: Latest version
            - description: Model description
            - intents: List of supported intents
            - tags: Model tags
            - creation_timestamp: When model was created
            - last_updated_timestamp: Last update time

    Raises:
        HTTPException(500): If there are errors accessing MLflow
    """
    try:
        client = mlflow.tracking.MlflowClient()
        results = []

        # Get all registered models
        registered_models = client.search_registered_models()

        for rm in registered_models:
            match = True
            model_info = {
                "name": rm.name,
                "description": rm.description,
                "creation_timestamp": rm.creation_timestamp,
                "last_updated_timestamp": rm.last_updated_timestamp,
            }

            # Get latest version
            latest_versions = client.get_latest_versions(rm.name, stages=["None"])
            if latest_versions:
                model_info["version"] = latest_versions[0].version

            # Get all tags
            tags = rm.tags if rm.tags else {}

            # Extract intents from tags
            model_intents = [
                tag.replace("intent_", "")
                for tag in tags.keys()
                if tag.startswith("intent_")
            ]
            model_info["intents"] = model_intents

            # Filter non-intent tags
            model_info["tags"] = {
                k: v for k, v in tags.items() if not k.startswith("intent_")
            }

            # Apply name filter if specified
            if (
                model_search_request.name_contains
                and model_search_request.name_contains.lower() not in rm.name.lower()
            ):
                match = False

            # Apply tag filters if specified
            if model_search_request.tags:
                for key, value in model_search_request.tags.items():
                    if key not in tags or tags[key] != value:
                        match = False
                        break

            # Apply intent filters if specified
            if model_search_request.intents:
                if not set(model_search_request.intents).issubset(set(model_intents)):
                    match = False

            if match:
                results.append(model_info)

            # Apply result limit
            if (
                model_search_request.limit
                and len(results) >= model_search_request.limit
            ):
                break

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching models: {str(e)}")


# Endpoint to update model information
@app.put("/model/{model_id}")
def update_model(model_id: str, model: RegisterModelRequest):
    # Find the model with the given ID and update its information
    pass


# Endpoint to create and train a model
@app.post("/model/register")
async def register_model(model: RegisterModelRequest):
    """
    Register an existing MLflow run as a named model.

    Args:
        model (RegisterModelRequest): Model registration details including:
            - name: Name for the registered model (must be unique in registry)
            - mlflow_run_id: ID of the MLflow run containing the model
            - description: Optional description of the model
            - tags: Optional dictionary of additional metadata tags

    Returns:
        dict: Registration result containing:
            - name: Name of the registered model
            - version: Version number assigned by MLflow
            - status: Registration status ("success" or "error")
            - message: Detailed success/failure message

    Raises:
        - HTTPException (404): If no model exists with the specified run ID
        - HTTPException (500): If registration fails due to name conflicts or other errors

    The endpoint performs the following operations:
    1. Verifies the existence of the model in the specified MLflow run
    2. Loads the model to extract intent labels
    3. Registers the model with the provided name
    4. Adds model description and metadata as registry tags
    5. Records all supported intents as model tags

    Notes:
        - Model names must be unique in the registry
        - Existing models with the same name will create a new version
        - All intent labels are automatically extracted and stored as tags
        - Custom tags can be used for filtering and organization
    """
    try:
        # Load the model to verify it exists
        try:
            loaded_model = mlflow.pyfunc.load_model(
                f"runs:/{model.mlflow_run_id}/intent_model"
            )
            intents = loaded_model._model_impl.python_model.intent_labels
            del loaded_model
        except mlflow.exceptions.MlflowException:
            raise HTTPException(
                status_code=404,
                detail=f"No model found with run ID: {model.mlflow_run_id}",
            )

        # Register the model
        model_uri = f"runs:/{model.mlflow_run_id}/intent_model"
        registered_model = mlflow.register_model(model_uri=model_uri, name=model.name)

        # Add description and tags
        client = mlflow.tracking.MlflowClient()
        client.update_registered_model(
            name=registered_model.name,
            description=model.description if model.description else None,
        )

        # Add intents as model tags
        for intent in intents:
            client.set_registered_model_tag(
                name=registered_model.name, key=f"intent_{intent}", value="true"
            )

        # Add any extra metadata as tags
        for key, value in model.tags.items():
            client.set_registered_model_tag(
                name=registered_model.name, key=key, value=str(value)
            )

        return {
            "name": registered_model.name,
            "version": registered_model.version,
            "status": "success",
            "message": f"Model {model.name} registered successfully",
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error registering model: {str(e)}"
        )


# Endpoint to delete a model
@app.delete("/model/{model_id}")
def delete_model(model_id: int):
    # Find the model with the given ID and remove it from the list of models
    pass


@app.post("/model/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest) -> dict:
    """
    Train a new intent classification model.

    Args:
        request (TrainingRequest): Training configuration including:
            - dataset_source: Source configuration for training data
                - source_type: Either 'url' or 'upload'
                - url: URL to download dataset (for 'url' type)
                - file_content: Base64 encoded CSV content (for 'upload' type)
            - intents: List of valid intents for classification
            - experiment_name: Optional MLflow experiment name
            - model_name: Optional base model name to use
            - training_config: Optional custom training parameters

    Returns:
        TrainingResponse: Training result containing:
            - model_id: MLflow run ID of the trained model
            - status: Training status ("success" or "error")
            - message: Detailed success/failure message

    Raises:
        - HTTPException (400): If dataset format is invalid or configuration errors
        - HTTPException (500): If training fails due to resource or processing errors

    The training process supports two data source types:
    - URL: Downloads CSV data from a specified URL
    - Upload: Accepts base64 encoded CSV content directly

    Notes:
        - Dataset must contain 'intent' and 'text' columns
        - All intents in dataset must be included in request's intent list
        - Training progress is tracked in MLflow for reproducibility
        - Large datasets may require significant processing time
    """
    try:
        if request.experiment_name:
            mlflow.set_experiment(request.experiment_name)
        # Load dataset based on source type
        if request.dataset_source.source_type == "url":
            if not request.dataset_source.url:
                raise HTTPException(
                    status_code=400,
                    detail="URL must be provided when source_type is 'url'",
                )
            # Download dataset from URL
            response = requests.get(str(request.dataset_source.url))
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download dataset from URL: {response.status_code}",
                )

            if isinstance(response.text, bytes):
                data = pl.read_csv(io.StringIO(response.text.decode()))
            else:
                data = pl.read_csv(io.StringIO(response.text))

        elif request.dataset_source.source_type == "upload":
            if not request.dataset_source.file_content:
                raise HTTPException(
                    status_code=400,
                    detail="File content must be provided when source_type is 'upload'",
                )
            # Decode base64 file content
            try:
                file_content = base64.b64decode(request.dataset_source.file_content)
                data = pl.read_csv(io.BytesIO(file_content))
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to decode file content: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=400, detail="Invalid source_type. Must be 'url' or 'upload'"
            )

        # Validate dataset columns
        required_columns = ["intent", "text"]
        if not all(col in data.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"Dataset must contain columns: {required_columns}",
            )

        # Validate intents in dataset match requested intents
        unique_intents = set(data["intent"].unique().to_list())
        if not unique_intents.issubset(set(request.intents)):
            raise HTTPException(
                status_code=400,
                detail="Dataset contains intents not specified in the model configuration",
            )

        # Create training config
        training_config = request.training_config or TrainingConfig()
        if request.model_name:
            training_config.base_model_name = request.model_name

        # Train the model with config
        model, intents, tokenizer = train_intent_classifier(data, training_config)
        run_id = package_model(model, intents, tokenizer)

        return TrainingResponse(
            model_id=run_id, status="success", message="Model trained successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@app.post("/model/{model_id}/predict")
async def predict(model_id: str, text: str) -> dict:
    """
    Generate intent predictions for input text.

    Args:
        model_id (str): ID of the model to use. Can be either:
            - A registered model name (loads latest version)
            - An MLflow run ID (loads specific run)
        text (str): Input text to classify (should be non-empty)

    Returns:
        dict: Dictionary containing intent confidence scores:
            - Keys: All possible intent labels
            - Values: Confidence scores (0.0 to 1.0) for each intent
            Example: {"intent1": 0.8, "intent2": 0.15, "intent3": 0.05}

    Raises:
        - HTTPException (404): If no model is found with the specified ID
        - HTTPException (500): If there are errors during prediction

    The endpoint provides flexible model loading and prediction:
    1. Attempts to load the model from the registry using the model_id as name
    2. If not found, attempts to load directly from MLflow run
    3. Processes the input text and generates confidence scores for all intents

    Notes:
        - Confidence scores sum to 1.0 across all intents
        - Empty or very short texts may result in unreliable predictions
        - Model loading time may vary based on size and storage location
        - Registered models always use the latest version unless specified
    """
    try:
        # First try loading as a registered model
        try:
            loaded_model = mlflow.pyfunc.load_model(f"models:/{model_id}/latest")
        except mlflow.exceptions.MlflowException:
            # If not found as registered model, try loading as a run
            try:
                loaded_model = mlflow.pyfunc.load_model(
                    f"runs:/{model_id}/intent_model"
                )
            except mlflow.exceptions.MlflowException:
                raise HTTPException(
                    status_code=404,
                    detail=f"No model found with ID {model_id} (tried both registered models and runs)",
                )

        # Create a pandas DataFrame with the input text
        test_data = pd.DataFrame({"text": [text]})

        # Get prediction
        prediction = loaded_model.predict(test_data)

        # Return the prediction dictionary (contains all intent scores)
        return prediction[0]  # First element since we only predicted one text
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
