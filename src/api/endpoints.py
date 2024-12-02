import asyncio
import base64
import io
import json
import multiprocessing
import os
import queue
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path

import mlflow
import mlflow.deployments
import pandas as pd
import polars as pl
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ml.huggingface import upload_model_to_hub
from ml.mlflow import log_dataset, package_model
from ml.train import train_intent_classifier
from schema import (
    HuggingFaceUploadRequest,
    HuggingFaceUploadResponse,
    ModelBuildRequest,
    ModelSearchRequest,
    RegisterModelRequest,
    TrainingConfig,
    TrainingRequest,
    TrainingResponse,
)

app = FastAPI()

# Set up templates and static files
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main UI page."""
    return templates.TemplateResponse("index.html", {"request": request})


# Replace deprecated get_latest_versions calls with newer API
def get_latest_model_version(client, model_name):
    """Get the latest version of a model without using deprecated stages API"""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    # Sort by version number (descending) and get the first one
    return sorted(versions, key=lambda x: int(x.version), reverse=True)[0]


@app.get("/model")
def get_models(name_contains: str = None, intents: str = None, limit: int = 100):
    """
    List all registered models and their details from MLflow.
    Supports filtering by name and intents.

    Parameters:
        name_contains (str, optional): Substring to match in model names
        intents (str, optional): Comma-separated list of required intents
        limit (int, optional): Maximum number of results to return (default: 100)

    Returns:
        list: List of matching model information dictionaries
    """
    try:
        client = mlflow.tracking.MlflowClient()
        results = []

        # Parse intents if provided
        required_intents = []
        if intents:
            required_intents = [i.strip() for i in intents.split(",") if i.strip()]

        # Get all registered models
        registered_models = client.search_registered_models()

        for rm in registered_models:
            # Apply name filter if specified
            if name_contains:
                if name_contains.lower() not in rm.name.lower():
                    continue  # Skip this model if name doesn't match

            model_info = {
                "name": rm.name,
                "description": rm.description,
                "creation_timestamp": rm.creation_timestamp,
                "last_updated_timestamp": rm.last_updated_timestamp,
            }

            # Get latest version
            latest = get_latest_model_version(client, rm.name)
            if latest:
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

            # Apply intent filters if specified
            if required_intents:
                if not set(required_intents).issubset(set(model_intents)):
                    continue  # Skip this model if it doesn't have all required intents

            results.append(model_info)

            # Apply result limit
            if len(results) >= limit:
                break

        return results

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
            latest_version = get_latest_model_version(client, model_id)

            # Basic model info
            model_info.update({
                "name": registered_model.name,
                "version": latest_version.version if latest_version else None,
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
            if latest_version and latest_version.run_id:
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
            latest = get_latest_model_version(client, rm.name)
            if latest:
                model_info["version"] = latest.version

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


def train_model_process(
    request_dict: dict, progress_queue: Queue, shutdown_event: Event
):
    """
    Process function that runs the model training.
    Checks shutdown_event periodically to handle graceful termination.
    """
    try:
        # Convert dict back to TrainingRequest
        request = TrainingRequest(**request_dict)

        if request.experiment_name:
            mlflow.set_experiment(request.experiment_name)

        # Load dataset based on source type
        if request.dataset_source.source_type == "url":
            if not request.dataset_source.url:
                progress_queue.put({
                    "error": "URL must be provided when source_type is 'url'"
                })
                return

            response = requests.get(str(request.dataset_source.url))
            if response.status_code != 200:
                progress_queue.put({
                    "error": f"Failed to download dataset from URL: {response.status_code}"
                })
                return

            if isinstance(response.text, bytes):
                data = pl.read_csv(io.StringIO(response.text.decode()))
            else:
                data = pl.read_csv(io.StringIO(response.text))

        elif request.dataset_source.source_type == "upload":
            if not request.dataset_source.file_content:
                progress_queue.put({
                    "error": "File content must be provided when source_type is 'upload'"
                })
                return

            try:
                file_content = base64.b64decode(request.dataset_source.file_content)
                data = pl.read_csv(io.BytesIO(file_content))
            except Exception as e:
                progress_queue.put({
                    "error": f"Failed to decode file content: {str(e)}"
                })
                return
        else:
            progress_queue.put({
                "error": "Invalid source_type. Must be 'url' or 'upload'"
            })
            return

        # Validate dataset columns
        required_columns = ["intent", "text"]
        if not all(col in data.columns for col in required_columns):
            progress_queue.put({
                "error": f"Dataset must contain columns: {required_columns}"
            })
            return

        # Validate intents in dataset match requested intents
        unique_intents = set(data["intent"].unique().to_list())
        if request.intents:  # Only validate if intents are specified
            if not set(request.intents).issubset(unique_intents):
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested intents {request.intents} are not all present in dataset intents {list(unique_intents)}",
                )

        # Create training config
        training_config = request.training_config or TrainingConfig()

        # Check for early termination
        if shutdown_event.is_set():
            progress_queue.put({"error": "Training cancelled by client"})
            return

        # Train the model with config and progress reporting
        progress_queue.put({
            "status": "training",
            "message": "Starting model training...",
        })
        with mlflow.start_run():
            if request.log_dataset_to_mlflow:
                progress_queue.put({
                    "status": "logging_dataset",
                    "message": "Logging dataset to MLflow...",
                })
                log_dataset(data)

            model, intents, tokenizer = train_intent_classifier(
                data, training_config, progress_queue, shutdown_event
            )

            # Check for termination after training
            if shutdown_event.is_set():
                progress_queue.put({"error": "Training cancelled by client"})
                return

            progress_queue.put({"status": "packaging", "message": "Packaging model..."})
            run_id = package_model(model, intents, tokenizer)

            progress_queue.put({
                "status": "complete",
                "run_id": run_id,
                "message": "Model training completed successfully",
            })

    except Exception as e:
        progress_queue.put({"error": str(e)})


@app.post("/model/train/stream")
async def train_model_stream(
    request: TrainingRequest, background_tasks: BackgroundTasks
):
    """
    Stream the training process for a new model.

    This endpoint runs the training in a separate process and streams progress updates.
    If the client disconnects, the training process will be terminated.

    Returns:
        StreamingResponse: A stream of JSON events containing training progress and final run_id
    """
    progress_queue = multiprocessing.Queue()
    shutdown_event = multiprocessing.Event()

    # Start training process
    process = multiprocessing.Process(
        target=train_model_process,
        args=(request.model_dump(), progress_queue, shutdown_event),
    )
    process.start()

    async def cleanup():
        """Cleanup function that runs when the client disconnects"""
        shutdown_event.set()  # Signal the training process to stop
        process.join(timeout=5)  # Wait up to 5 seconds for graceful shutdown
        if process.is_alive():
            process.terminate()  # Force terminate if still running
            process.join()  # Ensure process is fully cleaned up

    async def event_stream():
        try:
            while True:
                try:
                    # Non-blocking queue check
                    try:
                        data = progress_queue.get_nowait()
                        if "error" in data:
                            yield f"data: {json.dumps({'status': 'error', 'message': data['error']})}\n\n"
                            break
                        yield f"data: {json.dumps(data)}\n\n"
                        if data.get("status") == "complete":
                            break
                    except queue.Empty:
                        if not process.is_alive():  # Check if process died unexpectedly
                            yield f"data: {json.dumps({'status': 'error', 'message': 'Training process terminated unexpectedly'})}\n\n"
                            break
                        await asyncio.sleep(0.1)
                        continue

                except Exception as e:
                    yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
                    break
        finally:
            await cleanup()

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", background=background_tasks
    )


@app.post("/model/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest) -> dict:
    """
    Train a new model using the provided dataset and configuration.

    Returns:
        TrainingResponse: Contains the MLflow run ID and training status
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
        if request.intents:  # Only validate if intents are specified
            if not set(request.intents).issubset(unique_intents):
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested intents {request.intents} are not all present in dataset intents {list(unique_intents)}",
                )

        # Create training config
        training_config = request.training_config or TrainingConfig()

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
        if "model_cache" not in os.listdir():
            os.makedirs("model_cache", exist_ok=True)
        dst_path = f"model_cache/{model_id}"
        if model_id in os.listdir("model_cache"):
            try:
                loaded_model = mlflow.pyfunc.load_model(dst_path + "/intent_model")
            except mlflow.exceptions.MlflowException as e:
                raise HTTPException(status_code=500, detail=str(e))
        else:
            os.makedirs(dst_path, exist_ok=True)
            try:
                loaded_model = mlflow.pyfunc.load_model(
                    f"models:/{model_id}/latest",
                    dst_path=dst_path,
                )
            except mlflow.exceptions.MlflowException:
                # If not found as registered model, try loading as a run
                try:
                    loaded_model = mlflow.pyfunc.load_model(
                        f"runs:/{model_id}/intent_model",
                        dst_path=dst_path,
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


@app.post("/model/upload-huggingface", response_model=HuggingFaceUploadResponse)
async def upload_to_huggingface(request: HuggingFaceUploadRequest):
    """Upload a model to Hugging Face Hub."""
    try:
        model_url = upload_model_to_hub(
            run_id=request.run_id,
            repo_name=request.repo_name,
            hf_token=request.hf_token,
            organization=request.organization,
            private=request.private,
            commit_message=request.commit_message,
        )
        return HuggingFaceUploadResponse(model_url=model_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/build")
def build_model(model_build_request: ModelBuildRequest):
    mlflow.deployments.build_model(
        model_uri=model_build_request.model_uri,
        docker_image_name=model_build_request.docker_image_name,
    )
