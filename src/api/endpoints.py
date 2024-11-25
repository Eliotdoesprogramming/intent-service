import polars as pl
import mlflow
import pandas as pd
import requests
import base64
import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from schema import RegisterModelRequest, TrainingRequest, TrainingResponse, TrainingConfig
from ml.train import package_model, train_intent_classifier
app = FastAPI()

@app.get("/")
def redirect_to_docs():
    """
    Redirects users to the interactive API documentation page.
    
    This endpoint automatically redirects all root path requests to the Swagger UI
    documentation page, providing a user-friendly interface to explore and test the API.
    
    Returns:
        RedirectResponse: A 302 redirect to the /docs endpoint
    """
    return RedirectResponse(url="/docs")

# Endpoint to list available models
@app.get("/model")
def get_models():
    pass

# Endpoint to update model information
@app.put("/model/{model_id}")
def update_model(model_id: int, model: RegisterModelRequest):
    # Find the model with the given ID and update its information
    pass

# Endpoint to create and train a model
@app.post("/model/register")
def register_model(model: RegisterModelRequest):
    """
    Register an existing MLflow run as a named model in the MLflow Model Registry.
    
    This endpoint performs the following operations:
    1. Verifies the existence of the model in the specified MLflow run
    2. Loads the model to extract intent labels
    3. Registers the model with the provided name in MLflow's Model Registry
    4. Adds model description and metadata as registry tags
    5. Records all supported intents as model tags for discoverability
    
    Parameters:
        model (RegisterModelRequest): Model registration details including:
            - name: Name for the registered model (must be unique in registry)
            - mlflow_run_id: ID of the MLflow run containing the model
            - description: Optional description of the model's purpose and characteristics
            - tags: Optional dictionary of additional metadata tags for filtering and organization
    
    Returns:
        dict: Registration result containing:
            - name: Name of the registered model
            - version: Version number assigned by MLflow
            - status: Registration status ("success" or "error")
            - message: Detailed success/failure message
            
    Raises:
        HTTPException(404): If no model exists with the specified run ID or if the run doesn't contain
                          a valid intent classification model
        HTTPException(500): If registration fails due to:
                          - Name conflicts in the registry
                          - Invalid model format
                          - MLflow connection issues
                          - Other unexpected errors
    
    Notes:
        - Model names must be unique in the registry
        - Existing models with the same name will create a new version
        - All intent labels are automatically extracted and stored as tags
        - Custom tags can be used for filtering and organization
    """
    try:
        # Load the model to verify it exists
        try:
            loaded_model = mlflow.pyfunc.load_model(f"runs:/{model.mlflow_run_id}/intent_model")
            intents = loaded_model._model_impl.python_model.intent_labels
            del loaded_model
        except mlflow.exceptions.MlflowException:
            raise HTTPException(
                status_code=404,
                detail=f"No model found with run ID: {model.mlflow_run_id}"
            )
            
        # Register the model
        model_uri = f"runs:/{model.mlflow_run_id}/intent_model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model.name
        )
        
        # Add description and tags
        client = mlflow.tracking.MlflowClient()
        client.update_registered_model(
            name=registered_model.name,
            description=model.description if model.description else None
        )
        
        # Add intents as model tags
        for intent in intents:
            client.set_registered_model_tag(
                name=registered_model.name,
                key=f"intent_{intent}",
                value="true"
            )
            
        # Add any extra metadata as tags
        for key, value in model.tags.items():
            client.set_registered_model_tag(
                name=registered_model.name,
                key=key,
                value=str(value)
            )
            
        return {
            "name": registered_model.name,
            "version": registered_model.version,
            "status": "success",
            "message": f"Model {model.name} registered successfully"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error registering model: {str(e)}"
        )

# Endpoint to delete a model
@app.delete("/model/{model_id}")
def delete_model(model_id: int):
    # Find the model with the given ID and remove it from the list of models
    pass

@app.post("/model/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest) -> dict:
    """
    Train a new intent classification model using the provided dataset and configuration.
    
    This endpoint handles the complete training pipeline:
    1. Dataset loading and validation from either URL or direct upload
    2. Data preprocessing and integrity checks
    3. Model training with specified configuration
    4. Model packaging and MLflow tracking
    
    The training process supports two data source types:
    - URL: Downloads CSV data from a specified URL
    - Upload: Accepts base64 encoded CSV content directly
    
    Parameters:
        request (TrainingRequest): Training configuration including:
            - dataset_source: Source configuration for training data
                - source_type: Either 'url' or 'upload'
                - url: URL to download dataset (required if source_type is 'url')
                - file_content: Base64 encoded CSV content (required if source_type is 'upload')
            - intents: List of valid intents for classification (must match dataset)
            - experiment_name: Optional MLflow experiment name for organization
            - model_name: Optional base model name to use for training
            - training_config: Optional custom training parameters including:
                - learning_rate
                - batch_size
                - epochs
                - etc.
    
    Returns:
        TrainingResponse: Training result containing:
            - model_id: MLflow run ID of the trained model (used for registration/prediction)
            - status: Training status ("success" or "error")
            - message: Detailed success/failure message
            
    Raises:
        HTTPException(400): If:
            - Dataset format is invalid (missing required columns)
            - Dataset contains intents not specified in configuration
            - URL is inaccessible or returns invalid data
            - Base64 content is malformed
            - Training configuration is invalid
        HTTPException(500): If training fails due to:
            - Insufficient data
            - Resource constraints
            - MLflow tracking issues
            - Model packaging failures
    
    Notes:
        - Dataset must contain 'intent' and 'text' columns
        - All intents in the dataset must be included in the request's intent list
        - Training progress is tracked in MLflow for reproducibility
        - Large datasets may require significant processing time
        - The endpoint is asynchronous to handle long-running training jobs
    """
    try:
        if request.experiment_name:
            mlflow.set_experiment(request.experiment_name)
        # Load dataset based on source type
        if request.dataset_source.source_type == "url":
            if not request.dataset_source.url:
                raise HTTPException(
                    status_code=400,
                    detail="URL must be provided when source_type is 'url'"
                )
            # Download dataset from URL
            response = requests.get(str(request.dataset_source.url))
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download dataset from URL: {response.status_code}"
                )
            
            if isinstance(response.text, bytes):
                data = pl.read_csv(io.StringIO(response.text.decode()))
            else:
                data = pl.read_csv(io.StringIO(response.text))
            
        elif request.dataset_source.source_type == "upload":
            if not request.dataset_source.file_content:
                raise HTTPException(
                    status_code=400,
                    detail="File content must be provided when source_type is 'upload'"
                )
            # Decode base64 file content
            try:
                file_content = base64.b64decode(request.dataset_source.file_content)
                data = pl.read_csv(io.BytesIO(file_content))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode file content: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid source_type. Must be 'url' or 'upload'"
            )

        # Validate dataset columns
        required_columns = ["intent", "text"]
        if not all(col in data.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"Dataset must contain columns: {required_columns}"
            )

        # Validate intents in dataset match requested intents
        unique_intents = set(data['intent'].unique().to_list())
        if not unique_intents.issubset(set(request.intents)):
            raise HTTPException(
                status_code=400,
                detail="Dataset contains intents not specified in the model configuration"
            )

        # Create training config
        training_config = request.training_config or TrainingConfig()
        if request.model_name:
            training_config.base_model_name = request.model_name
            
        # Train the model with config
        model, intents, tokenizer = train_intent_classifier(data, training_config)
        run_id = package_model(model, intents, tokenizer)
        
        return TrainingResponse(
            model_id=run_id,
            status="success",
            message="Model trained successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )

@app.post("/model/{model_id}/predict")
def predict(model_id: str, text: str) -> dict:
    """
    Generate intent predictions for a given text using a specified model.
    
    This endpoint provides flexible model loading and prediction:
    1. Attempts to load the model from the registry using the model_id as name
    2. If not found, attempts to load directly from MLflow run
    3. Processes the input text and generates confidence scores for all intents
    
    The prediction process:
    1. Loads the specified model (registered or from run)
    2. Preprocesses the input text using the model's tokenizer
    3. Generates confidence scores for all possible intents
    4. Returns scores as a normalized probability distribution
    
    Parameters:
        model_id (str): ID of the model to use, can be either:
            - A registered model name (loads latest version)
            - An MLflow run ID (loads specific run)
        text (str): Input text to classify (should be non-empty)
    
    Returns:
        dict: Dictionary containing:
            - Keys: All possible intent labels
            - Values: Confidence scores (0.0 to 1.0) for each intent
            Example: {"intent1": 0.8, "intent2": 0.15, "intent3": 0.05}
            
    Raises:
        HTTPException(404): If:
            - No registered model exists with the specified name
            - No run exists with the specified ID
            - Model exists but is not a valid intent classifier
        HTTPException(500): If prediction fails due to:
            - Model loading errors
            - Invalid model format
            - Text preprocessing failures
            - Resource constraints
    
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
                loaded_model = mlflow.pyfunc.load_model(f"runs:/{model_id}/intent_model")
            except mlflow.exceptions.MlflowException:
                raise HTTPException(
                    status_code=404,
                    detail=f"No model found with ID {model_id} (tried both registered models and runs)"
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

