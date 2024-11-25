import polars as pl
import mlflow
import pandas as pd
import requests
import base64
import io
from fastapi import FastAPI, HTTPException
from schema import IntentModel, TrainingRequest, TrainingResponse, TrainingConfig
from ml.train import package_model, train_intent_classifier
app = FastAPI()

# Endpoint to list available models
@app.get("/models")
def get_models():
    pass

# Endpoint to update model information
@app.put("/models/{model_id}")
def update_model(model_id: int, model: IntentModel):
    # Find the model with the given ID and update its information
    pass

# Endpoint to create and train a model
@app.post("/models")
def create_model(model: IntentModel):
    # Add the new model to the list of models
    pass

# Endpoint to delete a model
@app.delete("/models/{model_id}")
def delete_model(model_id: int):
    # Find the model with the given ID and remove it from the list of models
    pass

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest) -> dict:
    try:
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
            training_config.model_name = request.model_name
            
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

@app.post("/models/{model_id}/predict")
def predict(model_id: str, text: str) -> dict:
    try:
    # Load the model using the model_id as run_id
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{model_id}/intent_model")
        
        # Create a pandas DataFrame with the input text
        test_data = pd.DataFrame({"text": [text]})
        
        # Get prediction
        prediction = loaded_model.predict(test_data)
        
        # Return the prediction dictionary (contains all intent scores)
        return prediction[0]  # First element since we only predicted one text
    except mlflow.exceptions.MlflowException as e:
        if "Run" in str(e) and "not found" in str(e):
            raise HTTPException(status_code=400, detail=f"Model ID {model_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

