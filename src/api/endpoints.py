from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import warnings as pydantic_warnings
from typing import Optional
import polars as pl
from intent_service.schema import IntentModel
from intent_service.ml.train import train_intent_classifier
import mlflow
import pandas as pd
import requests
from io import StringIO
import warnings
from contextlib import contextmanager

app = FastAPI()

@contextmanager
def suppress_pydantic_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pydantic_warnings.PydanticDeprecatedSince20)
        yield

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

@app.post("/train")
async def train_model(
    dataset_url: Optional[str] = None,
    file: Optional[UploadFile] = File(None)
) -> str:
    if not dataset_url and not file:
        raise HTTPException(
            status_code=400, 
            detail="Either dataset_url or file must be provided"
        )
    
    try:
        # Handle URL download
        if dataset_url:
            response = requests.get(dataset_url)
            response.raise_for_status()
            df = pl.read_csv(StringIO(response.text))
        
        # Handle file upload
        elif file:
            contents = await file.read()
            df = pl.read_csv(StringIO(contents.decode()))
        
        # Validate required columns
        required_columns = {'intent', 'text'}
        if not required_columns.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"Dataset must contain columns: {required_columns}"
            )
        
        # Train the model and get the run ID
        run_id = train_intent_classifier(df)
        return run_id
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_id}/predict")
def predict(model_id: str, text: str) -> dict:
    try:
        # Load the model using the model_id as run_id
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{model_id}/intent_model")
        
        # Create a pandas DataFrame with the input text
        test_data = pd.DataFrame({"text": [text]})
    
        # Get prediction
        prediction = loaded_model.predict(test_data)
        
        return prediction[0]  # First element since we only predicted one text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

