from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import polars as pl
from schema import Dataset, IntentModel
import mlflow
import pandas as pd
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

@app.post("/train")
def train_model(model: IntentModel)->str:
    # Train the model and return the model ID
    pass

@app.post("/models/{model_id}/predict")
def predict(model_id: str, text: str) -> dict:
    # Load the model using the model_id as run_id
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{model_id}/intent_model")
        
        # Create a pandas DataFrame with the input text
        test_data = pd.DataFrame({"text": [text]})
        
        # Get prediction
        prediction = loaded_model.predict(test_data)
        
        # Return the prediction dictionary (contains all intent scores)
        return prediction[0]  # First element since we only predicted one text
        

