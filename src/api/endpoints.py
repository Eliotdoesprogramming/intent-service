from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import polars as pl
from schema import Dataset, IntentModel
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
