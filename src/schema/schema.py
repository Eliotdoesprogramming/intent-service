from pydantic import BaseModel, HttpUrl
from typing import Dict, List, Optional, Any, Literal

class RegisterModelRequest(BaseModel):
    mlflow_run_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Dict[str, Any] = {}
    class Config:
        arbitrary_types_allowed = True

class DatasetSource(BaseModel):
    source_type: Literal["url", "upload"]
    url: Optional[HttpUrl] = None
    file_content: Optional[str] = None  # Base64 encoded content

class TrainingConfig(BaseModel):
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 5e-5
    base_model_name: str = 'distilbert-base-uncased'
    max_length: int = 128  # Max sequence length for tokenizer
    warmup_steps: int = 0
    weight_decay: float = 0.01
    early_stopping_patience: Optional[int] = None
    validation_split: Optional[float] = None

class TrainingRequest(BaseModel):
    intents: List[str]
    dataset_source: DatasetSource
    model_name: Optional[str] = None
    experiment_name: Optional[str] = None
    training_config: TrainingConfig = TrainingConfig()

class TrainingResponse(BaseModel):
    model_id: str
    status: Literal["success", "failed"]
    message: str 

class ModelSearchRequest(BaseModel):
    pass
