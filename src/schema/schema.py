from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class RegisterModelRequest(BaseModel):
    mlflow_run_id: str = Field(..., description="MLflow run ID for the trained model")
    name: Optional[str] = Field(
        None,
        description="Display name for the model",
        examples=["intent-classifier-v1"],
    )
    description: Optional[str] = Field(
        None, description="Detailed description of the model"
    )
    tags: Dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata tags for the model"
    )

    class Config:
        arbitrary_types_allowed = True


class DatasetSource(BaseModel):
    source_type: Literal["url", "upload"] = Field(
        ..., description="Type of data source"
    )
    url: Optional[HttpUrl] = Field(None, description="URL to fetch dataset from")
    file_content: Optional[str] = Field(
        None, description="Base64 encoded dataset content"
    )


class TrainingConfig(BaseModel):
    num_epochs: int = Field(
        10, description="Number of training epochs", ge=1, examples=[5, 10, 20]
    )
    batch_size: int = Field(
        32, description="Training batch size", ge=1, examples=[16, 32, 64]
    )
    learning_rate: float = Field(
        5e-5,
        description="Learning rate for optimization",
        gt=0,
        examples=[1e-5, 3e-5, 5e-5],
    )
    base_model_name: str = Field(
        "distilbert-base-uncased",
        description="Base model to use for fine-tuning",
        examples=[
            "bert-base-uncased",
            "roberta-base",
            "distilbert-base-uncased",
            "albert-base-v2",
            "xlm-roberta-base",
            "microsoft/deberta-base",
            "google/electra-base-discriminator",
        ],
    )
    max_length: int = Field(
        128,
        description="Maximum sequence length for tokenizer",
        ge=1,
        examples=[128, 256, 512],
    )
    warmup_steps: int = Field(0, description="Number of warmup steps", ge=0)
    weight_decay: float = Field(0.01, description="Weight decay for optimization", ge=0)
    early_stopping_patience: Optional[int] = Field(
        None,
        description="Number of epochs to wait before early stopping",
        examples=[3, 5],
    )
    validation_split: Optional[float] = Field(
        None,
        description="Fraction of data to use for validation",
        ge=0.0,
        le=1.0,
        examples=[0.1, 0.2],
    )

    @field_validator("base_model_name")
    def validate_model_name(cls, v):
        # Optional: Add validation if you want to restrict to specific models
        return v


class TrainingRequest(BaseModel):
    intents: Optional[List[str]] = None
    dataset_source: DatasetSource
    model_name: Optional[str] = None
    experiment_name: Optional[str] = None
    training_config: TrainingConfig = TrainingConfig()


class TrainingResponse(BaseModel):
    model_id: str
    status: Literal["success", "failed"]
    message: str


class ModelSearchRequest(BaseModel):
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Tag key-value pairs to filter models by",
        examples=[{"framework": "pytorch", "task": "intent"}],
    )
    intents: Optional[List[str]] = Field(
        None,
        description="List of required intents that the model must support",
        examples=[["greeting", "farewell", "help_request"]],
    )
    name_contains: Optional[str] = Field(
        None,
        description="Substring to match in model names for filtering",
        examples=["intent", "bert"],
    )
    limit: Optional[int] = Field(
        100,
        description="Maximum number of models to return in the response",
        ge=1,
        le=1000,
        examples=[10, 50, 100],
    )
