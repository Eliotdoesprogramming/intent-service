from pydantic import BaseModel
import polars as pl
from typing import Optional, List, Any, Dict

class Dataset(BaseModel):
    id: int
    collection_name: str
    description: str
    data: Optional[pl.DataFrame]
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

class IntentModel(BaseModel):
    id: int
    name: str
    description: str
    intents: List[str]
    dataset: Dataset
    extra_data: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True