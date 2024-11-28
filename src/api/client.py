import base64
from typing import Dict, Optional

import httpx

from schema import RegisterModelRequest, TrainingConfig, TrainingRequest


class IntentServiceClient:
    """Client for interacting with the Intent Service API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            timeout=None
        )  # No timeout for long-running operations

    def predict(self, model_id: str, text: str) -> Dict[str, float]:
        """Make a prediction using a model."""
        response = self.client.post(
            f"{self.base_url}/model/{model_id}/predict",
            params={"text": text},
        )
        response.raise_for_status()
        return response.json()

    def train(
        self,
        dataset_path: str,
        training_config: Optional[TrainingConfig] = None,
        experiment_name: Optional[str] = None,
    ) -> str:
        """Train a new model using a local dataset."""
        # Read and encode the dataset
        with open(dataset_path, "rb") as f:
            file_content = base64.b64encode(f.read()).decode()

        # Create training request
        request = TrainingRequest(
            dataset_source={
                "source_type": "upload",
                "file_content": file_content,
            },
            training_config=training_config,
            experiment_name=experiment_name,
        )

        response = self.client.post(
            f"{self.base_url}/model/train", json=request.model_dump()
        )
        response.raise_for_status()
        return response.json()["model_id"]

    def register_model(
        self,
        run_id: str,
        name: str,
        description: Optional[str] = None,
        tags: Optional[Dict] = None,
    ) -> Dict:
        """Register a trained model."""
        request = RegisterModelRequest(
            mlflow_run_id=run_id,
            name=name,
            description=description,
            tags=tags or {},
        )

        response = self.client.post(
            f"{self.base_url}/model/register",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return response.json()

    def get_model_info(self, model_id: str) -> Dict:
        """Get information about a model."""
        response = self.client.get(f"{self.base_url}/model/{model_id}")
        response.raise_for_status()
        return response.json()

    def search_models(self, query: Dict) -> Dict:
        """Search for models."""
        response = self.client.post(f"{self.base_url}/model/search", json=query)
        response.raise_for_status()
        return response.json()
