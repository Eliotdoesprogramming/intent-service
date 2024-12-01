import shutil
from pathlib import Path

import mlflow
import yaml
from huggingface_hub import HfApi, create_repo, whoami


def create_model_card(run_id: str, run: mlflow.entities.Run) -> str:
    """Create a model card with metadata for Hugging Face Hub."""
    # Extract metrics and parameters from the run
    metrics = run.data.metrics
    params = run.data.params

    # Create metadata dictionary
    metadata = {
        "language": ["en"],
        "license": "mit",
        "library_name": "mlflow",
        "tags": ["intent-classification", "text-classification", "mlflow"],
        "datasets": ["custom"],
        "metrics": metrics,
        "model-index": [
            {
                "name": "Intent Classification Model",
                "results": [
                    {
                        "task": {
                            "type": "text-classification",
                            "subtype": "intent-classification",
                        },
                        "metrics": [
                            {"type": metric, "value": value}
                            for metric, value in metrics.items()
                        ],
                    }
                ],
            }
        ],
    }

    # Create the model card content
    model_card = f"""---
{yaml.dump(metadata, sort_keys=False)}
---

# Intent Classification Model

This is an intent classification model trained using MLflow and uploaded to the Hugging Face Hub.

## Model Details

- **Model Type:** Intent Classification
- **Framework:** MLflow
- **Run ID:** {run_id}

## Training Details

### Parameters
```yaml
{yaml.dump(params, sort_keys=False)}
```

### Metrics
```yaml
{yaml.dump(metrics, sort_keys=False)}
```

## Usage

This model can be used to classify intents in text. 
It was trained using MLflow and can be loaded using the MLflow model registry.
"""
    return model_card


def upload_model_to_hub(
    run_id: str,
    repo_name: str,
    hf_token: str,
    organization: str = None,
    private: bool = False,
    commit_message: str = "Upload intent classification model",
):
    """
    Upload an MLflow model to Hugging Face Hub.

    Args:
        run_id: MLflow run ID containing the model
        repo_name: Name for the Hugging Face repository
        hf_token: Hugging Face API token
        organization: Optional organization name to upload to
        private: Whether to create a private repository
        commit_message: Commit message for the upload

    Returns:
        str: URL of the model on Hugging Face Hub
    """
    # Initialize Hugging Face API
    api = HfApi(token=hf_token)

    # Get user info
    user_info = whoami(token=hf_token)
    username = user_info.get("name")

    # Get the MLflow run and model artifacts
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    # Create a temporary directory to prepare files
    temp_dir = Path("temp_model_upload")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Download the MLflow model
        local_path = client.download_artifacts(run_id, "intent_model", str(temp_dir))

        # Create the repository ID
        if organization:
            repo_id = f"{organization}/{repo_name}"
        else:
            repo_id = f"{username}/{repo_name}"

        print(f"Using repository ID: {repo_id}")

        # Create the repository
        try:
            repo_url = create_repo(
                repo_id=repo_id,
                token=hf_token,
                private=private,
                repo_type="model",
                exist_ok=True,
            )
            print(f"Repository created/accessed at: {repo_url}")

            # Create and upload README with model card
            readme_path = Path(local_path) / "README.md"
            model_card_content = create_model_card(run_id, run)
            with open(readme_path, "w") as f:
                f.write(model_card_content)

            print("Uploading README.md with model card...")
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                token=hf_token,
                commit_message="Add model card with metadata",
            )

            # Upload all files in the model directory
            print("Uploading model files...")
            for item in Path(local_path).glob("**/*"):
                if item.is_file() and item.name != "README.md":
                    relative_path = str(item.relative_to(local_path))
                    print(f"Uploading {relative_path}...")
                    api.upload_file(
                        path_or_fileobj=str(item),
                        path_in_repo=relative_path,
                        repo_id=repo_id,
                        token=hf_token,
                        commit_message=f"Upload {relative_path}",
                    )

            print("Upload completed successfully!")

        except Exception as e:
            print(f"Error during repository operations: {e}")
            raise

        # Return the model URL
        return f"https://huggingface.co/{repo_id}"

    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
