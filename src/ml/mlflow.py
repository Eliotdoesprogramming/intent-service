import os

import mlflow
import pandas as pd
import polars as pl
import torch


class IntentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import DistilBertTokenizer

        """Load the model artifacts"""
        self.model = torch.load(
            os.path.join(context.artifacts["model_path"], "model.pth"),
            weights_only=False,
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            context.artifacts["tokenizer_path"]
        )
        self.intent_labels = torch.load(
            os.path.join(context.artifacts["model_path"], "intent_labels.pth"),
            weights_only=False,
        )
        self.model.eval()

    def predict(self, context, model_input):
        # Tokenize the input texts
        inputs = self.tokenizer(
            model_input["text"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

        # Convert to numpy for easier handling
        probs = probabilities.numpy()

        # Create a list of dictionaries containing intent labels and their scores
        results = []
        for prob in probs:
            intent_scores = {
                intent: float(score) for intent, score in zip(self.intent_labels, prob)
            }
            results.append(intent_scores)

        return results


def log_dataset(
    dataframe: pl.DataFrame, dataset_name: str = "training_data", source=None
):
    """
    Log the dataset to MLflow.

    Args:
        dataframe: Polars DataFrame to log
        dataset_name: Name for the dataset in MLflow
    """
    # Convert to pandas for MLflow compatibility
    pandas_df = dataframe.to_pandas()

    # Create MLflow dataset
    dataset = mlflow.data.from_pandas(
        pandas_df, targets="intent", source=source, name=dataset_name
    )
    # Log the dataset to MLflow
    mlflow.log_input(dataset, context="training")
    return dataset


def package_model(model, intents, tokenizer):
    # After training is complete:
    if not mlflow.active_run():
        mlflow.start_run()

    # Save model artifacts
    artifact_path = "model"
    os.makedirs(artifact_path, exist_ok=True)

    # Save the model
    torch.save(model, os.path.join(artifact_path, "model.pth"))

    # Save intent labels
    torch.save(intents, os.path.join(artifact_path, "intent_labels.pth"))

    # Save tokenizer
    tokenizer.save_pretrained(os.path.join(artifact_path, "tokenizer"))

    # Create an example input
    example_input = pd.DataFrame({"text": ["example text for signature"]})

    # Define input and output schema
    from mlflow.types.schema import ColSpec, Schema

    input_schema = Schema([ColSpec("string", "text")])
    output_schema = Schema([
        ColSpec("double", "scores"),  # Probability scores for each intent
        ColSpec("string", "predicted_intent"),  # Predicted intent label
    ])
    signature = mlflow.models.signature.ModelSignature(
        inputs=input_schema, outputs=output_schema
    )

    # Log the model with its artifacts and signature
    mlflow.pyfunc.log_model(
        artifact_path="intent_model",
        python_model=IntentModel(),
        artifacts={
            "model_path": artifact_path,
            "tokenizer_path": os.path.join(artifact_path, "tokenizer"),
        },
        input_example=example_input,
        signature=signature,
    )
    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()

    return run_id
