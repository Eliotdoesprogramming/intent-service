import json
import os
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rich import print
from rich.console import Console
from rich.table import Table

from schema import RegisterModelRequest, TrainingConfig

app = typer.Typer(help="CLI for intent service model management")
console = Console()


def get_mlflow():
    """Lazy load MLflow only when needed"""
    import mlflow

    return mlflow


def get_train_utils():
    """Lazy load training utilities only when needed"""
    from ml.train import package_model, train_intent_classifier

    return package_model, train_intent_classifier


def get_api():
    """Lazy load API server utilities only when needed"""
    import uvicorn

    from api import app

    return app, uvicorn


@app.command()
def register(
    run_id: str = typer.Argument(..., help="MLflow run ID for the trained model"),
    name: str = typer.Argument(..., help="Name for the registered model"),
    description: Optional[str] = typer.Option(None, help="Description of the model"),
    tags: Optional[str] = typer.Option(None, help="JSON string of model tags"),
):
    """Register a trained model in MLflow."""
    try:
        mlflow = get_mlflow()  # Lazy load mlflow
        # Parse tags if provided
        tag_dict = json.loads(tags) if tags else {}

        # Create registration request
        request = RegisterModelRequest(
            mlflow_run_id=run_id,
            name=name,
            description=description,
            tags=tag_dict,
        )

        # Load the model to verify it exists
        try:
            loaded_model = mlflow.pyfunc.load_model(
                f"runs:/{request.mlflow_run_id}/intent_model"
            )
            intents = loaded_model._model_impl.python_model.intent_labels
            del loaded_model
        except mlflow.exceptions.MlflowException:
            print(
                f"[red]Error: No model found with run ID: {request.mlflow_run_id}[/red]"
            )
            raise typer.Exit(1)

        # Register the model
        model_uri = f"runs:/{request.mlflow_run_id}/intent_model"
        registered_model = mlflow.register_model(model_uri=model_uri, name=request.name)

        # Add description and tags
        client = mlflow.tracking.MlflowClient()
        client.update_registered_model(
            name=registered_model.name,
            description=request.description if request.description else None,
        )

        # Add intents as model tags
        for intent in intents:
            client.set_registered_model_tag(
                name=registered_model.name, key=f"intent_{intent}", value="true"
            )

        # Add any extra metadata as tags
        for key, value in request.tags.items():
            client.set_registered_model_tag(
                name=registered_model.name, key=key, value=str(value)
            )

        print(
            f"[green]Successfully registered model {name} (version {registered_model.version})[/green]"
        )

    except Exception as e:
        print(f"[red]Error registering model: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def search(
    name_contains: Optional[str] = typer.Option(None, help="Filter models by name"),
    tags: Optional[str] = typer.Option(None, help="JSON string of tags to filter by"),
    intents: Optional[str] = typer.Option(
        None, help="Comma-separated list of required intents"
    ),
    limit: Optional[int] = typer.Option(100, help="Maximum number of results"),
):
    """Search for registered models."""
    try:
        mlflow = get_mlflow()  # Lazy load mlflow
        client = mlflow.tracking.MlflowClient()
        results = []

        # Parse filters
        tag_dict = json.loads(tags) if tags else {}
        intent_list = intents.split(",") if intents else []

        # Get all registered models
        registered_models = client.search_registered_models()

        for rm in registered_models:
            match = True
            model_info = {
                "name": str(rm.name),
                "description": str(rm.description or ""),
                "creation_timestamp": str(rm.creation_timestamp),
                "last_updated_timestamp": str(rm.last_updated_timestamp),
            }

            # Get latest version
            latest_versions = client.get_latest_versions(rm.name, stages=["None"])
            if latest_versions:
                model_info["version"] = str(latest_versions[0].version)

            # Get all tags
            tags = dict(rm.tags) if rm.tags else {}

            # Extract intents from tags
            model_intents = [
                tag.replace("intent_", "")
                for tag in tags.keys()
                if tag.startswith("intent_")
            ]
            model_info["intents"] = model_intents

            # Filter non-intent tags
            model_info["tags"] = {
                k: str(v) for k, v in tags.items() if not k.startswith("intent_")
            }

            # Apply name filter if specified
            if name_contains and name_contains.lower() not in str(rm.name).lower():
                match = False

            # Apply tag filters if specified
            if tag_dict:
                for key, value in tag_dict.items():
                    if key not in tags or str(tags[key]) != str(value):
                        match = False
                        break

            # Apply intent filters if specified
            if intent_list:
                if not set(intent_list).issubset(set(model_intents)):
                    match = False

            if match:
                results.append(model_info)

            # Apply result limit
            if limit and len(results) >= limit:
                break

        # Create a table to display results
        table = Table(title="Model Search Results")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Intents", style="yellow")
        table.add_column("Tags", style="blue")

        for result in results:
            table.add_row(
                result["name"],
                result.get("version", "N/A"),
                result.get("description", ""),
                ", ".join(result["intents"]),
                json.dumps(result["tags"], indent=2),
            )

        console.print(table)

    except Exception as e:
        print(f"[red]Error searching models: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    dataset_path: Path = typer.Argument(..., help="Path to the training dataset CSV"),
    model_name: Optional[str] = typer.Option(None, help="Base model name"),
    experiment_name: Optional[str] = typer.Option(None, help="MLflow experiment name"),
    num_epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Training batch size"),
    learning_rate: float = typer.Option(5e-5, help="Learning rate"),
    max_length: int = typer.Option(128, help="Maximum sequence length"),
):
    """Train a new intent classification model."""
    try:
        mlflow = get_mlflow()
        package_model, train_intent_classifier = get_train_utils()

        # Set experiment if provided
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        # Load dataset
        try:
            data = pl.read_csv(dataset_path)
        except Exception as e:
            print(f"[red]Error loading dataset: {str(e)}[/red]")
            raise typer.Exit(1)

        # Validate dataset columns
        required_columns = ["intent", "text"]
        if not all(col in data.columns for col in required_columns):
            print(f"[red]Error: Dataset must contain columns: {required_columns}[/red]")
            raise typer.Exit(1)

        # Create training config
        training_config = TrainingConfig(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
        )
        if model_name:
            training_config.base_model_name = model_name

        # Train the model
        print("[yellow]Training model...[/yellow]")
        model, intents, tokenizer = train_intent_classifier(data, training_config)
        run_id = package_model(model, intents, tokenizer)

        print(f"[green]Successfully trained model. Run ID: {run_id}[/green]")

    except Exception as e:
        print(f"[red]Error training model: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def info(model_id: str = typer.Argument(..., help="Model ID (name or run ID)")):
    """Get information about a model."""
    try:
        mlflow = get_mlflow()  # Lazy load mlflow
        client = mlflow.tracking.MlflowClient()
        model_info = {}

        try:
            # Try to get as registered model
            registered_model = client.get_registered_model(model_id)
            latest_version = client.get_latest_versions(model_id, stages=["None"])[0]

            # Basic model info
            model_info.update({
                "name": registered_model.name,
                "version": latest_version.version,
                "description": registered_model.description,
                "creation_timestamp": registered_model.creation_timestamp,
                "last_updated_timestamp": registered_model.last_updated_timestamp,
            })

            # Get all tags
            tags = registered_model.tags if registered_model.tags else {}

            # Extract intent labels from tags
            intents = [
                tag.replace("intent_", "")
                for tag in tags.keys()
                if tag.startswith("intent_")
            ]

            model_info["intents"] = intents
            model_info["tags"] = {
                k: v for k, v in tags.items() if not k.startswith("intent_")
            }

            # Add run info if available
            if latest_version.run_id:
                run = client.get_run(latest_version.run_id)
                model_info["run_info"] = {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                }

        except mlflow.exceptions.MlflowException:
            # Try to get as run ID instead
            try:
                run = client.get_run(model_id)
                model_info.update({
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                })

                # Load model to get intents
                model = mlflow.pyfunc.load_model(f"runs:/{model_id}/intent_model")
                model_info["intents"] = model._model_impl.python_model.intent_labels

            except mlflow.exceptions.MlflowException:
                print(f"[red]Error: No model found with ID: {model_id}[/red]")
                raise typer.Exit(1)

        # Create a table to display model info
        table = Table(title=f"Model Information: {model_id}")

        for key, value in model_info.items():
            if key != "run_info":
                table.add_row(key, str(value))

        if "run_info" in model_info:
            table.add_section()
            table.add_row("Run Information", "")
            for key, value in model_info["run_info"].items():
                table.add_row(f"  {key}", str(value))

        console.print(table)

    except Exception as e:
        print(f"[red]Error getting model info: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def predict(
    model_id: str = typer.Argument(..., help="Model ID (name or run ID)"),
    text: str = typer.Argument(..., help="Text to classify"),
):
    """Make predictions with a model."""
    try:
        mlflow = get_mlflow()  # Lazy load mlflow
        # First try loading as a registered model
        try:
            loaded_model = mlflow.pyfunc.load_model(f"models:/{model_id}/latest")
        except mlflow.exceptions.MlflowException:
            # If not found as registered model, try loading as a run
            try:
                loaded_model = mlflow.pyfunc.load_model(
                    f"runs:/{model_id}/intent_model"
                )
            except mlflow.exceptions.MlflowException:
                print(f"[red]Error: No model found with ID {model_id}[/red]")
                raise typer.Exit(1)

        # Create a pandas DataFrame with the input text
        import pandas as pd

        test_data = pd.DataFrame({"text": [text]})

        # Get prediction
        prediction = loaded_model.predict(test_data)
        result = prediction[0]  # First element since we only predicted one text

        # Create a table to display predictions
        table = Table(title="Intent Predictions")
        table.add_column("Intent", style="cyan")
        table.add_column("Confidence", style="magenta")

        for intent, confidence in result.items():
            table.add_row(intent, f"{confidence:.4f}")

        console.print(table)

    except Exception as e:
        print(f"[red]Error making prediction: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    environment: str = typer.Option(None, help="Environment (prod/dev)"),
):
    """Start the intent classification API server."""
    try:
        app, uvicorn = get_api()

        print(f"[yellow]Starting API server on {host}:{port}[/yellow]")

        if environment == "prod":
            # Production mode: No reload, direct app import
            uvicorn.run(
                "api:app",
                host=host,
                port=port,
                workers=workers,
            )
        elif os.getenv("VSCODE_DEBUGGER"):
            # VS Code debug mode: Direct app instance
            uvicorn.run(app, host=host, port=port)
        else:
            # Development mode: Auto-reload enabled
            uvicorn.run("api:app", host=host, port=port, reload=True, workers=workers)

    except Exception as e:
        print(f"[red]Error starting server: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def list():
    """List all registered models."""
    try:
        mlflow = get_mlflow()  # Lazy load mlflow
        client = mlflow.tracking.MlflowClient()

        # Get all registered models
        registered_models = client.search_registered_models()

        if not registered_models:
            print("[yellow]No registered models found[/yellow]")
            return

        # Create a table to display results
        table = Table(title="Registered Models")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Intents", style="yellow")
        table.add_column("Last Updated", style="blue")

        for rm in registered_models:
            # Get latest version
            latest_versions = client.get_latest_versions(rm.name, stages=["None"])
            version = latest_versions[0].version if latest_versions else "N/A"

            # Get intents from tags
            tags = rm.tags if rm.tags else {}
            intents = [
                tag.replace("intent_", "")
                for tag in tags.keys()
                if tag.startswith("intent_")
            ]

            table.add_row(
                rm.name,
                str(version),
                rm.description or "",
                ", ".join(intents),
                str(rm.last_updated_timestamp),
            )

        console.print(table)

    except Exception as e:
        print(f"[red]Error listing models: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
