import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from api.client import IntentServiceClient
from schema import TrainingConfig

app = typer.Typer(help="CLI for intent service model management")
console = Console()

# Global API client instance
_api_client = None


def get_api_client() -> IntentServiceClient:
    """Get or create the API client instance."""
    global _api_client
    if _api_client is None:
        # Get API URL from environment variable or use default
        api_url = os.getenv("INTENT_SERVICE_URL", "http://localhost:8000")
        _api_client = IntentServiceClient(base_url=api_url)
    return _api_client


def set_api_client(client: IntentServiceClient) -> None:
    """Set the API client instance (used for testing)."""
    global _api_client
    _api_client = client


def get_api():
    """Lazy load API server utilities only when needed"""
    import uvicorn

    from api import app

    return app, uvicorn


@app.callback()
def common_options(
    api_url: Optional[str] = typer.Option(
        "http://localhost:8000",
        "--api-url",
        "-u",
        help="API server URL. Can also be set via INTENT_SERVICE_URL environment variable.",
        envvar="INTENT_SERVICE_URL",
    ),
):
    """Common options for all commands."""
    if api_url:
        set_api_client(IntentServiceClient(base_url=api_url))


@app.command()
def register(
    run_id: str = typer.Argument(..., help="MLflow run ID for the trained model"),
    name: str = typer.Argument(..., help="Name for the registered model"),
    description: Optional[str] = typer.Option(None, help="Description of the model"),
    tags: Optional[str] = typer.Option(None, help="JSON string of model tags"),
):
    """Register a trained model in MLflow."""
    try:
        # Parse tags if provided
        tag_dict = json.loads(tags) if tags else {}

        # Register the model using the API
        result = get_api_client().register_model(
            run_id=run_id,
            name=name,
            description=description,
            tags=tag_dict,
        )

        print(
            f"[green]Successfully registered model {name} (version: {result['version']})[/green]"
        )

    except Exception as e:
        print(f"[red]Error registering model: {str(e)}[/red]")
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
        # Check if dataset file exists
        if not dataset_path.exists():
            msg = f"Error: Dataset file not found: {str(dataset_path).strip()}"
            print(f"[red]{msg}[/red]")
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

        # Train the model using the API
        print("[yellow]Training model...[/yellow]")
        run_id = get_api_client().train(
            dataset_path=str(dataset_path),
            training_config=training_config,
            experiment_name=experiment_name,
        )

        print(f"[green]Successfully trained model. Run ID: {run_id}[/green]")

    except Exception as e:
        print(f"[red]Error training model: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def predict(
    model_id: str = typer.Argument(..., help="Model ID (name or run ID)"),
    text: str = typer.Argument(..., help="Text to classify"),
):
    """Make predictions with a model."""
    try:
        # Get prediction using the API
        result = get_api_client().predict(model_id=model_id, text=text)

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
def info(model_id: str = typer.Argument(..., help="Model ID (name or run ID)")):
    """Get information about a model."""
    try:
        # Get model info using the API
        model_info = get_api_client().get_model_info(model_id)

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
        print(f"[red]Error retrieving model info: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def search(
    name_contains: Optional[str] = typer.Option(None, help="Filter by model name"),
    tags: Optional[str] = typer.Option(None, help="JSON string of tags to filter by"),
    intents: Optional[str] = typer.Option(None, help="Comma-separated list of intents"),
):
    """Search for registered models."""
    try:
        # Parse search parameters
        query = {}
        if name_contains:
            query["name_contains"] = name_contains
        if tags:
            query["tags"] = json.loads(tags)
        if intents:
            query["intents"] = intents.split(",")

        # Search models using the API
        results = get_api_client().search_models(query)

        # Display results
        if not results:
            print("[yellow]No models found matching the search criteria.[/yellow]")
            return

        table = Table(title="Search Results")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="magenta")
        table.add_column("Description")
        table.add_column("Tags")
        table.add_column("Intents")

        for model in results:
            table.add_row(
                model["name"],
                str(model.get("version", "N/A")),
                model.get("description", ""),
                json.dumps(model.get("tags", {})),
                ", ".join(model.get("intents", [])),
            )

        console.print(table)

    except Exception as e:
        print(f"[red]Error searching models: {str(e)}[/red]")
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
def upload_huggingface(
    run_id: str = typer.Argument(..., help="MLflow run ID containing the model"),
    repo_name: str = typer.Argument(..., help="Name for the Hugging Face repository"),
    organization: Optional[str] = typer.Option(
        None, help="Optional organization name to upload to"
    ),
    private: bool = typer.Option(False, help="Whether to create a private repository"),
    commit_message: str = typer.Option(
        "Upload intent classification model", help="Commit message for the upload"
    ),
):
    """Upload a model to Hugging Face Hub"""

    def get_hf_token() -> str:
        """Get Hugging Face token from environment or prompt user"""
        token = os.getenv("HF_TOKEN")
        if not token:
            try:
                token = typer.prompt(
                    "Please enter your Hugging Face token",
                    hide_input=True,
                    show_default=False,
                )
            except (typer.Abort, EOFError):
                print("[red]Token input aborted.[/red]")
                sys.exit(1)

            if not token:
                print("[red]Token cannot be empty.[/red]")
                sys.exit(1)
        return token

    try:
        # Get token
        token = get_hf_token()

        # Upload model
        print("[bold blue]Uploading model to Hugging Face Hub...[/bold blue]")
        result = get_api_client().upload_to_huggingface(
            run_id=run_id,
            repo_name=repo_name,
            hf_token=token,
            organization=organization,
            private=private,
            commit_message=commit_message,
        )

        print("[bold green]✓ Model successfully uploaded![/bold green]")
        print(f"[bold]Model URL:[/bold] {result['model_url']}")

    except Exception as e:
        print(f"[bold red]Error uploading model:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def list(
    limit: int = typer.Option(100, help="Maximum number of models to display"),
):
    """List all registered models."""
    try:
        # Get all models using the API
        results = get_api_client().get_models(limit=limit)

        # Display results
        if not results:
            print("[yellow]No models found.[/yellow]")
            return

        table = Table(title="Registered Models")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="magenta")
        table.add_column("Description")
        table.add_column("Tags")
        table.add_column("Intents")
        table.add_column("Last Updated", style="green")
        table.add_column("MLflow URI", style="blue", no_wrap=False)

        for model in results:
            # Format timestamp to readable date
            from datetime import datetime

            last_updated = datetime.fromtimestamp(
                model["last_updated_timestamp"] / 1000
            ).strftime("%Y-%m-%d %H:%M:%S")

            # Construct MLflow URI - prioritize models URI
            mlflow_uri = f"models:/{model['name']}/latest"
            if not model.get("version"):  # Only show run URI if model isn't registered
                if "run_info" in model and model["run_info"].get("run_id"):
                    mlflow_uri = f"runs:/{model['run_info']['run_id']}/intent_model"

            table.add_row(
                model["name"],
                str(model.get("version", "N/A")),
                model.get("description", ""),
                json.dumps(model.get("tags", {})),
                ", ".join(model.get("intents", [])),
                last_updated,
                mlflow_uri,
            )

        console.print(table)

    except Exception as e:
        print(f"[red]Error listing models: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
