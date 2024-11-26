# Intent Service

# Intent-Service: Simplifying Fine-Tuning of Encoder Models for Classification

The intent-service is a tool designed to streamline the process of fine-tuning encoder-based models, such as BERT, for classification tasks. Specifically, this project focuses on simplifying the training of models for intent classification, which is a critical task in natural language processing (NLP) applications such as chatbots, virtual assistants, and other conversational AI systems.

## Background
Encoder models like BERT (Bidirectional Encoder Representations from Transformers) have revolutionized the way we process and understand language. These models are pre-trained on vast amounts of text data and can be fine-tuned to perform a wide range of downstream tasks with minimal effort. One of the most common applications of these models is intent classification—the task of determining the user's intent based on their input text.

Intent classification plays a central role in conversational AI systems, such as Google Assistant, Siri, Alexa, and countless custom chatbot solutions. By understanding the user's intent (e.g., "set an alarm," "get the weather," "play music"), these systems can trigger appropriate actions or provide relevant responses.

However, fine-tuning these models for intent classification can be challenging. It requires a well-organized approach to dataset preparation, hyperparameter tuning, and model optimization. Intent Classifier aims to simplify this process, making it easier for developers to deploy high-performance intent classification models for their applications.


## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. To get started:

1. Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:

```bash
git clone https://github.com/yourusername/intent-service.git
cd intent-service
```

3. Create a virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

uv pip install -r requirements.txt
```

## Development Setup

### Code Quality Tools

We use several tools to maintain code quality:

- **Ruff**: For fast Python linting and formatting
- **Pytest**: For unit testing

Install development dependencies:

```bash
uv pip install -r requirements-dev.txt
```

### Running Code Quality Checks

```bash
# Run linting
ruff check .

# Run tests
pytest
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality before commits. To set up:

```bash
pre-commit install
```

## API Usage

The service provides a REST API for intent processing. Here are the main endpoints:

### Model Management

#### Get Model Information
```bash
GET /model/{model_id}
```
Retrieves detailed information about a specific model. The `model_id` can be either a registered model name or MLflow run ID.

#### Search Models
```bash
POST /model/search
```
Search for registered models based on various criteria:
```json
{
  "tags": {"version": "1.0.0"},
  "intents": ["greeting", "farewell"],
  "name_contains": "bert",
  "limit": 10
}
```

#### Register Model
```bash
POST /model/register
```
Register an existing MLflow run as a named model:
```json
{
  "mlflow_run_id": "run_123",
  "name": "intent-classifier-v1",
  "description": "Intent classifier using DistilBERT",
  "tags": {
    "version": "1.0.0",
    "author": "team"
  }
}
```

### Training

#### Train New Model
```bash
POST /model/train
```
Train a new intent classification model:
```json
{
  "intents": ["greeting", "farewell", "help"],
  "dataset_source": {
    "source_type": "url",
    "url": "https://example.com/dataset.csv"
  },
  "model_name": "distilbert-base-uncased",
  "experiment_name": "intent-training",
  "training_config": {
    "num_epochs": 5,
    "batch_size": 32,
    "learning_rate": 5e-5
  }
}
```

### Prediction

#### Generate Predictions
```bash
POST /model/{model_id}/predict?text=Hello%20there
```
Generate intent predictions for input text. Returns confidence scores for each intent:
```json
{
  "greeting": 0.85,
  "farewell": 0.10,
  "help": 0.05
}
```

### API Documentation

Full API documentation is available at `/docs` when running the service. This provides an interactive Swagger UI where you can:
- View detailed endpoint specifications
- Try out API calls directly
- See request/response schemas
- Access example payloads

## Development Workflow

1. Create a new branch for your feature/fix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure all tests pass:

   ```bash
   pytest
   ```

3. Run code quality checks:

   ```bash
   ruff check .
   mypy .
   ```

4. Commit your changes:

   ```bash
   git commit -m "feat: add your feature description"
   ```

5. Push your changes and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Environment Variables

Create a `.env` file in the root directory based on the provided `.env.example`. Here are the available configuration options:

### Application Settings
```env
DEBUG=True
LOG_LEVEL=INFO
API_KEY=your_api_key_here
ENVIRONMENT=dev  # Options: dev, prod
VSCODE_DEBUGGER=False
```

### Server Settings
```env
HOST=0.0.0.0
PORT=8000
```

### MLflow Settings
```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_TRACKING_USERNAME=mlflow
MLFLOW_TRACKING_PASSWORD=mlflow123
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000  # For MinIO/S3 artifact storage
MLFLOW_ARTIFACT_ROOT=s3://mlflow/artifacts
AWS_ACCESS_KEY_ID=minioadmin          # For MinIO/S3 access
AWS_SECRET_ACCESS_KEY=minioadmin123   # For MinIO/S3 access
MLFLOW_EXPERIMENT_NAME=intent-service  # Default experiment name
```

### Model Settings
```env
DEFAULT_MODEL_NAME=distilbert-base-uncased
MAX_SEQUENCE_LENGTH=128
BATCH_SIZE=32
```

To get started:
```bash
cp .env.example .env
```
Then edit the `.env` file with your specific configuration values.

## Running the Service

Development mode:

```bash
uvicorn app.main:app --reload
```

Production mode:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## CLI Usage

The service provides a command-line interface for model management and server operations:

### Starting the Server

```bash
# Development mode (auto-reload enabled)
intent-cli serve

# Production mode
intent-cli serve --environment prod --workers 4

# Custom configuration
intent-cli serve --port 9000 --host 127.0.0.1
```

### Model Management

Train a new model:

```bash
intent-cli train \
    --dataset-path data/training.csv \
    --experiment-name "my-experiment" \
    --num-epochs 5
```

Register a trained model:

```bash
intent-cli register \
    <run_id> \
    "my-model-name" \
    --description "Description of the model" \
    --tags '{"version": "1.0.0", "author": "team"}'
```

Search for models:

```bash
intent-cli search \
    --name-contains "bert" \
    --tags '{"version": "1.0.0"}' \
    --intents "greeting,farewell"
```

Get model information:

```bash
intent-cli info <model_id>
```

Make predictions:

```bash
intent-cli predict <model_id> "your text here"
```

### CLI Options

Each command supports various options. Use the `--help` flag to see detailed documentation:

```bash
intent-cli --help  # Show all commands
intent-cli serve --help  # Show options for serve command
intent-cli train --help  # Show options for train command
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
