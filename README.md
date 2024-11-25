# Intent Service

A Python-based service for processing and handling intents through a REST API.

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
- **Mypy**: For static type checking
- **Pytest**: For unit testing

Install development dependencies:
```bash
uv pip install -r requirements-dev.txt
```

### Running Code Quality Checks

```bash
# Run linting
ruff check .

# Run type checking
mypy .

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

### Process Intent

```http
POST /api/v1/process
Content-Type: application/json

{
    "text": "your input text",
    "context": {
        "user_id": "optional_user_id",
        "session_id": "optional_session_id"
    }
}
```

Response:
```json
{
    "intent": "detected_intent",
    "confidence": 0.95,
    "entities": [
        {
            "type": "entity_type",
            "value": "entity_value",
            "confidence": 0.9
        }
    ]
}
```

### Health Check

```http
GET /api/v1/health
```

Response:
```json
{
    "status": "healthy",
    "version": "1.0.0"
}
```

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

Create a `.env` file in the root directory with the following variables:

```env
DEBUG=True
LOG_LEVEL=INFO
API_KEY=your_api_key
```

## Running the Service

Development mode:
```bash
uvicorn app.main:app --reload
```

Production mode:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
