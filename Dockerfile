# Use Python slim image as the base
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
COPY src/ ./src/
RUN uv sync

# Copy only the necessary source files
# Expose the port the app runs on

EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "src/run_service.py"]