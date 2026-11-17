
# Base application image
FROM python:3.14.10-slim-bookworm AS runtime

WORKDIR /app

# Install application dependencies
COPY requirements-runtime.txt .

RUN python -m pip install --no-cache-dir -r requirements-runtime.txt

# Copy application source code
COPY rag_qa.py .
COPY retrieve_hybrid.py .
COPY equation_utils.py .
COPY ollama_client.py .
COPY logging_config.py .
COPY config.py .

# Default application command
CMD ["python", "rag_qa.py"]


# Testing image
FROM runtime AS test

RUN python -m pip install --no-cache-dir pytest==9.2.0

# GPU diagnostic modules
COPY gpu_telemetry.py .
COPY gpu_tool_dispatcher.py .

COPY tests/ ./tests/

CMD ["python", "-m", "pytest", "tests", "-v"]


# Production image
FROM runtime AS production