# Lightweight CPU-only inference image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app

# CPU wheels of torch are far smaller than the default CUDA ones
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY serving/ serving/
COPY artifacts/model.pt artifacts/model.pt

# Run as a non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')" || exit 1
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT}"]
