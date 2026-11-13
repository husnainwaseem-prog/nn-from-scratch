# Neural Network from Scratch — runs the NumPy network in a container.
# NOTE: the image is not built in this sandbox (no Docker daemon here);
# it was statically checked. Build and run on a machine with Docker:
#   docker build -t nn-from-scratch .
#   docker run --rm nn-from-scratch
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY README.md REPORT.md ./

# Training downloads MNIST (~11 MB) on first run and caches it in data/.
RUN mkdir -p data results/plots

CMD ["python", "-m", "src.cli", "train"]
