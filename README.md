# MLOps PyTorch Pipeline

An end-to-end machine learning project for training and serving a PyTorch image classifier using a reproducible Git workflow, configuration-driven training, FastAPI, and Docker.

This repository currently implements:

- **Part A:** Repository setup and Git methodology
- **Part B:** PyTorch model training and inference API
- **Part C:** Docker containerization for training and serving

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Training the Model](#training-the-model)
- [Running the API](#running-the-api)
- [Testing the Endpoints](#testing-the-endpoints)
- [Running Tests](#running-tests)
- [Docker Training](#docker-training)
- [Docker Model Serving](#docker-model-serving)
- [Configuration](#configuration)
- [Structured Logging and Early Stopping](#structured-logging-and-early-stopping)
- [Git Methodology](#git-methodology)
- [Validation Results](#validation-results)
- [Troubleshooting](#troubleshooting)
- [Security and Repository Hygiene](#security-and-repository-hygiene)
- [Future Work](#future-work)
- [AI Assistance Disclosure](#ai-assistance-disclosure)

## Overview

The project trains a ResNet-18 image classifier on CIFAR-10 and saves the best model as a PyTorch checkpoint. A FastAPI service loads the checkpoint and exposes health and prediction endpoints. Separate Docker images provide reproducible environments for training and inference.

### Workflow

```text
CIFAR-10 dataset
       |
       v
PyTorch training pipeline
       |
       v
classifier_v1.pt checkpoint
       |
       v
FastAPI inference service
       |
       v
/health and /predict endpoints
```

### Docker workflow

```text
Host data directory ----------------> /app/data
                                          |
                                          v
                                 Training container
                                          |
                                          v
Host checkpoints directory <------ /app/checkpoints
                                          |
                                          v
                                  Serving container
                                          |
                                          v
                              HTTP API on port 8080
```

## Architecture

The solution separates training and inference responsibilities:

1. `src/dataset.py` downloads and prepares CIFAR-10 data.
2. `src/model.py` creates a CIFAR-10-compatible ResNet-18 classifier.
3. `src/train.py` reads YAML configuration, trains and validates the model, logs JSON events, applies early stopping, and saves the best checkpoint.
4. `src/serve.py` loads the checkpoint and exposes the inference API.
5. `docker/Dockerfile.train` provides the training environment.
6. `docker/Dockerfile.serve` provides a non-root serving environment with a health check.

## Features

### Repository and Git

- Structured machine learning repository
- `main`, `develop`, and `feature/*` branching strategy
- Pull Request based integration
- Conventional Commit messages
- GitHub Actions CI workflow
- Git and Docker ignore rules

### Model training

- CIFAR-10 dataset loading
- Training data augmentation
- ResNet-18 adapted for 32 by 32 images
- YAML-driven hyperparameters
- CPU and CUDA device detection
- Structured JSON Lines logging
- Batch-level progress output
- Validation after every epoch
- Validation-loss-based early stopping
- Best-checkpoint persistence

### Model serving

- FastAPI inference service
- `GET /health` endpoint
- `POST /predict` endpoint
- PNG and JPEG upload validation
- Predicted class and confidence
- Probabilities for all 10 CIFAR-10 classes
- HTTP 503 response when the model is unavailable
- HTTP 400 response for invalid uploads

### Docker

- Separate training and serving images
- Multi-stage Docker builds
- Slim Python runtime images
- Pinned dependencies
- Mounted data and checkpoint directories
- Read-only checkpoint mount for serving
- Non-root serving user
- Port 8080 exposure
- Docker health check

## Project Structure

```text
mlops-pytorch-pipeline/
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- configs/
|   |-- training_config.yaml
|   `-- training_config.docker.yaml
|-- docker/
|   |-- Dockerfile.train
|   `-- Dockerfile.serve
|-- k8s/
|   |-- namespace.yaml
|   |-- training-job.yaml
|   |-- serving-deployment.yaml
|   |-- serving-service.yaml
|   |-- configmap.yaml
|   `-- hpa.yaml
|-- requirements/
|   |-- train.txt
|   `-- serve.txt
|-- src/
|   |-- dataset.py
|   |-- model.py
|   |-- train.py
|   `-- serve.py
|-- tests/
|   `-- test_model.py
|-- .dockerignore
|-- .gitignore
`-- README.md
```

> Kubernetes files are reserved for the later deployment parts of the assignment. Parts A, B, and C are covered by the current implementation.

## Technology Stack

- Python 3.14
- PyTorch and torchvision
- CIFAR-10
- PyYAML
- FastAPI
- Uvicorn
- Pillow
- Pytest
- Docker Desktop
- Git and GitHub
- GitHub Actions

## Prerequisites

Install the following before running the project:

- Git
- Python 3.14, 64-bit
- Visual Studio Code with the Microsoft Python extension
- Docker Desktop using Linux containers

Confirm the tools in PowerShell:

```powershell
python --version
git --version
docker version
```

## Local Setup

### 1. Clone the repository

```powershell
git clone https://github.com/amitkislay/mlops-pytorch-pipeline.git
Set-Location .\mlops-pytorch-pipeline
```

### 2. Create the virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow it for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. Upgrade installation tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 5. Install dependencies

```powershell
python -m pip install -r .\requirements\train.txt
python -m pip install -r .\requirements\serve.txt
python -m pip install pytest
```

### 6. Verify imports

```powershell
python -c "import torch, torchvision, yaml, fastapi, uvicorn, PIL; print('Dependencies ready')"
```

## Training the Model

### 1. Verify or download CIFAR-10

```powershell
python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='data', train=True, download=True); print('CIFAR-10 ready')"
```

A completed download displays:

```text
Files already downloaded and verified
CIFAR-10 ready
```

### 2. Review the training configuration

Local training uses:

```text
configs/training_config.yaml
```

For a quick functional run, set:

```yaml
training:
  epochs: 1
```

For the complete configured run, increase the value as required.

### 3. Start training

```powershell
python .\src\train.py --config .\configs\training_config.yaml
```

Example structured output:

```json
{"event": "training_started", "architecture": "resnet18", "device": "cpu", "epochs": 1, "batch_size": 64}
{"event": "epoch_complete", "epoch": 1, "train_loss": 1.3483, "train_accuracy": 0.5105, "val_loss": 1.4255, "val_accuracy": 0.558}
{"event": "checkpoint_saved", "epoch": 1, "path": "checkpoints\\classifier_v1.pt", "val_loss": 1.4255}
{"event": "training_complete", "best_val_loss": 1.4255, "checkpoint_path": "checkpoints\\classifier_v1.pt"}
```

### 4. Verify the checkpoint

```powershell
Get-Item .\checkpoints\classifier_v1.pt |
    Select-Object Name, Length, LastWriteTime
```

Inspect its keys:

```powershell
python -c "import torch; c=torch.load('checkpoints/classifier_v1.pt', map_location='cpu', weights_only=False); print(c.keys())"
```

## Running the API

The API expects the checkpoint at:

```text
checkpoints/classifier_v1.pt
```

Start the server:

```powershell
python -m uvicorn serve:app --app-dir src --host 0.0.0.0 --port 8080
```

Open the interactive API documentation at:

```text
http://localhost:8080/docs
```

Stop the server with `Ctrl+C`.

## Testing the Endpoints

Open a second PowerShell terminal while the server is running.

### Health endpoint

```powershell
curl.exe -i "http://localhost:8080/health"
```

Expected status:

```text
HTTP/1.1 200 OK
```

Example body:

```json
{"status":"healthy","model_loaded":true,"device":"cpu"}
```

### Create a test image

```powershell
python -c "from torchvision.datasets import CIFAR10; d=CIFAR10(root='data', train=False, download=True); d[0][0].save('test_image.png'); print('Test image created')"
```

### Prediction endpoint

```powershell
curl.exe -X POST "http://localhost:8080/predict" -F "image=@.\test_image.png"
```

The response includes:

- `predicted_class`
- `predicted_class_index`
- `confidence`
- `probabilities` for all CIFAR-10 classes

## Running Tests

Compile the Python files:

```powershell
python -m compileall .\src .\tests
```

Run the test suite:

```powershell
python -m pytest .\tests -v
```

The tests cover:

- Correct model output dimensions
- Configurable class count
- Unsupported architecture handling

## Docker Training

### 1. Start Docker Desktop

Confirm the engine is running:

```powershell
docker version
```

Both `Client` and `Server` sections must be displayed.

### 2. Build the training image

```powershell
docker build -f .\docker\Dockerfile.train -t mlops-train:v1 .
```

### 3. Confirm image creation

```powershell
docker image ls mlops-train
```

### 4. Confirm the Python version in the image

```powershell
docker run --rm --entrypoint python mlops-train:v1 --version
```

### 5. Prepare mounted directories

```powershell
New-Item -ItemType Directory -Force .\data
New-Item -ItemType Directory -Force .\checkpoints
```

### 6. Run containerized training

```powershell
docker run --rm `
  --name mlops-training `
  --mount "type=bind,source=$($PWD.Path)\data,target=/app/data" `
  --mount "type=bind,source=$($PWD.Path)\checkpoints,target=/app/checkpoints" `
  mlops-train:v1
```

The data mount allows the container to reuse downloaded CIFAR-10 files. The checkpoint mount persists `classifier_v1.pt` on the host after the container exits.

Follow logs from another terminal if the container was started in detached mode:

```powershell
docker logs -f mlops-training
```

## Docker Model Serving

### 1. Build the serving image

```powershell
docker build -f .\docker\Dockerfile.serve -t mlops-serve:v1 .
```

### 2. Verify the checkpoint

```powershell
Test-Path .\checkpoints\classifier_v1.pt
```

The expected result is `True`.

### 3. Start the serving container

```powershell
docker run --rm `
  --name mlops-serving `
  -p 8080:8080 `
  --mount "type=bind,source=$($PWD.Path)\checkpoints,target=/app/checkpoints,readonly" `
  mlops-serve:v1
```

The checkpoint is mounted read-only because the API only needs to load it.

### 4. Validate serving from another terminal

```powershell
curl.exe -i "http://localhost:8080/health"
```

```powershell
curl.exe -X POST "http://localhost:8080/predict" -F "image=@.\test_image.png"
```

### 5. Check Docker health

Wait for the configured health-check start period, then run:

```powershell
docker inspect --format="{{.State.Health.Status}}" mlops-serving
```

Expected:

```text
healthy
```

### 6. Verify non-root execution

```powershell
docker exec mlops-serving whoami
```

Expected:

```text
appuser
```

### 7. Stop the container

```powershell
docker stop mlops-serving
```

## Configuration

### Local configuration

`configs/training_config.yaml` uses host-relative paths:

```yaml
data:
  data_dir: data

output:
  checkpoint_dir: checkpoints
  model_name: classifier_v1.pt
```

### Docker configuration

`configs/training_config.docker.yaml` uses container paths:

```yaml
data:
  data_dir: /app/data

output:
  checkpoint_dir: /app/checkpoints
  model_name: classifier_v1.pt
```

### Main parameters

```yaml
model:
  architecture: resnet18
  num_classes: 10

training:
  epochs: 1
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 3
  num_workers: 2
  seed: 42
```

## Structured Logging and Early Stopping

Training events are printed as one JSON object per line. This format is compatible with container logs and centralized logging platforms.

The checkpoint is updated only when validation loss improves. If validation loss does not improve for the configured patience period, training stops early to avoid unnecessary computation and overfitting.

## Git Methodology

### Branches

- `main`: stable, submission-ready code
- `develop`: integration branch
- `feature/*`: isolated implementation branches

Example branches:

```text
feature/project-setup
feature/pytorch-model
feature/docker-containerization
```

### Recommended workflow

```powershell
git switch develop
git pull origin develop
git switch -c feature/example-change
```

After implementation and testing:

```powershell
git add .
git commit -m "feat: describe the implemented feature"
git push -u origin feature/example-change
```

Create a Pull Request from the feature branch into `develop`. After integrated features are validated, create a final Pull Request from `develop` into `main`.

### Conventional Commit examples

```text
chore: initialize repository structure
feat: implement CIFAR-10 data pipeline
feat: add configuration-driven training
feat: add FastAPI prediction service
test: validate model output shape
build: containerize training and serving workloads
fix: expose CIFAR-10 normalization constants
docs: finalize project documentation
```

## Validation Results

The initial one-epoch functional training run produced:

```text
Training loss:       1.3483
Training accuracy:   0.5105
Validation loss:     1.4255
Validation accuracy: 0.5580
```

The run successfully generated:

```text
checkpoints/classifier_v1.pt
```

Successful workflow checks include:

- Source compilation
- Unit tests
- CIFAR-10 loading
- Local training
- JSON logging
- Checkpoint creation and loading
- Local health endpoint
- Local prediction endpoint
- Training image build and execution
- Serving image build and execution
- Docker health check
- Non-root serving process
- Containerized prediction

## Troubleshooting

### PowerShell cannot activate `.venv`

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Pillow tries to compile from source

Use a Python 3.14-compatible Pillow release in `requirements/serve.txt`, then install a binary wheel:

```powershell
python -m pip install --only-binary=:all: Pillow
```

### PyYAML tries to compile from source

Use a Python 3.14-compatible PyYAML release in `requirements/train.txt`:

```powershell
python -m pip install --only-binary=:all: PyYAML
```

### `CIFAR10_MEAN` import error

Ensure `src/dataset.py` defines:

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
```

### Prediction curl error 26

Check that the image exists:

```powershell
Test-Path .\test_image.png
```

Then send it with:

```powershell
curl.exe -X POST "http://localhost:8080/predict" -F "image=@.\test_image.png"
```

### Health endpoint returns HTTP 503

Verify the checkpoint:

```powershell
Test-Path .\checkpoints\classifier_v1.pt
```

Check the serving logs:

```powershell
docker logs mlops-serving
```

### Port 8080 is already in use

```powershell
docker ps
Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue
```

Stop the old container or map another host port:

```powershell
docker run --rm -p 8081:8080 mlops-serve:v1
```

### View container training progress

```powershell
docker ps
docker logs -f mlops-training
```

## Security and Repository Hygiene

The repository intentionally excludes:

- `.venv/`
- `data/`
- `checkpoints/`
- Environment-variable files
- Private keys and secrets
- Python cache files
- IDE-specific files
- Temporary logs

The serving container runs as a non-root user and receives the model checkpoint through a read-only mount.

Never commit passwords, access tokens, credentials, downloaded datasets, or generated model checkpoints.

## Future Work

The next project stages can add:

- Kubernetes namespace and ConfigMap
- PersistentVolumeClaim-backed training
- Kubernetes training Job
- Model-serving Deployment
- ClusterIP or LoadBalancer Service
- Liveness and readiness probes
- Horizontal Pod Autoscaler
- End-to-end Kubernetes validation

## AI Assistance Disclosure

AI assistance was used for implementation guidance, troubleshooting, Dockerfile structure, validation procedures, and documentation. All changes were manually reviewed, executed, tested, and understood before submission.

## License

This repository was created for an individual academic assignment. Add a project license only if required by the course or repository owner.
