# MLOps PyTorch Pipeline

An end-to-end MLOps project for training and serving a PyTorch image
classification model using Docker and Kubernetes.

## Project Objectives

- Train a PyTorch image classifier
- Containerize training and serving workloads
- Deploy training as a Kubernetes Job
- Deploy model serving as a Kubernetes Deployment
- Manage configuration using ConfigMaps and Secrets
- Implement CI using GitHub Actions

## Project Structure

- `src/`: Model, dataset, training, and serving code
- `configs/`: Training configuration
- `docker/`: Training and serving Dockerfiles
- `k8s/`: Kubernetes manifests
- `requirements/`: Python dependencies
- `tests/`: Automated tests
- `.github/workflows/`: CI workflow

## Branching Strategy

- `main`: Stable and submission-ready code
- `develop`: Integration branch
- `feature/*`: Individual development tasks

## Status

Project structure initialized. Implementation is in progress.
