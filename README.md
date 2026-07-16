# Sentiment Analysis MLOps

A production-grade sentiment analysis system built with end-to-end MLOps practices — from data ingestion through automated deployment on Kubernetes. Built by following a tutorial on MLOps-integrated development.

---

## Table of Contents

- [Overview](#overview)
- [MLOps Architecture](#mlops-architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [ML Pipeline](#ml-pipeline)
- [Flask Application](#flask-application)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment](#deployment)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Running Tests](#running-tests)

---

## Overview

This project classifies text reviews as **positive** or **negative** sentiment. The focus is less on the model itself and more on wrapping it in a complete MLOps lifecycle:

- **Reproducible pipelines** managed by DVC
- **Experiment tracking** via MLflow on DagsHub
- **Automated model registration** and promotion to Production
- **Containerized serving** via Docker + Flask (Gunicorn in production)
- **CI/CD** with GitHub Actions (4-stage pipeline)
- **Cloud deployment** on AWS EKS, with the image stored in AWS ECR
- **Observability** via Prometheus metrics exposed from the Flask app

---

## MLOps Architecture

```
Raw Data (GitHub URL)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                    DVC Pipeline                         │
│                                                         │
│  data_ingestion → data_preprocessing → feature_eng      │
│        │                                    │           │
│        ▼                                    ▼           │
│   data/raw/                          data/processed/    │
│   (train.csv, test.csv)          (train_bow.csv,        │
│                                   test_bow.csv,         │
│                                   vectorizer.pkl)       │
│                                            │            │
│                                            ▼            │
│                                    model_building       │
│                                            │            │
│                                            ▼            │
│                                    model_evaluation     │
│                                    (MLflow tracking)    │
│                                            │            │
│                                            ▼            │
│                                    model_registration   │
│                                    (MLflow registry →   │
│                                     Staging stage)      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│               GitHub Actions CI/CD                      │
│                                                         │
│  [1] ML Pipeline  →  [2] App Tests  →  [3] Docker ECR   │
│                                              │          │
│                                              ▼          │
│                                     [4] Deploy to EKS   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
  Flask App (served via Gunicorn, 2 replicas on EKS)
  + Prometheus /metrics endpoint
```

---

## Project Structure

```
sentiment-analysis-mlops/
│
├── src/                          # Modular ML source code
│   ├── data/
│   │   ├── data_ingestion.py     # Fetch, filter, split raw data
│   │   └── data_preprocessing.py # Text cleaning (stop words, lemmatization, etc.)
│   ├── features/
│   │   └── feature_engineering.py # Bag-of-Words vectorization (CountVectorizer)
│   ├── model/
│   │   ├── model_building.py     # Train Logistic Regression model
│   │   ├── model_evaluation.py   # Evaluate + log to MLflow
│   │   └── register_model.py     # Register model to MLflow registry (→ Staging)
│   ├── connection/               # (S3 connection utilities)
│   └── logger/                   # Centralized logging module
│
├── app/                          # Flask web application
│   ├── app.py                    # Flask app with predict + metrics routes
│   ├── preprocessing_utility.py  # Text normalization helpers
│   ├── templates/                # HTML templates (index.html)
│   └── requirements.txt          # App-only dependencies
│
├── tests/
│   ├── test_model.py             # Model load, signature, and performance tests
│   └── test_flask_app.py         # Flask route tests
│
├── scripts/
│   └── promote_model.py          # Promote Staging model → Production in MLflow
│
├── k8s/
│   ├── deployment.yaml           # EKS Deployment (2 replicas, resource limits)
│   └── service.yaml              # Kubernetes Service manifest
│
├── .github/workflows/
│   └── ci.yaml                   # 4-job GitHub Actions pipeline
│
├── data/                         # DVC-tracked data directories
│   ├── raw/                      # Output of data_ingestion
│   ├── interim/                  # Output of data_preprocessing
│   └── processed/                # Output of feature_engineering (BoW CSVs)
│
├── models/                       # DVC-tracked model artifacts
│   ├── model.pkl                 # Trained Logistic Regression model
│   └── vectorizer.pkl            # Fitted CountVectorizer
│
├── reports/
│   ├── metrics.json              # Evaluation metrics (accuracy, precision, recall, AUC)
│   └── experiment_info.json      # MLflow run ID + model path (used for registration)
│
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # DVC pipeline lock file
├── params.yaml                   # Pipeline hyperparameters
├── Dockerfile                    # Multi-stage Docker image for Flask app
├── Makefile                      # Developer convenience commands
├── pyproject.toml                # Project metadata and dependencies
├── requirements.txt              # Full project dependencies
└── .env                          # Local environment variables (not committed)
```

---

## Tech Stack

| Category | Tool |
|---|---|
| **Language** | Python 3.11+ |
| **ML** | scikit-learn (Logistic Regression, CountVectorizer) |
| **NLP** | NLTK (stopwords, WordNetLemmatizer) |
| **Pipeline Orchestration** | DVC |
| **Experiment Tracking** | MLflow + DagsHub |
| **Web Framework** | Flask + Gunicorn |
| **Containerization** | Docker |
| **Container Registry** | AWS ECR |
| **Orchestration** | AWS EKS (Kubernetes) |
| **CI/CD** | GitHub Actions |
| **Monitoring** | Prometheus Client |
| **Env Management** | python-dotenv |
| **Package Manager** | uv / pip |

---

## ML Pipeline

The pipeline is defined in `dvc.yaml` and orchestrated by DVC. Run the full pipeline with:

```bash
dvc repro
```

### Stages

| Stage | Script | Input | Output |
|---|---|---|---|
| `data_ingestion` | `src/data/data_ingestion.py` | Remote CSV URL | `data/raw/train.csv`, `data/raw/test.csv` |
| `data_preprocessing` | `src/data/data_preprocessing.py` | `data/raw/` | `data/interim/` |
| `feature_engineering` | `src/features/feature_engineering.py` | `data/interim/` | `data/processed/`, `models/vectorizer.pkl` |
| `model_building` | `src/model/model_building.py` | `data/processed/` | `models/model.pkl` |
| `model_evaluation` | `src/model/model_evaluation.py` | `models/model.pkl` | `reports/metrics.json`, MLflow run logged |
| `model_registration` | `src/model/register_model.py` | `reports/experiment_info.json` | Model registered to MLflow → `Staging` |

### Hyperparameters (`params.yaml`)

```yaml
data_ingestion:
  test_size: 0.25        # 75/25 train-test split

feature_engineering:
  max_features: 50       # Vocabulary size for CountVectorizer
```

Modify these and re-run `dvc repro` to track experiments.

### Model

- **Algorithm**: Logistic Regression (`C=1`, `solver=liblinear`, `penalty=l1`)
- **Features**: Bag-of-Words (CountVectorizer, top 50 features)
- **Metrics tracked**: Accuracy, Precision, Recall, AUC-ROC

---

## Flask Application

The `app/` directory contains a Flask web app that:

1. Loads the **Production** model version from the MLflow registry at startup (falls back to Staging → None)
2. Loads the `vectorizer.pkl` from the local `models/` directory
3. Exposes three routes:

| Route | Method | Description |
|---|---|---|
| `/` | GET | Renders the prediction UI |
| `/predict` | POST | Accepts text input, returns sentiment label |
| `/metrics` | GET | Exposes Prometheus metrics |

### Prometheus Metrics

The app instruments three custom metrics:

- `app_request_count` — total requests per method/endpoint
- `app_request_latency_seconds` — request latency histogram per endpoint
- `model_prediction_count` — prediction count per class label

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yaml`) has **4 sequential jobs**:

```
machine-learning-pipeline
        │
        ▼
   app-testing
        │
        ▼
build-and-push-docker
        │
        ▼
  deploy-on-eks
```

### Job 1 — `machine-learning-pipeline`

- Checks out code, installs dependencies
- Runs `dvc repro` to execute the full ML pipeline
- Uploads `models/` as a GitHub Actions artifact
- Runs model tests: `tests/test_model.py` (load, signature, performance)
- Promotes the model to Production via `scripts/promote_model.py`

### Job 2 — `app-testing`

- Downloads the trained model artifact from Job 1
- Installs NLTK data (stopwords, wordnet)
- Runs Flask app tests: `tests/test_flask_app.py`

### Job 3 — `build-and-push-docker`

- Authenticates to AWS ECR using repository secrets
- Builds the Docker image from the `Dockerfile`
- Tags and pushes the image to ECR

### Job 4 — `deploy-on-eks`

- Configures AWS credentials and `kubectl`
- Updates kubeconfig for the `flask-app-cluster` EKS cluster
- Creates/updates a Kubernetes secret (`sentanal-secret`) from GitHub Secrets
- Applies `k8s/` manifests to roll out the new deployment

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `DAGSHUB_TOKEN` | DagsHub personal access token |
| `DAGSHUB_REPO_OWNER` | DagsHub username |
| `DAGSHUB_REPO_NAME` | DagsHub repository name |
| `AWS_ACCESS_KEY_ID` | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key |
| `AWS_REGION` | AWS region (e.g. `us-east-1`) |
| `AWS_ACCOUNT_ID` | AWS account ID |
| `ECR_REPOSITORY` | ECR repository name |

---

## Deployment

### Docker (local)

```bash
# Build the image
docker build -t sentiment-analysis-mlops .

# Run locally
docker run -p 5000:5000 \
  -e DAGSHUB_TOKEN=<your_token> \
  -e DAGSHUB_REPO_OWNER=<your_username> \
  -e DAGSHUB_REPO_NAME=<your_repo_name> \
  sentiment-analysis-mlops
```

The container runs Gunicorn with a 120-second timeout:
```
gunicorn --bind 0.0.0.0:5000 --timeout 120 app:app
```

### Kubernetes (EKS)

The `k8s/` manifests deploy the app with:
- **2 replicas** for basic high availability
- Resource requests: `256Mi` memory, `250m` CPU
- Resource limits: `512Mi` memory, `1` CPU
- DagsHub credentials injected via a Kubernetes secret (`sentanal-secret`)

```bash
# Apply manually (if not using CI/CD)
kubectl apply -f k8s/
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) or pip
- DVC (`pip install dvc`)
- A [DagsHub](https://dagshub.com) account with an MLflow-enabled repository

### 1. Clone the repository

```bash
git clone https://github.com/vinayak-ktp/sentiment-analysis-mlops.git
cd sentiment-analysis-mlops
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with uv:

```bash
uv sync
```

### 3. Configure environment variables

Copy `.env` and fill in your credentials:

```bash
cp .env .env.local
```

```env
DAGSHUB_TOKEN=<your_dagshub_token>
DAGSHUB_REPO_OWNER=<your_dagshub_username>
DAGSHUB_REPO_NAME=<your_dagshub_repo_name>
```

### 4. Run the DVC pipeline

```bash
dvc repro
```

This executes all pipeline stages in order and caches intermediate outputs. Re-running only executes stages whose dependencies have changed.

### 5. Run the Flask app locally

```bash
cd app
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `DAGSHUB_TOKEN` | DagsHub personal access token (used as MLflow password) | ✅ |
| `DAGSHUB_REPO_OWNER` | DagsHub account username | ✅ |
| `DAGSHUB_REPO_NAME` | DagsHub repository name | ✅ |

---

## Running Tests

### Model tests (load + signature + performance thresholds)

```bash
python -m unittest tests/test_model.py
```

Tests verify:
- Model loads correctly from the MLflow registry (Staging stage)
- Input/output shapes match the vectorizer signature
- Accuracy, Precision, Recall, and F1 all exceed **0.40**

### Flask app tests

```bash
python -m unittest tests/test_flask_app.py
```

### All tests

```bash
python -m unittest discover -s tests
```

---

## Key MLOps Concepts Practiced

| Concept | Implementation |
|---|---|
| **Reproducible Pipelines** | DVC stages with dependency tracking and `dvc.lock` |
| **Experiment Tracking** | MLflow logging metrics, params, and model artifacts to DagsHub |
| **Model Registry** | Automated Staging → Production promotion via MLflow client |
| **Data Versioning** | DVC-tracked `data/` and `models/` directories |
| **Parameterization** | Centralized `params.yaml` consumed by pipeline stages |
| **Modular Code** | `src/` package with separate modules per concern |
| **Automated Testing** | Model quality gates + Flask smoke tests in CI |
| **Containerization** | Production-ready Dockerfile with Gunicorn |
| **CI/CD** | 4-stage GitHub Actions pipeline (train → test → build → deploy) |
| **Secrets Management** | GitHub Secrets → Kubernetes Secrets, never hardcoded |
| **Observability** | Prometheus metrics exposed at `/metrics` |

---

## License

[MIT](LICENSE)
