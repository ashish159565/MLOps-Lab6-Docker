# MLOps Lab 6 - Docker Containerization

A containerized machine learning pipeline for training and serving a Wine Quality prediction model using PyTorch and Docker.

## Architecture

This project demonstrates a simple ML pipeline with Docker containers:

- **Training Service**: PyTorch neural network training container
- **Serving API**: FastAPI-based REST API for predictions
- **Shared Volume**: Model persistence between training and serving

## Prerequisites

- Docker
- Docker Compose
- NVIDIA GPU + Docker GPU support (optional, for GPU training)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ashish159565/MLOps-Lab6-Docker.git
cd MLOps-Lab6-Docker
```

### 2. Build the Docker Images

```bash
docker-compose build
```

### 3. Train the Model

```bash
docker-compose run training
```

This will:

- Download the Wine Quality dataset
- Train a PyTorch neural network
- Save the model to a shared volume at `model_store/model.pth`

### 4. Start the Serving API

```bash
docker-compose up -d serving
```

### 5. Access the API

- **Prediction API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔮 Making Predictions

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

### Using Python

```python
import requests

data = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Project Structure

```
.
├── docker-compose.yml          # Container orchestration
├── training/
│   ├── Dockerfile             # Training container image
│   ├── requirements.txt       # Python dependencies (torch, pandas, scikit-learn)
│   └── train.py              # Training script
└── serving/
    ├── Dockerfile             # Serving container image
    ├── requirements.txt       # Python dependencies (fastapi, uvicorn, torch)
    └── app.py                # FastAPI prediction service
```

## 🐳 Docker Services

### Training Container

- PyTorch-based neural network
- Downloads Wine Quality dataset from UCI repository
- Trains a 4-layer neural network (128→64→32→1)
- Saves model to shared volume (`model_store/model.pth`)
- GPU-enabled (requires NVIDIA Docker runtime)

### Serving Container

- FastAPI REST API
- Loads model from shared volume
- Provides `/predict` endpoint for predictions
- Provides `/health` endpoint for health checks
- Interactive API docs at `/docs`

## Common Commands

### View Logs

```bash
# Training logs
docker-compose logs training

# Serving logs
docker-compose logs -f serving
```

### Stop Services

```bash
docker-compose down
```

### Remove Volumes

```bash
docker-compose down -v
```

### Rebuild Images

```bash
docker-compose build
docker-compose up -d serving
```

### Run Training Again

```bash
docker-compose run training
```

## Model Details

### Neural Network Architecture

```
Input (11 features) → Linear(128) → ReLU
                    → Linear(64)  → ReLU
                    → Linear(32)  → ReLU
                    → Linear(1)   → Output
```

### Features

The model predicts wine quality based on 11 input features:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

### Training Configuration

- Optimizer: Adam (lr=0.001)
- Loss: MSE Loss
- Epochs: 50
- Batch size: 32
- Train/Test split: 80/20
- Dataset: UCI Wine Quality (Red Wine)
- Feature scaling: StandardScaler

## Troubleshooting

### Model Not Found in Serving

Ensure training has completed and model is saved:

```bash
docker-compose run training
docker exec serving ls -la model_store/
```

### GPU Not Available

If you don't have NVIDIA GPU, remove the GPU configuration from `docker-compose.yml`:

```yaml
# Remove this section from training service:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Port Conflicts

If port 8000 is in use, modify the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "8001:8000" # Change host port (left side)
```

### Container Build Issues

Clean rebuild:

```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose run training
docker-compose up -d serving
```

## Notes

- First run downloads Docker base images (may take a few minutes)
- Training downloads the Wine Quality dataset on each run
- Model persists in a Docker volume shared between containers
- Serving container must be started after training completes
- GPU acceleration is optional but recommended for faster training
