import os
import torch
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import torch.nn as nn

MODEL_PATH = "model_store/model.pth"

app = FastAPI(title="Wine Quality NN Prediction API")

model = None


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

class WineQualityNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

@app.on_event("startup")
def load_model():
    global model
    print("Loading model from volume:", MODEL_PATH)
    input_dim = 11  # Number of features
    model = WineQualityNet(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: WineFeatures):
    df = pd.DataFrame([features.dict()])
    tensor = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        pred = model(tensor).item()
    return {"quality_prediction": pred}
