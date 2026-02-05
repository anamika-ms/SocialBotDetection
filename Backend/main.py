import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ==============================
# FASTAPI INIT
# ==============================

app = FastAPI()

# Allow React frontend (adjust port if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# DEVICE
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# MODEL DEFINITIONS
# ==============================

class StructuredEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

class BotClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# ==============================
# LOAD MODELS ON STARTUP
# ==============================

structured_scaler = joblib.load("structured_scaler.pkl")
graph_scaler = joblib.load("graph_scaler.pkl")

structured_encoder = StructuredEncoder().to(device)
graph_encoder = GraphEncoder().to(device)
classifier = BotClassifier().to(device)

structured_encoder.load_state_dict(torch.load("structured_encoder.pt", map_location=device))
graph_encoder.load_state_dict(torch.load("graph_encoder.pt", map_location=device))
classifier.load_state_dict(torch.load("classifier.pt", map_location=device))

structured_encoder.eval()
graph_encoder.eval()
classifier.eval()

threshold_dict = torch.load("threshold.pt", map_location=device, weights_only=False)
threshold = threshold_dict["best_threshold"]

# ==============================
# REQUEST SCHEMA
# ==============================

class UserFeatures(BaseModel):
    structured_features: list
    graph_features: list

# ==============================
# PREDICTION ENDPOINT
# ==============================

@app.post("/predict")
def predict(data: UserFeatures):
    structured_input = np.array(data.structured_features).reshape(1, -1)
    graph_input = np.array(data.graph_features).reshape(1, -1)

    structured_scaled = structured_scaler.transform(structured_input)
    graph_scaled = graph_scaler.transform(graph_input)

    structured_tensor = torch.tensor(structured_scaled, dtype=torch.float32).to(device)
    graph_tensor = torch.tensor(graph_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        z_struct = structured_encoder(structured_tensor)
        z_graph = graph_encoder(graph_tensor)

        fused = torch.cat([z_struct, z_graph], dim=1)

        logits = classifier(fused)
        probs = torch.softmax(logits, dim=1)

        bot_prob = probs[:, 1].item()

        prediction = "BOT" if bot_prob > threshold else "HUMAN"

    return {
        "prediction": prediction,
        "bot_probability": bot_prob,
        "threshold": threshold
    }