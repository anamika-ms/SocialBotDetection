import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ==============================
# FASTAPI INIT
# ==============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# LOAD DATA ON STARTUP
# ==============================

structured_features = np.load("structured_features.npy")
graph_features = np.load("graph_features.npy")
user_ids = np.load("user_ids.npy")

labels_df = pd.read_csv("label.csv")
label_dict = dict(zip(labels_df["id"], labels_df["label"]))

# ==============================
# LOAD MODELS
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
# REQUEST MODEL
# ==============================

class UserRequest(BaseModel):
    user_id: str

# ==============================
# GET USER LIST
# ==============================

@app.get("/users")
def get_users():
    return {"users": user_ids.tolist()[:200]}  # limit for dropdown

# ==============================
# PREDICTION ENDPOINT
# ==============================

@app.post("/predict")
def predict(data: UserRequest):
    user_id = data.user_id

    if user_id not in label_dict:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        index = np.where(user_ids == user_id)[0][0]
    except IndexError:
        raise HTTPException(status_code=404, detail="User ID not in feature matrix")

    structured_input = structured_features[index].reshape(1, -1)
    graph_input = graph_features[index].reshape(1, -1)

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

    prediction = "bot" if bot_prob > threshold else "human"
    true_label = label_dict[user_id]

    return {
        "user_id": user_id,
        "prediction": prediction,
        "true_label": true_label,
        "correct": prediction == true_label,
        "bot_probability": round(bot_prob, 4),
        "threshold": threshold
        
    }