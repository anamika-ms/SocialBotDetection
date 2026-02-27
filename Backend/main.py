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
            nn.Linear(50,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )

    def forward(self,x):
        return F.normalize(self.net(x),dim=1)


class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(5,32),
            nn.ReLU(),
            nn.Linear(32,64)
        )

    def forward(self,x):
        return F.normalize(self.net(x),dim=1)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(768,256),
            nn.ReLU(),
            nn.Linear(256,64)
        )

    def forward(self,x):
        return F.normalize(self.net(x),dim=1)


# ==============================
# FULL MODEL (ATTENTION)
# ==============================

class AttentionBotClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.attention = nn.Linear(64,1)

        self.classifier = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self,zs,zg,zt):

        views = torch.stack([zs,zg,zt],dim=1)

        scores = self.attention(views)
        weights = torch.softmax(scores,dim=1)

        fused = torch.sum(weights*views,dim=1)

        return self.classifier(fused)


# ==============================
# OPTIMIZED MODEL (FIXED)
# ==============================

class OptClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(128,64),   # ✅ FIXED
            nn.ReLU(),
            nn.Linear(64,2)      # ✅ FIXED
        )

    def forward(self, x):
        return self.net(x)


# ==============================
# LOAD DATA
# ==============================

print("Loading feature matrices...")

structured_features = np.load("structured_features.npy")
graph_features = np.load("graph_features.npy")
text_embeddings = np.load("text_embeddings.npy", mmap_mode="r")
user_ids = np.load("user_ids.npy")

print("Features loaded")


# ==============================
# LABELS
# ==============================

labels_df = pd.read_csv("label.csv")
label_dict = dict(zip(labels_df["id"], labels_df["label"]))


# ==============================
# LOAD SCALERS
# ==============================

structured_scaler = joblib.load("structured_scaler.pkl")
graph_scaler = joblib.load("graph_scaler.pkl")


# ==============================
# LOAD FULL MODEL
# ==============================

print("Loading full model...")

structured_encoder = StructuredEncoder().to(device)
graph_encoder = GraphEncoder().to(device)
text_encoder = TextEncoder().to(device)
classifier = AttentionBotClassifier().to(device)

structured_encoder.load_state_dict(torch.load("structured_encoder.pt", map_location=device))
graph_encoder.load_state_dict(torch.load("graph_encoder.pt", map_location=device))
text_encoder.load_state_dict(torch.load("text_encoder.pt", map_location=device))
classifier.load_state_dict(torch.load("classifier.pt", map_location=device))

structured_encoder.eval()
graph_encoder.eval()
text_encoder.eval()
classifier.eval()


# ==============================
# LOAD OPTIMIZED MODEL
# ==============================

print("Loading optimized model...")

opt_structured_encoder = StructuredEncoder().to(device)
opt_graph_encoder = GraphEncoder().to(device)
opt_classifier = OptClassifier().to(device)

opt_structured_encoder.load_state_dict(torch.load("opt_structured_encoder.pt", map_location=device))
opt_graph_encoder.load_state_dict(torch.load("opt_graph_encoder.pt", map_location=device))
opt_classifier.load_state_dict(torch.load("opt_classifier.pt", map_location=device))

opt_structured_encoder.eval()
opt_graph_encoder.eval()
opt_classifier.eval()


# ==============================
# THRESHOLDS
# ==============================

threshold = 0.65
opt_threshold = 0.65


# ==============================
# USER LOOKUP
# ==============================

user_index_map = {uid:i for i,uid in enumerate(user_ids)}


# ==============================
# REQUEST MODEL
# ==============================

class UserRequest(BaseModel):
    user_id: str
    model_type: str   # "optimized" or "full"


# ==============================
# ROUTES
# ==============================

@app.get("/users")
def get_users():
    return {"users": user_ids.tolist()[:200]}


@app.post("/predict")
def predict(data: UserRequest):

    user_id = data.user_id
    model_type = data.model_type

    if user_id not in user_index_map:
        raise HTTPException(status_code=404, detail="User not found")

    index = user_index_map[user_id]

    structured_input = structured_features[index].reshape(1,-1)
    graph_input = graph_features[index].reshape(1,-1)

    structured_scaled = structured_scaler.transform(structured_input)
    graph_scaled = graph_scaler.transform(graph_input)

    structured_tensor = torch.tensor(structured_scaled,dtype=torch.float32).to(device)
    graph_tensor = torch.tensor(graph_scaled,dtype=torch.float32).to(device)

    with torch.no_grad():

        # ======================
        # OPTIMIZED MODEL
        # ======================
        if model_type == "optimized":

            zs = opt_structured_encoder(structured_tensor)
            zg = opt_graph_encoder(graph_tensor)

            fused = torch.cat([zs, zg], dim=1)   # ✅ FIXED

            logits = opt_classifier(fused)
            probs = torch.softmax(logits, dim=1)

            bot_prob = probs[:,1].item()
            used_model = "Optimized Contrastive"

        # ======================
        # FULL MODEL
        # ======================
        elif model_type == "full":

            text_input = text_embeddings[index].reshape(1,-1)
            text_tensor = torch.tensor(text_input,dtype=torch.float32).to(device)

            zs = structured_encoder(structured_tensor)
            zg = graph_encoder(graph_tensor)
            zt = text_encoder(text_tensor)

            logits = classifier(zs, zg, zt)
            probs = torch.softmax(logits, dim=1)

            bot_prob = probs[:,1].item()
            used_model = "Full Contrastive (Attention)"

        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

    prediction = "bot" if bot_prob > threshold else "human"
    true_label = label_dict.get(user_id,"unknown")

    return {
        "user_id": user_id,
        "model_used": used_model,
        "prediction": prediction,
        "true_label": true_label,
        "correct": prediction == true_label,
        "bot_probability": round(bot_prob,4)
    }