import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
# LOAD SCALERS
# ==============================

structured_scaler = joblib.load("structured_scaler.pkl")
graph_scaler = joblib.load("graph_scaler.pkl")
print("Scalers loaded successfully")

# ==============================
# INITIALIZE MODELS
# ==============================

structured_encoder = StructuredEncoder().to(device)
graph_encoder = GraphEncoder().to(device)
classifier = BotClassifier().to(device)

# ==============================
# LOAD WEIGHTS
# ==============================

structured_encoder.load_state_dict(torch.load("structured_encoder.pt", map_location=device))
graph_encoder.load_state_dict(torch.load("graph_encoder.pt", map_location=device))
classifier.load_state_dict(torch.load("classifier.pt", map_location=device))

structured_encoder.eval()
graph_encoder.eval()
classifier.eval()

print("Models loaded successfully")

# ==============================
# LOAD THRESHOLD
# ==============================

threshold_dict = torch.load("threshold.pt", map_location=device, weights_only=False)
threshold = threshold_dict["best_threshold"]

print("Threshold loaded:", threshold)

print("All components loaded correctly ✅")



import numpy as np

# ==============================
# TEST WITH DUMMY DATA
# ==============================

# Dummy structured features (50 features)
structured_input = np.random.rand(1, 50)
structured_scaled = structured_scaler.transform(structured_input)
structured_tensor = torch.tensor(structured_scaled, dtype=torch.float32).to(device)

# Dummy graph features (5 features)
graph_input = np.random.rand(1, 5)
graph_scaled = graph_scaler.transform(graph_input)
graph_tensor = torch.tensor(graph_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    z_struct = structured_encoder(structured_tensor)
    z_graph = graph_encoder(graph_tensor)

    print("Structured embedding shape:", z_struct.shape)
    print("Graph embedding shape:", z_graph.shape)

    fused = torch.cat([z_struct, z_graph], dim=1)
    print("Fused embedding shape:", fused.shape)

    logits = classifier(fused)
    probs = torch.softmax(logits, dim=1)

    print("Logits:", logits)
    print("Probabilities:", probs)

    bot_prob = probs[:, 1].item()

    prediction = "BOT" if bot_prob > threshold else "HUMAN"

    print("Final Prediction:", prediction)
    print("Bot Probability:", bot_prob)