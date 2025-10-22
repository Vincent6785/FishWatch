import torch
from ultralytics import YOLO
from config import MODEL_PATH

def load_model():
    print(f"[INFO] Loading YOLO model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    device = 'cuda' if torch.cuda.is_available() else exit(1)
    model.to(device)
    return model
