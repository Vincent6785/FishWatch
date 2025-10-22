import sys
import torch
from ultralytics import YOLO
from config import MODEL_PATH

def load_model():
    """
    Charge le modèle YOLO entraîné et le déplace sur le périphérique approprié (GPU si disponible).

    Returns:
        Le modèle YOLO prêt à l'emploi.
    """
    print(f"[INFO] Loading YOLO model from {MODEL_PATH}")

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load YOLO model: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] CUDA not available — using CPU (processing will be slower).")

    model.to(device)
    print(f"[INFO] Model loaded successfully on {device.upper()}.")

    return model
