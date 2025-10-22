import os

MODEL_PATH = "./runs/detect/train208/weights/best.pt"
VIDEO_DIR = "./videos/validation"
OUTPUT_DIR = "./report_validation"
TEMP_DIR = os.path.join(OUTPUT_DIR, "tmp_detections")
OUTPUT_VIDEOS_DIR = os.path.join(OUTPUT_DIR, "annotated_videos")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

CLASSES = [
    'Ancistrus', 'Botia_Almorhae', 'Botia_Striata',
    'Corydoras_Albinos', 'Corydoras_Panda', 'G_Matriarche',
    'Pterophyllum_Scalare_B', 'Pterophyllum_Scalare_BB',
    'Pterophyllum_Scalare_W', 'Pterophyllum_Scalare_Y',
    'Pterophyllum_Scalare_YWB'
]

SCALARE_CLASSES = [
    'Pterophyllum_Scalare_B', 'Pterophyllum_Scalare_BB',
    'Pterophyllum_Scalare_W', 'Pterophyllum_Scalare_Y',
    'Pterophyllum_Scalare_YWB'
]

DIST_THRESH = [50, 100, 275]
MAX_GPU_TEMP = 40
CHECK_INTERVAL = 10
