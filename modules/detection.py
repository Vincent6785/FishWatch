import os
import time
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import VIDEO_DIR, TEMP_DIR, OUTPUT_VIDEOS_DIR, CLASSES, SCALARE_CLASSES, MAX_GPU_TEMP, CHECK_INTERVAL
import pynvml

def get_gpu_temp() -> float | None:
    """
    Récupère la température actuelle du GPU à l'aide de NVML.

    Returns:
        Température du GPU en degrés Celsius, ou None si lecture impossible.
    """
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return temp
    except Exception:
        return None


def process_videos(model, video_infos: List[Tuple[str, int, float, float]]) -> Tuple[List[str], List[str]]:
    """
    Traite les vidéos : détection d'objets, annotation, sauvegarde CSV temporaire.

    Args:
        model: Modèle YOLO chargé.
        video_infos: Liste de tuples (nom_fichier, nb_frames, fps, durée_s).

    Returns:
        Tuple :
            - Liste des chemins des CSV temporaires générés.
            - Liste des chemins des vidéos annotées générées.
    """
    cumulative_time = 0.0
    temp_csv_files: List[str] = []
    annotated_videos: List[str] = []

    total_frames = sum(info[1] for info in video_infos)
    with tqdm(total=total_frames, desc="Processing all videos", ncols=110) as global_pbar:
        for video_name, frame_count, fps, duration in video_infos:
            temp_final_csv = os.path.join(TEMP_DIR, f"{video_name}_final.csv")
            annotated_path = os.path.join(OUTPUT_VIDEOS_DIR, f"{video_name}_annotated.mp4")

            # Skip si déjà traité
            if os.path.exists(temp_final_csv) and os.path.exists(annotated_path):
                print(f"[SKIP] {video_name} already processed, skipping.")
                cumulative_time += duration
                global_pbar.update(frame_count)
                continue

            video_path = os.path.join(VIDEO_DIR, video_name)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[WARN] Failed to open {video_name}, skipping.")
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

            temp_data: List[Tuple] = []

            for frame_idx in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = cumulative_time + (frame_idx / fps)
                results = model(frame, conf=0.8, verbose=False)[0]

                for box in results.boxes:
                    cls_id = int(box.cls)
                    if cls_id >= len(CLASSES):
                        continue
                    label = CLASSES[cls_id]
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    color = (0, 255, 0) if label in SCALARE_CLASSES else (255, 0, 0)
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    cv2.putText(frame, label, (xyxy[0], xyxy[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    temp_data.append((label, timestamp, *xyxy))

                out_writer.write(frame)
                global_pbar.update(1)

            # Sauvegarde CSV temporaire
            if temp_data:
                temp_df = pd.DataFrame(temp_data, columns=["Class", "Timestamp", "x1", "y1", "x2", "y2"])
                temp_df.to_csv(temp_final_csv, index=False)
                temp_csv_files.append(temp_final_csv)

            cap.release()
            out_writer.release()
            annotated_videos.append(annotated_path)
            cumulative_time += duration

            # Refroidissement GPU si nécessaire
            while True:
                gpu_temp = get_gpu_temp()
                if not gpu_temp or gpu_temp <= MAX_GPU_TEMP:
                    break
                print(f"[WAIT] GPU {gpu_temp}°C > {MAX_GPU_TEMP}°C, waiting {CHECK_INTERVAL}s...")
                time.sleep(CHECK_INTERVAL)

    return temp_csv_files, annotated_videos
