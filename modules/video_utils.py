import os
import re
import cv2
from config import VIDEO_DIR

def discover_videos(extension=".avi"):
    return [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(extension)]

def extract_datetime_key(filename):
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        date_part, time_part = match.groups()
        return f"{date_part}{time_part}"
    return filename

def get_video_infos(video_files):
    infos = []
    total_frames = 0
    for video_name in sorted(video_files, key=extract_datetime_key):
        cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_name))
        if not cap.isOpened():
            print(f"[WARN] Cannot open {video_name}, skipping.")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        total_frames += frame_count
        infos.append((video_name, frame_count, fps, duration))
        cap.release()
    print(f"[INFO] Total frames: {total_frames}")
    return infos
