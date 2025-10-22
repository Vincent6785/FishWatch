import os
import re
import cv2
from config import VIDEO_DIR

def discover_videos(extension: str = ".avi") -> list[str]:
    """
    Recherche tous les fichiers vidéo dans VIDEO_DIR correspondant à une extension donnée.

    Args:
        extension: Extension des fichiers à rechercher (par défaut ".avi").

    Returns:
        Liste des noms de fichiers vidéo trouvés.
    """
    videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(extension)]
    if not videos:
        print(f"[WARN] No {extension} files found in {VIDEO_DIR}.")
    else:
        print(f"[INFO] Found {len(videos)} video(s) to process.")
    return videos


def extract_datetime_key(filename: str) -> str:
    """
    Extrait une clé triable basée sur un motif de date/heure présent dans le nom du fichier.

    Args:
        filename: Nom du fichier.

    Returns:
        Chaîne utilisée pour le tri (format YYYYMMDDHHMMSS si détectée).
    """
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        date_part, time_part = match.groups()
        return f"{date_part}{time_part}"
    return filename


def get_video_infos(video_files: list[str]) -> list[tuple[str, int, float, float]]:
    """
    Récupère les informations essentielles de chaque vidéo (nombre d’images, FPS, durée).

    Args:
        video_files: Liste des fichiers vidéo à analyser.

    Returns:
        Liste de tuples : (nom_fichier, nombre_images, fps, durée_secondes)
    """
    infos = []
    total_frames = 0

    for video_name in sorted(video_files, key=extract_datetime_key):
        path = os.path.join(VIDEO_DIR, video_name)
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print(f"[WARN] Cannot open {video_name}, skipping.")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0

        infos.append((video_name, frame_count, fps, duration))
        total_frames += frame_count
        cap.release()

    print(f"[INFO] Total frames across all videos: {total_frames}")
    return infos
