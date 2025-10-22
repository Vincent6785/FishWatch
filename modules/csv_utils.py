import os
import pandas as pd
from config import TEMP_DIR, OUTPUT_DIR

def merge_csvs():
    """Fusionne tous les fichiers CSV valides du dossier temporaire et crée un CSV global trié par timestamp."""

    csv_files = [
        os.path.join(TEMP_DIR, f)
        for f in os.listdir(TEMP_DIR)
        if f.lower().endswith(".csv") and os.path.getsize(os.path.join(TEMP_DIR, f)) > 0
    ]

    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV valide trouvé dans {TEMP_DIR}")

    # Chargement et fusion
    merged_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    merged_df.sort_values("Timestamp", inplace=True)

    # Sauvegarde
    merged_path = os.path.join(OUTPUT_DIR, "merged_detections.csv")
    merged_df.to_csv(merged_path, index=False)

    print(f"[INFO] Merged detections saved: {merged_path}")
    print(f"[INFO] Total rows merged: {len(merged_df):,}")

    return merged_path
