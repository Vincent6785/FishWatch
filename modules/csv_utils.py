import os
import pandas as pd
from config import TEMP_DIR, OUTPUT_DIR

def merge_csvs():
    merged_df = pd.concat([pd.read_csv(os.path.join(TEMP_DIR,f)) for f in os.listdir(TEMP_DIR)
                           if os.path.getsize(os.path.join(TEMP_DIR,f))>0], ignore_index=True)
    merged_df.sort_values("Timestamp", inplace=True)
    merged_path = os.path.join(OUTPUT_DIR,"merged_detections.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"[INFO] Merged detections saved: {merged_path}")
    return merged_path
