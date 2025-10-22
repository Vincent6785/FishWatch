import pandas as pd
import numpy as np
from config import OUTPUT_DIR, CLASSES

def generate_activity_periods(merged_csv, output_csv):
    merged_df = pd.read_csv(merged_csv)

    def group_timestamps(ts, max_gap=10.0):
        if not ts: return []
        ts = np.array(sorted(ts))
        periods = []
        start = ts[0]
        prev = start
        for t in ts[1:]:
            if t-prev > max_gap:
                periods.append((start, prev))
                start = t
            prev = t
        periods.append((start, prev))
        return periods

    activity_periods = {}
    for cls in CLASSES:
        times = merged_df.loc[merged_df["Class"]==cls,"Timestamp"].tolist()
        activity_periods[cls] = group_timestamps(times)

    rows = []
    for cls, periods in activity_periods.items():
        for s,e in periods:
            rows.append({"Class":cls,"Start (s)":round(s,2),"End (s)":round(e,2),"Duration (s)":round(e-s,2)})

    final_df = pd.DataFrame(rows)
    final_df.to_csv(output_csv, index=False)
    print(f"[INFO] Activity report saved: {output_csv}")
    return output_csv
