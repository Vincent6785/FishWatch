import matplotlib.pyplot as plt
import pandas as pd
import os
from config import OUTPUT_DIR

def generate_gantt_charts(activity_csv):
    activity_df = pd.read_csv(activity_csv, usecols=["Class","Start (s)","End (s)"])
    periods = {cls:list(zip(sub["Start (s)"], sub["End (s)"])) for cls,sub in activity_df.groupby("Class")}
    total_time = activity_df["End (s)"].max()

    def plot_gantt(periods, start, end, title, filepath, max_segments=2000):
        plt.figure(figsize=(14,7))
        y_labels = list(periods.keys())
        for i, cls in enumerate(y_labels):
            segs = [(max(s,start), min(e,end)) for s,e in periods[cls] if not (e<start or s>end)]
            if len(segs) > max_segments:
                idx = np.linspace(0,len(segs)-1,max_segments,dtype=int)
                segs = [segs[j] for j in idx]
            for s,e in segs:
                plt.plot([s,e],[i,i], linewidth=4)
        plt.yticks(range(len(y_labels)), y_labels)
        plt.xlabel("Time (s)")
        plt.ylabel("Class")
        plt.title(title)
        plt.grid(True,axis='x',linestyle='--',alpha=0.4)
        plt.tight_layout()
        plt.savefig(filepath,dpi=200)
        plt.close()

    plot_gantt(periods, 0, total_time, "Gantt Chart - Total Duration", os.path.join(OUTPUT_DIR,"gantt_total.png"))
    half = total_time/2
    plot_gantt(periods, 0, half, "Gantt Chart - First Half", os.path.join(OUTPUT_DIR,"gantt_half_1.png"))
    plot_gantt(periods, half, total_time, "Gantt Chart - Second Half", os.path.join(OUTPUT_DIR,"gantt_half_2.png"))
    quarter = total_time/4
    for i in range(4):
        plot_gantt(periods, i*quarter, (i+1)*quarter, f"Gantt Chart - Quarter {i+1}", os.path.join(OUTPUT_DIR,f"gantt_quarter_{i+1}.png"))

    print("[INFO] Gantt charts generated.")
