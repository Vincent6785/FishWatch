import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import SCALARE_CLASSES, DIST_THRESH, OUTPUT_DIR

def analyze_scalar_pairs(merged_csv, output_dir=OUTPUT_DIR):
    df = pd.read_csv(merged_csv)
    scal_df = df[df["Class"].isin(SCALARE_CLASSES)].copy()
    if scal_df.empty:
        print("[INFO] No scalare detections found.")
        return

    scal_df["cx"] = (scal_df["x1"]+scal_df["x2"])/2
    scal_df["cy"] = (scal_df["y1"]+scal_df["y2"])/2
    avg_fps = 30  # optionnel, calculable si infos disponibles
    scal_df["time_bucket"] = (scal_df["Timestamp"]//1)*1
    pair_counts = {th:{} for th in DIST_THRESH}

    for t, group in tqdm(scal_df.groupby("time_bucket"), desc="Scalar pair analysis"):
        coords = group[["cx","cy"]].to_numpy(dtype=np.float32)
        classes = group["Class"].to_numpy()
        n = len(coords)
        if n<2: continue
        diff_x = coords[:,0].reshape(-1,1) - coords[:,0]
        diff_y = coords[:,1].reshape(-1,1) - coords[:,1]
        dist_matrix = np.sqrt(diff_x**2 + diff_y**2)
        for i in range(n):
            for j in range(i+1,n):
                c1,c2 = classes[i],classes[j]
                if c1==c2: continue
                d = dist_matrix[i,j]
                for th in DIST_THRESH:
                    if d<th:
                        key="_".join(sorted([c1,c2]))
                        pair_counts[th][key] = pair_counts[th].get(key,0)+1

    for th, counts in pair_counts.items():
        if not counts: continue
        df_out = pd.DataFrame(list(counts.items()), columns=["PairKey","FramesTogether"])
        df_out["TimeTogether_s"] = df_out["FramesTogether"]/avg_fps
        df_out[["Class1","Class2"]] = df_out["PairKey"].str.split("_", n=1, expand=True)
        df_out.to_csv(f"{output_dir}/scalar_pair_{th}px.csv", index=False)

        top10 = df_out.sort_values("FramesTogether",ascending=False).head(10)
        if not top10.empty:
            plt.figure(figsize=(10,6))
            top10["Pair"] = top10["Class1"] + " ↔ " + top10["Class2"]
            plt.barh(top10["Pair"][::-1], top10["FramesTogether"][::-1], color="skyblue")
            plt.xlabel("Frames Together")
            plt.title(f"Top 10 Scalar Pairs (<{th}px)")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/scalar_pair_top10_{th}px.png")
            plt.close()

    print("[INFO] Scalar pair analysis completed.")
