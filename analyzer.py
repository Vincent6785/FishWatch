import cv2
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
import re
import numpy as np
import time
import pynvml

# ==============================
# CONFIGURATION
# ==============================
MODEL_PATH = "./runs/detect/train208/weights/best.pt"
VIDEO_DIR = "./videos"
OUTPUT_DIR = "./reports"
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

# ==============================
# MODEL LOADING
# ==============================
print(f"[INFO] Loading YOLO model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)
device = 'cuda' if torch.cuda.is_available() else exit(1)
model.to(device)

# ==============================
# VIDEO DISCOVERY & SORTING
# ==============================
avi_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".avi")]
if not avi_files:
    raise RuntimeError(f"No .avi files found in {VIDEO_DIR}")

def extract_datetime_key(filename):
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        date_part, time_part = match.groups()
        return f"{date_part}{time_part}"
    return filename

avi_files.sort(key=extract_datetime_key)

print(f"[INFO] Found {len(avi_files)} video files to process:")
for f in avi_files:
    print(f"   - {f}")

# ==============================
# PRE-CALCULATE TOTAL FRAMES
# ==============================
video_infos = []
total_frames = 0
for video_name in avi_files:
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_name))
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_name}, skipping.")
        continue
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    total_frames += frame_count
    video_infos.append((video_name, frame_count, fps, duration))
    cap.release()

print(f"[INFO] Total frames to process across all videos: {total_frames}")

# ==============================
# GPU TEMPERATURE CONTROL
# ==============================
MAX_GPU_TEMP = 40
CHECK_INTERVAL = 10

def get_gpu_temp():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return temp
    except Exception:
        return None

# ==============================
# MAIN PROCESSING LOOP
# ==============================
cumulative_time = 0.0
temp_csv_files = []
annotated_videos = []

with tqdm(total=total_frames, desc="Processing all videos", ncols=110) as global_pbar:
    for video_name, frame_count, fps, duration in video_infos:
        temp_final_csv = os.path.join(TEMP_DIR, f"{video_name}_final.csv")
        annotated_path = os.path.join(OUTPUT_VIDEOS_DIR, f"{video_name}_annotated.mp4")
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

        print(f"[INFO] Processing {video_name} ({frame_count} frames, {fps:.1f} FPS, {duration:.1f}s)")
        temp_data = []

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cumulative_time + (frame_idx / fps)
            results = model(frame, conf=0.8, verbose=False)[0]

            for box in results.boxes:
                cls_id = int(box.cls)
                if cls_id < len(CLASSES):
                    label = CLASSES[cls_id]
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    color = (0, 255, 0) if label in SCALARE_CLASSES else (255, 0, 0)
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    cv2.putText(frame, label, (xyxy[0], xyxy[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    temp_data.append((label, timestamp, *xyxy))

            out_writer.write(frame)
            global_pbar.update(1)

        if temp_data:
            temp_df = pd.DataFrame(temp_data, columns=["Class","Timestamp","x1","y1","x2","y2"])
            temp_df.to_csv(temp_final_csv, index=False)
            temp_csv_files.append(temp_final_csv)

        cap.release()
        out_writer.release()
        annotated_videos.append(annotated_path)
        cumulative_time += duration

        # Attente GPU refroidissement
        while True:
            gpu_t = get_gpu_temp()
            if not gpu_t or gpu_t <= MAX_GPU_TEMP:
                break
            print(f"[WAIT] GPU {gpu_t}°C > {MAX_GPU_TEMP}°C, waiting {CHECK_INTERVAL}s...")
            time.sleep(CHECK_INTERVAL)

# ==============================
# MERGE CSV
# ==============================
merged_df = pd.concat([pd.read_csv(os.path.join(TEMP_DIR,f)) for f in os.listdir(TEMP_DIR)
                       if os.path.getsize(os.path.join(TEMP_DIR,f))>0], ignore_index=True)
merged_df.sort_values("Timestamp", inplace=True)
merged_path = os.path.join(OUTPUT_DIR,"merged_detections.csv")
merged_df.to_csv(merged_path,index=False)
print(f"[INFO] Merged detections saved: {merged_path}")

# ==============================
# CONCAT VIDEOS
# ==============================
if annotated_videos:
    first_video = cv2.VideoCapture(annotated_videos[0])
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = first_video.get(cv2.CAP_PROP_FPS)
    first_video.release()

    merged_video_path = os.path.join(OUTPUT_DIR,"merged_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    merged_writer = cv2.VideoWriter(merged_video_path, fourcc, fps, (width,height))

    for vid_path in annotated_videos:
        cap = cv2.VideoCapture(vid_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, os.path.basename(vid_path), (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            merged_writer.write(frame)
        cap.release()
    merged_writer.release()
    print(f"[INFO] Final merged video saved: {merged_video_path}")

# ==============================
# ACTIVITY PERIODS
# ==============================
def group_timestamps(ts,max_gap=10.0):
    if not ts: return []
    ts.sort()
    ts = np.array(ts)
    periods = []
    start = ts[0]
    prev = start
    for t in ts[1:]:
        if t-prev>max_gap:
            periods.append((start,prev))
            start=t
        prev=t
    periods.append((start,prev))
    return periods

activity_periods={}
for cls in CLASSES:
    times = merged_df.loc[merged_df["Class"]==cls,"Timestamp"].tolist()
    activity_periods[cls]=group_timestamps(times)

rows=[]
for cls,periods in activity_periods.items():
    for s,e in periods:
        rows.append({"Class":cls,"Start (s)":round(s,2),"End (s)":round(e,2),"Duration (s)":round(e-s,2)})
final_df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR,"activity_periods.csv")
final_df.to_csv(csv_path,index=False)
print(f"[INFO] Activity report saved: {csv_path}")

# ==============================
# GANTT CHARTS
# ==============================
def plot_gantt(periods,start,end,title,filepath,max_segments=2000):
    plt.figure(figsize=(14,7))
    y_labels = list(periods.keys())
    for i,cls in enumerate(y_labels):
        segs = [(max(s,start),min(e,end)) for s,e in periods[cls] if not (e<start or s>end)]
        if len(segs)>max_segments:
            idx = np.linspace(0,len(segs)-1,max_segments,dtype=int)
            segs=[segs[j] for j in idx]
        for s,e in segs:
            plt.plot([s,e],[i,i],linewidth=4)
    plt.yticks(range(len(y_labels)),y_labels)
    plt.xlabel("Time (s)")
    plt.ylabel("Class")
    plt.title(title)
    plt.grid(True,axis='x',linestyle='--',alpha=0.4)
    plt.tight_layout()
    plt.savefig(filepath,dpi=200)
    plt.close()

activity_df = pd.read_csv(csv_path,usecols=["Class","Start (s)","End (s)"])
activity_periods = {cls:list(zip(sub["Start (s)"],sub["End (s)"])) for cls,sub in activity_df.groupby("Class")}
total_time = activity_df["End (s)"].max()

plot_gantt(activity_periods,0,total_time,"Gantt Chart - Total Duration",os.path.join(OUTPUT_DIR,"gantt_total.png"))
half = total_time/2
plot_gantt(activity_periods,0,half,"Gantt Chart - First Half",os.path.join(OUTPUT_DIR,"gantt_half_1.png"))
plot_gantt(activity_periods,half,total_time,"Gantt Chart - Second Half",os.path.join(OUTPUT_DIR,"gantt_half_2.png"))
quarter = total_time/4
for i in range(4):
    plot_gantt(activity_periods,i*quarter,(i+1)*quarter,f"Gantt Chart - Quarter {i+1}",
               os.path.join(OUTPUT_DIR,f"gantt_quarter_{i+1}.png"))

print("[INFO] Gantt charts generated.")

# ==============================
# SCALARE PAIR ANALYSIS
# ==============================
DIST_THRESH = [50,100,275]
scal_df = merged_df[merged_df["Class"].isin(SCALARE_CLASSES)].copy()
if not scal_df.empty:
    scal_df["cx"]=(scal_df["x1"]+scal_df["x2"])/2
    scal_df["cy"]=(scal_df["y1"]+scal_df["y2"])/2
    avg_fps = np.mean([fps for _,_,fps,_ in video_infos])
    scal_df["time_bucket"] = (scal_df["Timestamp"]//1)*1
    pair_counts={th:{} for th in DIST_THRESH}
    for t,group in tqdm(scal_df.groupby("time_bucket"),desc="Scalar pair analysis"):
        coords=group[["cx","cy"]].to_numpy(dtype=np.float32)
        classes=group["Class"].to_numpy()
        n=len(coords)
        if n<2: continue
        diff_x = coords[:,0].reshape(-1,1)-coords[:,0]
        diff_y = coords[:,1].reshape(-1,1)-coords[:,1]
        dist_matrix = np.sqrt(diff_x**2+diff_y**2)
        for i in range(n):
            for j in range(i+1,n):
                c1,c2 = classes[i],classes[j]
                if c1==c2: continue
                d = dist_matrix[i,j]
                for th in DIST_THRESH:
                    if d<th:
                        key="_".join(sorted([c1,c2]))
                        pair_counts[th][key]=pair_counts[th].get(key,0)+1
    for th,counts in pair_counts.items():
        if not counts: continue
        df = pd.DataFrame(list(counts.items()),columns=["PairKey","FramesTogether"])
        df["TimeTogether_s"]=df["FramesTogether"]/avg_fps
        def split_key(k):
            for cls in SCALARE_CLASSES:
                if k.startswith(cls+"_"):
                    return pd.Series([cls,k[len(cls)+1:]])
            return pd.Series(k.split("_"))
        df[["Class1","Class2"]]=df["PairKey"].apply(split_key)
        df.to_csv(os.path.join(OUTPUT_DIR,f"scalar_pair_{th}px.csv"),index=False)
        top10 = df.sort_values("FramesTogether",ascending=False).head(10)
        top10["Pair"]=top10["Class1"]+" ↔ "+top10["Class2"]
        plt.figure(figsize=(10,6))
        plt.barh(top10["Pair"][::-1],top10["FramesTogether"][::-1],color="skyblue")
        plt.xlabel("Frames Together")
        plt.title(f"Top 10 Scalar Pairs (<{th}px)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR,f"scalar_pair_top10_{th}px.png"))
        plt.close()
print("[INFO] Scalar pair analysis completed.")
print(f"[INFO] Temporary files stored in: {TEMP_DIR}")
print("[INFO] Processing completed successfully.")
