from modules.model_loader import load_model
from modules.video_utils import discover_videos, get_video_infos
from modules.detection import process_videos
from modules.csv_utils import merge_csvs
from modules.analysis.activity_analysis import generate_activity_periods
from modules.analysis.gantt_charts import generate_gantt_charts
from modules.analysis.scalar_pair_analysis import analyze_scalar_pairs
from config import OUTPUT_DIR

def main():
    model = load_model()
    video_files = discover_videos('.avi')
    if not video_files:
        raise RuntimeError('No .avi files found in videos directory')

    video_infos = get_video_infos(video_files)
    temp_csvs, annotated_videos = process_videos(model, video_infos)
    merged_csv = merge_csvs()
    activity_csv = f"{OUTPUT_DIR}/activity_periods.csv"
    generate_activity_periods(merged_csv, activity_csv)
    generate_gantt_charts(activity_csv)
    analyze_scalar_pairs(merged_csv, OUTPUT_DIR)
    print("[INFO] Processing completed successfully.")

if __name__=="__main__":
    main()
