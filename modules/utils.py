import re

def extract_datetime_key(filename: str) -> str:
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        date_part, time_part = match.groups()
        return f"{date_part}{time_part}"
    return filename