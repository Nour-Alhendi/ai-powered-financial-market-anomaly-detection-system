import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/detection")
OUTPUT_DIR = Path("data/detection")

def combine(file_path):
    df = pd.read_parquet(file_path)
    df["anomaly_score"] = (
        (df["z_anomaly"] == True).astype(int) + 
        (df["if_anomaly"] == -1).astype(int) +
        (df["ae_anomaly"] == True).astype(int)
    )
    df["combined_anomaly"] = df["anomaly_score"] > 0 
    return df

def run_combine():
    for file in INPUT_DIR.glob("*.parquet"):
        df = combine(file)
        df.to_parquet(OUTPUT_DIR /file.name)
        print(f"Saved: {file.name}")
    
if __name__ == "__main__":
    run_combine()

