
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("events.csv")

P1_START = pd.Timestamp("2015-05-03 00:00:00", tz="UTC")
P1_END   = pd.Timestamp("2015-05-17 23:59:59", tz="UTC")

P2_START = pd.Timestamp("2015-09-01 00:00:00", tz="UTC")
P2_END   = pd.Timestamp("2015-09-18 23:59:59", tz="UTC")

df = pd.read_csv(INPUT_FILE)

df["datetime"] = pd.to_datetime(
    df["timestamp"], unit="ms", utc=True
)

p1 = df[
    (df["datetime"] >= P1_START) &
    (df["datetime"] <= P1_END)
].copy()

p2 = df[
    (df["datetime"] >= P2_START) &
    (df["datetime"] <= P2_END)
].copy()

p1.to_csv("events_P1_training.csv", index=False)
p2.to_csv("events_P2_testing.csv", index=False)

print("HASIL FILTER RETAILROCKET")
print(f"Total data awal : {len(df):,}")
print(f"Data P1 Training: {len(p1):,}")
print(f"Data P2 Testing : {len(p2):,}")

print("\nRentang waktu P1:")
print(p1["datetime"].min(), "sampai", p1["datetime"].max())

print("\nRentang waktu P2:")
print(p2["datetime"].min(), "sampai", p2["datetime"].max())

print("\nFile CSV berhasil disimpan.")
