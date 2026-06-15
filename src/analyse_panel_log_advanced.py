import pandas as pd
import matplotlib.pyplot as plt

# Load time series file (modify if filename is different)
df = pd.read_csv("panel_timeseries_log_withxy.csv", engine="python")

# Distance basic stats
stats = {
    "count": df["Distance_cm"].count(),
    "mean": df["Distance_cm"].mean(),
    "std": df["Distance_cm"].std(),
    "min": df["Distance_cm"].min(),
    "median": df["Distance_cm"].median(),
    "max": df["Distance_cm"].max(),
}

# Save stats to a new CSV
stats_df = pd.DataFrame([stats])
stats_df.to_csv("distance_stats_report.csv", index=False)

print("Summary statistics saved to distance_stats_report.csv")
print(stats_df)

# Plot 1: Distance time series
plt.figure(figsize=(10,5))
plt.plot(pd.to_datetime(df["Time"]), df["Distance_cm"], label="Raw Distance")
plt.xlabel("Time")
plt.ylabel("Distance (cm)")
plt.title("Distance Sensor Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("distance_timeseries.png")
plt.show()

# Plot 2: Histogram of Distance values
plt.figure(figsize=(8,5))
plt.hist(df["Distance_cm"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distance Value Distribution")
plt.xlabel("Distance (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("distance_histogram.png")
plt.show()

# Plot 3: Show min, median, max as separate horizontal lines
plt.figure(figsize=(10,5))
plt.plot(pd.to_datetime(df["Time"]), df["Distance_cm"], alpha=0.5, label="Distance")
for stat, value in [("Min", stats["min"]), ("Max", stats["max"]), ("Median", stats["median"]), ("Mean", stats["mean"])]:
    plt.axhline(y=value, linestyle="--", label=f"{stat}: {value:.2f}")
plt.xlabel("Time")
plt.ylabel("Distance (cm)")
plt.title("Distance with Min/Max/Mean/Median Lines")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("distance_statlines.png")
plt.show()
