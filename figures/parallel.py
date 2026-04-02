import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

csv_path = "figures/img2_parallel.csv"
df = pd.read_csv(csv_path, skipinitialspace=True)
df.columns = df.columns.str.strip()

# Parse numbers
df["Training Time (s)"] = pd.to_numeric(df["Training Time (s)"], errors="coerce")
df["No. of Params"] = pd.to_numeric(df["No. of Params"], errors="coerce")

# Calculate cumulative time for each thread (bands are executed sequentially within each thread)
df["Start Time"] = 0.0
df["End Time"] = 0.0

for thread in df["Thread"].unique():
    thread_mask = df["Thread"] == thread
    thread_data = df[thread_mask].copy()
    
    cumulative_time = 0
    for idx in thread_data.index:
        df.loc[idx, "Start Time"] = cumulative_time
        df.loc[idx, "End Time"] = cumulative_time + df.loc[idx, "Training Time (s)"]
        cumulative_time = df.loc[idx, "End Time"]

# Print summary
print("Thread Summary:")
print(df.groupby("Thread").agg({
    "Training Time (s)": "sum",
    "No. of Params": "sum",
    "Frequency Band": "count"
}).rename(columns={"Frequency Band": "Band Count"}))

# Create Gantt chart
fig, ax = plt.subplots(figsize=(14, 8))

# Get unique threads
threads = df["Thread"].unique()
thread_positions = {thread: i for i, thread in enumerate(sorted(threads))}

# Color map based on channel (Y, U, V)
channel_colors = {
    'Y': '#e7d7c0',  
    'U': '#7f8f6d',  
    'V': '#ebac34', 
}

# Plot each band as a horizontal bar
for idx, row in df.iterrows():
    thread = row["Thread"]
    y_pos = thread_positions[thread]
    start = row["Start Time"]
    duration = row["Training Time (s)"]
    
    # Determine color based on channel
    band = row["Frequency Band"]
    channel = band[0]  # First character is Y, U, or V
    color = channel_colors.get(channel, '#808080')  # Default to gray if not found
    
    # Draw bar
    ax.barh(y_pos, duration, left=start, height=0.6, 
            color=color, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add label inside bar
    label = f"{row['Frequency Band']}\n{row['Training Time (s)']:.1f}s\n{int(row['No. of Params']):,}"
    ax.text(start + duration/2, y_pos, label, 
            ha='center', va='center', fontsize=7, fontweight='normal')

# Formatting
ax.set_yticks(range(len(threads)))
ax.set_yticklabels([f"{t}" for t in sorted(threads)])
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Thread", fontsize=12)
ax.grid(axis='x', alpha=0.3)

# Add vertical line at max time
max_time = df["End Time"].max()
ax.axvline(x=max_time, color='red', linestyle='--', linewidth=2, label=f'Total Time: {max_time:.1f}s')

# Add legend for channels
legend_elements = [
    Patch(facecolor='#e7d7c0', edgecolor='black', label='Y Channel'),
    Patch(facecolor='#7f8f6d', edgecolor='black', label='U Channel'),
    Patch(facecolor='#ebac34', edgecolor='black', label='V Channel'),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Total Time: {max_time:.1f}s')
]
ax.legend(handles=legend_elements, loc='upper center', fontsize=9, ncol=4, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout()
plt.show()

# Print total statistics
print(f"\nTotal Parallel Execution Time: {max_time:.1f}s")
print(f"Total Training Time (sum of all threads): {df['Training Time (s)'].sum():.1f}s")
print(f"Total Parameters: {int(df['No. of Params'].sum()):,}")
print(f"Parallelization Efficiency: {(df['Training Time (s)'].sum() / max_time):.2f}x")
