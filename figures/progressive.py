import pandas as pd
import matplotlib.pyplot as plt

csv_path = "figures/img2_progressive.csv"
df = pd.read_csv(csv_path, skipinitialspace=True)
df.columns = df.columns.str.strip()

# Parse numbers
df["Training Time (s)"] = pd.to_numeric(df["Training Time (s)"], errors="coerce")
df["No. of Params"] = pd.to_numeric(df["No. of Params"], errors="coerce")
df["PSNR (dB)"] = pd.to_numeric(df["PSNR (dB)"], errors="coerce")

# Sequential order in CSV
df["Step"] = range(1, len(df) + 1)

# Cumulative time along the sequence
df["Cumulative Time (s)"] = df["Training Time (s)"].cumsum()

# Print with sequence
print(df[["Step", "Frequency Band", "Training Time (s)", "No. of Params", "Cumulative Time (s)", "PSNR (dB)"]])

# Plot PSNR vs cumulative time
plt.figure(figsize=(10, 6))
plt.plot(df["Cumulative Time (s)"], df["PSNR (dB)"], marker="o")

# Highlight specific points (index 0, 9, 18)
highlight_indices = [0, 9, 18]
for idx in highlight_indices:
	if idx < len(df):
		row = df.iloc[idx]
		plt.plot(row["Cumulative Time (s)"], row["PSNR (dB)"], 
		         marker="o", markersize=10, color="orange", zorder=5)

plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("PSNR (dB)", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True, alpha=0.3)

# Set x-axis starting from 30s
max_time = df["Cumulative Time (s)"].max()
plt.xlim(30, max_time + 5)
xticks = [30] + list(range(35, int(max_time) + 5, 5))
plt.xticks(xticks, fontsize=12)

# Label each point with band name and parameter count (alternate up/down to reduce overlap)
highlight_indices = [0, 9, 18]
for idx, row in df.iterrows():
	if pd.notna(row["PSNR (dB)"]):
		y_offset = 5 if idx % 2 == 0 and idx!=18 and idx!=16 else -5
		va = "bottom" if y_offset > 0 else "top"
		
		# Add PSNR to label for highlighted points
		if idx in highlight_indices:
			params_label = f"{row['Frequency Band']}\n{int(row['No. of Params']):,}\n{row['PSNR (dB)']:.2f} dB"
			fontweight = "bold"
		else:
			params_label = f"{row['Frequency Band']}\n{int(row['No. of Params']):,}"
			fontweight = "normal"
		
		plt.annotate(
			params_label,
			(row["Cumulative Time (s)"], row["PSNR (dB)"]),
			textcoords="offset points",
			xytext=(5, y_offset),
			ha="center",
			va=va,
			fontsize=8,
			fontweight=fontweight,
		)

# Add legend explaining label format
from matplotlib.lines import Line2D
legend_elements = [
	Line2D([0], [0], color='none', marker='', linestyle='', label='Label:'),
	Line2D([0], [0], color='none', marker='', linestyle='', label='  • Frequency Band'),
	Line2D([0], [0], color='none', marker='', linestyle='', label='  • No. of Model Params'),
	Line2D([0], [0], color='none', marker='', linestyle='', label='  • Cumulative PSNR'),
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=12, handlelength=0, handletextpad=0)

plt.tight_layout()
plt.show()