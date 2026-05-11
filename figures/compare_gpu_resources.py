import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CSV = Path("figures/gpu_time_coin_dwt_cmp.csv")
DEFAULT_OUTPUT = Path("figures/coin_vs_dwt_comparison.png")


def load_csv(path: Path) -> pd.DataFrame:
    """Load and normalize the comparison CSV."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df['method'] = df['Method'].str.split('-', n=1).str[0]
    df['variant'] = df['Method'].str.extract(r'-(\d+)').astype(int)
    return df


def build_method_data(df: pd.DataFrame, method: str) -> dict:
    """Extract data for a specific method (COIN or DWT)."""
    method_df = df[df['method'] == method].sort_values('variant')
    return {
        "method": method,
        "variants": method_df['variant'].tolist(),
        "time_s": method_df['GPU Time (s)'].tolist(),
        "memory_mb": method_df['GPU Memory (MB)'].tolist(),
        "bpp": method_df['BPP'].tolist(),
        "psnr_db": method_df['PSNR (dB)'].tolist(),
        "total_time_s": method_df['GPU Time (s)'].sum(),
        "peak_memory_mb": method_df['GPU Memory (MB)'].max(),
    }


def plot_comparison(coin_data: dict, dwt_data: dict, output_path: Path) -> None:
    """Plot COIN vs DWT GPU time and memory separately with legend at top."""
    x = range(1, 4)  # groups 1, 2, 3
    width = 0.35

    # Figure 1: GPU Time
    fig_time, ax_time = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars1 = ax_time.bar(
        [i - width / 2 for i in x],
        coin_data["time_s"],
        width,
        label="COIN",
        color="#f28e2b",
        edgecolor="black",
        linewidth=0.8,
    )
    bars2 = ax_time.bar(
        [i + width / 2 for i in x],
        dwt_data["time_s"],
        width,
        label="DWT",
        color="#4e79a7",
        edgecolor="black",
        linewidth=0.8,
    )
    ax_time.set_ylabel("GPU Time (s)", fontsize=14, fontweight="bold")
    ax_time.set_title("GPU Training Time Comparison", fontsize=16, fontweight="bold")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels([f"Group {i}" for i in x], fontsize=12)
    ax_time.legend(fontsize=13, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2)
    ax_time.tick_params(axis="y", labelsize=11)
    ax_time.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_time.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}s",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    time_output = Path(str(output_path).replace(".png", "_time.png"))
    fig_time.savefig(time_output, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {time_output}")

    # Figure 2: GPU Memory
    fig_mem, ax_mem = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars3 = ax_mem.bar(
        [i - width / 2 for i in x],
        coin_data["memory_mb"],
        width,
        label="COIN",
        color="#ffbe7d",
        edgecolor="black",
        linewidth=0.8,
    )
    bars4 = ax_mem.bar(
        [i + width / 2 for i in x],
        dwt_data["memory_mb"],
        width,
        label="DWT",
        color="#7da0ca",
        edgecolor="black",
        linewidth=0.8,
    )
    ax_mem.set_ylabel("Peak GPU Memory (MB)", fontsize=14, fontweight="bold")
    ax_mem.set_title("Peak GPU Memory Comparison", fontsize=16, fontweight="bold")
    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels([f"Group {i}" for i in x], fontsize=12)
    ax_mem.legend(fontsize=13, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2)
    ax_mem.tick_params(axis="y", labelsize=11)
    ax_mem.grid(axis="y", alpha=0.3)

    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax_mem.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    mem_output = Path(str(output_path).replace(".png", "_memory.png"))
    fig_mem.savefig(mem_output, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {mem_output}")

    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GPU time and memory: COIN vs DWT.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the comparison CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to save the output figure")
    parser.add_argument("--no-show", action="store_true", help="Do not open the figure window")
    args = parser.parse_args()

    df = load_csv(args.csv)
    coin_data = build_method_data(df, "COIN")
    dwt_data = build_method_data(df, "DWT")

    print("COIN summary:")
    print(f"  Total time: {coin_data['total_time_s']:.2f} s")
    print(f"  Peak memory: {coin_data['peak_memory_mb']:.2f} MB")
    print("DWT summary:")
    print(f"  Total time: {dwt_data['total_time_s']:.2f} s")
    print(f"  Peak memory: {dwt_data['peak_memory_mb']:.2f} MB")

    os.makedirs(args.output.parent, exist_ok=True)
    plot_comparison(coin_data, dwt_data, args.output)

    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
