import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), 'gpu_time_coin_dwt_cmp.csv')
OUT_PNG = os.path.join(os.path.dirname(__file__), 'coin_vs_dwt_comparison.png')


def load_and_prepare(path):
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Extract method (COIN/DWT) and variant index (1/2/3)
    df['method'] = df['Method'].str.split('-', n=1).str[0]
    df['variant'] = df['Method'].str.extract(r'-(\d+)').astype(int)
    return df


def plot_comparison(df, out_path):
    # Pivot for grouped bars by variant
    time_pivot = df.pivot(index='variant', columns='method', values='GPU Time (s)')
    mem_pivot = df.pivot(index='variant', columns='method', values='GPU Memory (MB)')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # GPU Time
    time_pivot.plot(kind='bar', ax=axes[0], rot=0)
    axes[0].set_title('GPU Time (s) — COIN vs DWT')
    axes[0].set_xlabel('Variant')
    axes[0].set_ylabel('GPU Time (s)')

    # GPU Memory
    mem_pivot.plot(kind='bar', ax=axes[1], rot=0)
    axes[1].set_title('Peak GPU Memory (MB) — COIN vs DWT')
    axes[1].set_xlabel('Variant')
    axes[1].set_ylabel('GPU Memory (MB)')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print('Saved comparison figure to', out_path)


if __name__ == '__main__':
    df = load_and_prepare(CSV_PATH)
    plot_comparison(df, OUT_PNG)
