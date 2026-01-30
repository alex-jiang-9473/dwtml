import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image
import os

# Configuration
IMAGEID = "kodim01"
LEVELS = 1
WAVELET = "db4"
OUTPUT_DIR = "results/dwt_split_yuv_channels/analysis"

def rgb_to_yuv(rgb_img):
    """Convert RGB image to YUV color space (BT.601)"""
    rgb = np.array(rgb_img, dtype=np.float32)
    
    mat = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    
    yuv = np.dot(rgb, mat.T)
    yuv[:, :, 1:] += 128.0
    
    return yuv

def analyze_channel_distribution(channel_data, channel_name):
    """Analyze and return statistics for a single channel"""
    stats = {
        'mean': np.mean(channel_data),
        'std': np.std(channel_data),
        'min': np.min(channel_data),
        'max': np.max(channel_data),
        'range': np.max(channel_data) - np.min(channel_data),
        'median': np.median(channel_data),
        'q25': np.percentile(channel_data, 25),
        'q75': np.percentile(channel_data, 75)
    }
    
    print(f"\n{channel_name} Channel Statistics:")
    print(f"  Mean:   {stats['mean']:.2f}")
    print(f"  Std:    {stats['std']:.2f}")
    print(f"  Min:    {stats['min']:.2f}")
    print(f"  Max:    {stats['max']:.2f}")
    print(f"  Range:  {stats['range']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Q25:    {stats['q25']:.2f}")
    print(f"  Q75:    {stats['q75']:.2f}")
    
    return stats

def analyze_dwt_coeffs(coeffs, channel_name, threshold_factor=1.0):
    """Analyze DWT coefficient distributions"""
    print(f"\n{'='*70}")
    print(f"{channel_name} Channel - DWT Coefficient Analysis")
    print(f"{'='*70}")
    
    # LL band
    ll_coeffs = coeffs[0]
    print(f"\nLL Band ({ll_coeffs.shape}):")
    print(f"  Mean: {np.mean(ll_coeffs):.2f}")
    print(f"  Std:  {np.std(ll_coeffs):.2f}")
    print(f"  Min:  {np.min(ll_coeffs):.2f}")
    print(f"  Max:  {np.max(ll_coeffs):.2f}")
    
    # Threshold analysis for LL
    threshold = threshold_factor * np.std(ll_coeffs)
    above_threshold = np.sum(np.abs(ll_coeffs - np.mean(ll_coeffs)) > threshold)
    sparsity_pct = 100.0 * above_threshold / ll_coeffs.size
    print(f"  Threshold (mean-centered): {threshold:.2f}")
    print(f"  Above threshold: {above_threshold}/{ll_coeffs.size} ({sparsity_pct:.1f}%)")
    
    band_stats = {'LL': {
        'shape': ll_coeffs.shape,
        'mean': np.mean(ll_coeffs),
        'std': np.std(ll_coeffs),
        'min': np.min(ll_coeffs),
        'max': np.max(ll_coeffs),
        'sparsity': sparsity_pct,
        'coeffs': ll_coeffs.flatten()
    }}
    
    # HF bands
    for level_idx, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
        print(f"\nLevel {level_idx} HF Bands:")
        
        for band_name, band_coeffs in [('cH', cH), ('cV', cV), ('cD', cD)]:
            threshold = threshold_factor * np.std(band_coeffs)
            sparse_mask = np.abs(band_coeffs) > threshold
            num_sparse = np.sum(sparse_mask)
            sparsity_pct = 100.0 * num_sparse / band_coeffs.size
            
            print(f"  {band_name} ({band_coeffs.shape}):")
            print(f"    Mean: {np.mean(band_coeffs):.2f}")
            print(f"    Std:  {np.std(band_coeffs):.2f}")
            print(f"    Min:  {np.min(band_coeffs):.2f}")
            print(f"    Max:  {np.max(band_coeffs):.2f}")
            print(f"    Threshold: {threshold:.2f}")
            print(f"    Sparse coeffs: {num_sparse}/{band_coeffs.size} ({sparsity_pct:.1f}%)")
            
            band_full_name = f"{band_name}_L{level_idx}"
            band_stats[band_full_name] = {
                'shape': band_coeffs.shape,
                'mean': np.mean(band_coeffs),
                'std': np.std(band_coeffs),
                'min': np.min(band_coeffs),
                'max': np.max(band_coeffs),
                'sparsity': sparsity_pct,
                'num_sparse': num_sparse,
                'coeffs': band_coeffs.flatten()
            }
    
    return band_stats

def plot_distributions(y_stats, u_stats, v_stats, output_dir):
    """Create distribution plots for all channels and bands"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Original channel distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (channel_name, channel_coeffs) in enumerate([('Y', y_stats['LL']['coeffs']), 
                                                            ('U', u_stats['LL']['coeffs']), 
                                                            ('V', v_stats['LL']['coeffs'])]):
        axes[idx].hist(channel_coeffs, bins=100, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{channel_name} Channel (LL Band) Distribution')
        axes[idx].set_xlabel('Coefficient Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_ll_distributions.png'), dpi=150)
    print(f"\nSaved: {os.path.join(output_dir, 'channel_ll_distributions.png')}")
    plt.close()
    
    # Plot 2: HF bands comparison
    hf_bands = ['cH_L1', 'cV_L1', 'cD_L1', 'cH_L2', 'cV_L2', 'cD_L2']
    
    for band_name in hf_bands:
        if band_name not in y_stats:
            continue
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'{band_name} Coefficient Distribution Across Channels', fontsize=14)
        
        for idx, (channel_name, stats) in enumerate([('Y', y_stats), ('U', u_stats), ('V', v_stats)]):
            if band_name in stats:
                coeffs = stats[band_name]['coeffs']
                axes[idx].hist(coeffs, bins=100, alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'{channel_name} - {band_name}')
                axes[idx].set_xlabel('Coefficient Value')
                axes[idx].set_ylabel('Frequency')
                axes[idx].axvline(0, color='red', linestyle='--', alpha=0.5, label='Zero')
                
                # Mark threshold
                threshold = stats[band_name]['std']
                axes[idx].axvline(threshold, color='green', linestyle='--', alpha=0.5, label=f'Threshold (±{threshold:.2f})')
                axes[idx].axvline(-threshold, color='green', linestyle='--', alpha=0.5)
                
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{band_name}_comparison.png'), dpi=150)
        print(f"Saved: {os.path.join(output_dir, f'{band_name}_comparison.png')}")
        plt.close()
    
    # Plot 3: Sparsity comparison
    bands = ['LL']
    y_sparsity = [y_stats['LL']['sparsity']]
    u_sparsity = [u_stats['LL']['sparsity']]
    v_sparsity = [v_stats['LL']['sparsity']]
    
    for band_name in hf_bands:
        if band_name in y_stats:
            bands.append(band_name)
            y_sparsity.append(y_stats[band_name]['sparsity'])
            u_sparsity.append(u_stats[band_name]['sparsity'])
            v_sparsity.append(v_stats[band_name]['sparsity'])
    
    x = np.arange(len(bands))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, y_sparsity, width, label='Y', alpha=0.8)
    ax.bar(x, u_sparsity, width, label='U', alpha=0.8)
    ax.bar(x + width, v_sparsity, width, label='V', alpha=0.8)
    
    ax.set_xlabel('Band')
    ax.set_ylabel('Sparsity (%)')
    ax.set_title('Coefficient Sparsity Comparison (threshold = 1.0 × std)')
    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sparsity_comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'sparsity_comparison.png')}")
    plt.close()
    
    # Plot 4: Standard deviation comparison
    y_stds = [y_stats[b]['std'] for b in bands]
    u_stds = [u_stats[b]['std'] for b in bands]
    v_stds = [v_stats[b]['std'] for b in bands]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, y_stds, width, label='Y', alpha=0.8)
    ax.bar(x, u_stds, width, label='U', alpha=0.8)
    ax.bar(x + width, v_stds, width, label='V', alpha=0.8)
    
    ax.set_xlabel('Band')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Coefficient Standard Deviation Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'std_comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'std_comparison.png')}")
    plt.close()

def main():
    # Load image
    img_path = f"kodak-dataset/{IMAGEID}.png"
    img_rgb = Image.open(img_path)
    print(f"Loaded image: {img_path}")
    print(f"Size: {img_rgb.size}")
    
    # Convert to YUV
    img_yuv = rgb_to_yuv(img_rgb)
    y_channel = img_yuv[:, :, 0]
    u_channel = img_yuv[:, :, 1]
    v_channel = img_yuv[:, :, 2]
    
    # Analyze original channel distributions
    print("\n" + "="*70)
    print("ORIGINAL CHANNEL ANALYSIS")
    print("="*70)
    
    y_channel_stats = analyze_channel_distribution(y_channel, 'Y')
    u_channel_stats = analyze_channel_distribution(u_channel, 'U')
    v_channel_stats = analyze_channel_distribution(v_channel, 'V')
    
    # Perform DWT
    print(f"\nPerforming {LEVELS}-level DWT with {WAVELET} wavelet...")
    y_coeffs = pywt.wavedec2(y_channel, WAVELET, level=LEVELS)
    u_coeffs = pywt.wavedec2(u_channel, WAVELET, level=LEVELS)
    v_coeffs = pywt.wavedec2(v_channel, WAVELET, level=LEVELS)
    
    # Analyze DWT coefficients
    y_dwt_stats = analyze_dwt_coeffs(y_coeffs, 'Y')
    u_dwt_stats = analyze_dwt_coeffs(u_coeffs, 'U')
    v_dwt_stats = analyze_dwt_coeffs(v_coeffs, 'V')
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    plot_distributions(y_dwt_stats, u_dwt_stats, v_dwt_stats, OUTPUT_DIR)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nKey Insights:")
    print(f"  Y channel has higher variation (std={y_channel_stats['std']:.2f})")
    print(f"  U channel has low variation (std={u_channel_stats['std']:.2f})")
    print(f"  V channel has low variation (std={v_channel_stats['std']:.2f})")
    print(f"\n  This explains why U/V need lower thresholds for effective compression!")
    print(f"  Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
