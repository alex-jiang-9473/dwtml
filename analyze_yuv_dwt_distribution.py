import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image
import os

# Configuration
IMAGEID = "kodim02"
LEVELS = 2
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

def analyze_dwt_coeffs(coeffs, channel_name, threshold_factor=1.5):
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
    
    # Create a comprehensive combined figure
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: LL band distributions for Y, U, V
    for idx, (channel_name, stats) in enumerate([('Y', y_stats), ('U', u_stats), ('V', v_stats)]):
        ax = fig.add_subplot(gs[0, idx])
        channel_coeffs = stats['LL']['coeffs']
        ax.hist(channel_coeffs, bins=100, alpha=0.7, edgecolor='black', color=f'C{idx}')
        ax.set_title(f'{channel_name} Channel (LL Band)', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # Add statistics and threshold
        mean_val = stats['LL']['mean']
        std_val = stats['LL']['std']
        threshold = 1.5 * std_val
        
        # Mark threshold lines (mean-centered for LL band)
        ax.axvline(mean_val + threshold, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Threshold (±{threshold:.2f})')
        ax.axvline(mean_val - threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(mean_val, color='green', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Mean ({mean_val:.2f})')
        
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        ax.text(0.02, 0.98, f'μ = {mean_val:.2f}\nσ = {std_val:.2f}\nThreshold = 1.5σ', 
                transform=ax.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Row 2-3: Selected HF bands (cH_L1, cV_L1, cD_L1 for L1 and L2)
    hf_bands_to_plot = ['cH_L1', 'cV_L1', 'cD_L1', 'cH_L2', 'cV_L2', 'cD_L2']
    
    for band_idx, band_name in enumerate(hf_bands_to_plot):
        row = 1 + band_idx // 3
        col = band_idx % 3
        
        if band_name not in y_stats:
            continue
            
        ax = fig.add_subplot(gs[row, col])
        
        # Plot all three channels overlaid
        for channel_name, stats, color in [('Y', y_stats, 'C0'), ('U', u_stats, 'C1'), ('V', v_stats, 'C2')]:
            if band_name in stats:
                coeffs = stats[band_name]['coeffs']
                ax.hist(coeffs, bins=80, alpha=0.4, label=channel_name, color=color, edgecolor='black')
        
        ax.set_title(f'{band_name} Coefficients', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # Add threshold lines (zero-centered for HF bands)
        thresholds = []
        for channel_name, stats, color in [('Y', y_stats, 'C0'), ('U', u_stats, 'C1'), ('V', v_stats, 'C2')]:
            if band_name in stats:
                threshold = 1.5 * stats[band_name]['std']
                thresholds.append(threshold)
        
        if thresholds:
            max_threshold = max(thresholds)
            ax.axvline(max_threshold, color='red', linestyle='--', linewidth=2, 
                      alpha=0.7, label=f'Threshold (±{max_threshold:.2f})')
            ax.axvline(-max_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.axvline(0, color='green', linestyle=':', linewidth=1.5, alpha=0.6, label='Zero')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Add overall title
    fig.suptitle('YUV DWT Coefficient Analysis - Complete Overview', fontsize=16, fontweight='bold', y=0.995)
    
    # Save combined figure
    combined_path = os.path.join(output_dir, 'combined_distribution_analysis.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved combined analysis: {combined_path}")
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
