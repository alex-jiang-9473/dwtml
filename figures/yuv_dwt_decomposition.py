import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image
import os

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

def normalize_band(band):
    """Normalize a single band to [0, 255] range for visualization"""
    band_min = band.min()
    band_max = band.max()
    if band_max - band_min > 1e-6:  # Avoid division by zero
        normalized = 255.0 * (band - band_min) / (band_max - band_min)
    else:
        normalized = np.zeros_like(band)
    return normalized

def create_dwt_composite(coeffs, normalize_independently=True):
    """Create a single composite image showing all DWT coefficients in hierarchical layout
    
    Args:
        coeffs: Output from pywt.wavedec2 - (LL, (LH2, HL2, HH2), (LH1, HL1, HH1))
        where level 2 is coarser (same size as LL) and level 1 is finer (larger)
        normalize_independently: If True, normalize each band to [0, 255] independently
                                If False, use raw coefficient values
    
    Returns:
        Composite image with all DWT bands arranged hierarchically
    """
    # Extract coefficients
    # For 2-level DWT: coeffs[0]=LL, coeffs[1]=level2 details, coeffs[2]=level1 details
    LL = coeffs[0].copy()
    (cH2, cV2, cD2) = coeffs[1]
    cH2, cV2, cD2 = cH2.copy(), cV2.copy(), cD2.copy()
    (cH1, cV1, cD1) = coeffs[2]
    cH1, cV1, cD1 = cH1.copy(), cV1.copy(), cD1.copy()
    
    # Apply independent normalization if requested
    if normalize_independently:
        LL = normalize_band(LL)
        cH2 = normalize_band(cH2)
        cV2 = normalize_band(cV2)
        cD2 = normalize_band(cD2)
        cH1 = normalize_band(cH1)
        cV1 = normalize_band(cV1)
        cD1 = normalize_band(cD1)
    
    # Get dimensions
    h_ll, w_ll = LL.shape  # e.g., (304, 405)
    h_l1, w_l1 = cH1.shape  # e.g., (601, 803)
    
    # Create canvas for composite (same size as level 1 bands × 2)
    h_total = h_l1 * 2
    w_total = w_l1 * 2
    composite = np.zeros((h_total, w_total), dtype=np.float32)
    
    # Top-left quadrant contains: 2×2 grid of (LL, cH2, cV2, cD2)
    # All these are the same size as LL
    
    # Position LL in top-left of the top-left quadrant
    composite[0:h_ll, 0:w_ll] = LL
    
    # Position cH2 (horizontal details L2) to the right of LL
    composite[0:h_ll, w_ll:2*w_ll] = cH2
    
    # Position cV2 (vertical details L2) below LL
    composite[h_ll:2*h_ll, 0:w_ll] = cV2
    
    # Position cD2 (diagonal details L2) in bottom-right of top-left quadrant
    composite[h_ll:2*h_ll, w_ll:2*w_ll] = cD2
    
    # Top-right quadrant: cH1 (horizontal details at level 1)
    composite[0:h_l1, w_l1:w_total] = cH1
    
    # Bottom-left quadrant: cV1 (vertical details at level 1)
    composite[h_l1:h_total, 0:w_l1] = cV1
    
    # Bottom-right quadrant: cD1 (diagonal details at level 1)
    composite[h_l1:h_total, w_l1:w_total] = cD1
    
    return composite

def main():
    # Configuration
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Option 1: Use Barns image in parent directory
    IMAGE_PATH = os.path.join(os.path.dirname(script_dir), "figures","Barns_grand_tetons.jpg")
    # Option 2: Use kodak dataset image
    # IMAGE_PATH = os.path.join(os.path.dirname(script_dir), "kodak-dataset", "kodim08.png")
    WAVELET = "db4"
    LEVELS = 2
    
    # Load image
    print(f"Loading image: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}")
        print(f"Please place 'Barns_grand_tetons.jpg' in the project root directory")
        print(f"Or update IMAGE_PATH in the script to point to your image")
        return
    
    img_rgb = Image.open(IMAGE_PATH)
    print(f"  Size: {img_rgb.size}")
    
    # Convert to YUV
    print(f"\nConverting to YUV...")
    img_yuv = rgb_to_yuv(img_rgb)
    y_channel = img_yuv[:, :, 0]
    u_channel = img_yuv[:, :, 1]
    v_channel = img_yuv[:, :, 2]
    
    print(f"YUV Statistics:")
    print(f"  Y: mean={np.mean(y_channel):.2f}, std={np.std(y_channel):.2f}, range=[{np.min(y_channel):.2f}, {np.max(y_channel):.2f}]")
    print(f"  U: mean={np.mean(u_channel):.2f}, std={np.std(u_channel):.2f}, range=[{np.min(u_channel):.2f}, {np.max(u_channel):.2f}]")
    print(f"  V: mean={np.mean(v_channel):.2f}, std={np.std(v_channel):.2f}, range=[{np.min(v_channel):.2f}, {np.max(v_channel):.2f}]")
    
    # Perform 2-level DWT on each channel
    print(f"\nPerforming {LEVELS}-level DWT with {WAVELET} wavelet...")
    y_coeffs = pywt.wavedec2(y_channel, WAVELET, level=LEVELS)
    u_coeffs = pywt.wavedec2(u_channel, WAVELET, level=LEVELS)
    v_coeffs = pywt.wavedec2(v_channel, WAVELET, level=LEVELS)
    
    print(f"  Y LL shape: {y_coeffs[0].shape}")
    print(f"  Y Level 1 detail shapes: {y_coeffs[1][0].shape}")
    print(f"  Y Level 2 detail shapes: {y_coeffs[2][0].shape}")
    
    # Print coefficient statistics to understand why U/V high frequencies appear black
    print(f"\nCoefficient Statistics:")
    print(f"  Y Channel:")
    print(f"    LL:  min={y_coeffs[0].min():.2f}, max={y_coeffs[0].max():.2f}, std={y_coeffs[0].std():.2f}")
    print(f"    cH2: min={y_coeffs[1][0].min():.2f}, max={y_coeffs[1][0].max():.2f}, std={y_coeffs[1][0].std():.2f}")
    print(f"    cH1: min={y_coeffs[2][0].min():.2f}, max={y_coeffs[2][0].max():.2f}, std={y_coeffs[2][0].std():.2f}")
    
    print(f"  U Channel:")
    print(f"    LL:  min={u_coeffs[0].min():.2f}, max={u_coeffs[0].max():.2f}, std={u_coeffs[0].std():.2f}")
    print(f"    cH2: min={u_coeffs[1][0].min():.2f}, max={u_coeffs[1][0].max():.2f}, std={u_coeffs[1][0].std():.2f}")
    print(f"    cH1: min={u_coeffs[2][0].min():.2f}, max={u_coeffs[2][0].max():.2f}, std={u_coeffs[2][0].std():.2f}")
    
    print(f"  V Channel:")
    print(f"    LL:  min={v_coeffs[0].min():.2f}, max={v_coeffs[0].max():.2f}, std={v_coeffs[0].std():.2f}")
    print(f"    cH2: min={v_coeffs[1][0].min():.2f}, max={v_coeffs[1][0].max():.2f}, std={v_coeffs[1][0].std():.2f}")
    print(f"    cH1: min={v_coeffs[2][0].min():.2f}, max={v_coeffs[2][0].max():.2f}, std={v_coeffs[2][0].std():.2f}")
    
    print(f"\n  Note: U/V high-frequency coefficients (cH1, cV1, cD1) have very small magnitudes")
    print(f"        compared to Y channel. This is normal - chrominance has less high-frequency content.")
    
    # Create composite visualizations
    print("\nGenerating composite DWT visualizations...")
    
    # Y channel composite
    y_composite = create_dwt_composite(y_coeffs)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(y_composite, cmap='gray')
    ax.set_title('Y Channel - 2-Level DWT (Composite)', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('yuv_dwt_y_composite.png', dpi=150, bbox_inches='tight')
    print("  Saved: yuv_dwt_y_composite.png")
    plt.close()
    
    # U channel composite
    u_composite = create_dwt_composite(u_coeffs)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(u_composite, cmap='gray')
    ax.set_title('U Channel - 2-Level DWT (Composite)', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('yuv_dwt_u_composite.png', dpi=150, bbox_inches='tight')
    print("  Saved: yuv_dwt_u_composite.png")
    plt.close()
    
    # V channel composite
    v_composite = create_dwt_composite(v_coeffs)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(v_composite, cmap='gray')
    ax.set_title('V Channel - 2-Level DWT (Composite)', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('yuv_dwt_v_composite.png', dpi=150, bbox_inches='tight')
    print("  Saved: yuv_dwt_v_composite.png")
    plt.close()
    
    # Create a combined view of all three channels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(y_composite, cmap='gray')
    axes[0].set_title('Y Channel DWT', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(u_composite, cmap='gray')
    axes[1].set_title('U Channel DWT', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(v_composite, cmap='gray')
    axes[2].set_title('V Channel DWT', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('yuv_dwt_all_channels.png', dpi=150, bbox_inches='tight')
    print("  Saved: yuv_dwt_all_channels.png")
    
    print("\nDone! All composite visualizations saved.")
    plt.show()

if __name__ == "__main__":
    main()
