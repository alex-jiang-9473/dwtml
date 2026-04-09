# pip install pywavelets pillow
import numpy as np
import pywt
from PIL import Image

# ---------------------------
# CONFIG: hard-code number of levels
IMAGEID = "kodim02"
LEVELS = 1        # change this to 1,2,3...
WAVELET = "db4"   # change to 'db1','db4','sym4', etc.
# ---------------------------

def ycbcr_to_rgb_channel(y_val, cb_val, cr_val):
    """Convert YCbCr single channel visualization to RGB for color display"""
    # Determine dimensions from the array parameter
    if isinstance(y_val, np.ndarray):
        height, width = y_val.shape
    elif isinstance(cb_val, np.ndarray):
        height, width = cb_val.shape
    elif isinstance(cr_val, np.ndarray):
        height, width = cr_val.shape
    else:
        height, width = 256, 256
    
    if isinstance(y_val, (int, float)):
        Y = np.full((height, width), y_val, dtype=np.uint8)
    else:
        Y = y_val.astype(np.uint8)
    
    if isinstance(cb_val, (int, float)):
        Cb = np.full((height, width), cb_val, dtype=np.uint8)
    else:
        Cb = cb_val.astype(np.uint8)
        
    if isinstance(cr_val, (int, float)):
        Cr = np.full((height, width), cr_val, dtype=np.uint8)
    else:
        Cr = cr_val.astype(np.uint8)
    
    # Merge and convert to RGB
    ycbcr_img = Image.merge("YCbCr", [Image.fromarray(Y), Image.fromarray(Cb), Image.fromarray(Cr)])
    rgb_img = ycbcr_img.convert("RGB")
    return rgb_img

def save_dwt_channel(channel_data, channel_name, imageid, levels, wavelet):
    """Process and save DWT for a single channel"""
    # Multi-level 2D DWT
    coeffs = pywt.wavedec2(channel_data, wavelet=wavelet, level=levels)
    
    # Print coefficient information
    print(f"\n{channel_name} Channel DWT Coefficients:")
    print(f"  LL (approx):")
    print(f"    Shape: {coeffs[0].shape}")
    print(f"    Min: {coeffs[0].min():.4f}, Max: {coeffs[0].max():.4f}, Mean: {coeffs[0].mean():.4f}, Std: {coeffs[0].std():.4f}")
    print(f"    Sample values (top-left 3x3):\n{coeffs[0][:3, :3]}")
    
    for level_idx in range(1, len(coeffs)):
        cH, cV, cD = coeffs[level_idx]
        print(f"  Level {level_idx}:")
        print(f"    cH: shape={cH.shape}, min={cH.min():.4f}, max={cH.max():.4f}, mean={cH.mean():.4f}, std={cH.std():.4f}")
        print(f"        Sample (top-left 3x3):\n{cH[:3, :3]}")
        print(f"    cV: shape={cV.shape}, min={cV.min():.4f}, max={cV.max():.4f}, mean={cV.mean():.4f}, std={cV.std():.4f}")
        print(f"        Sample (top-left 3x3):\n{cV[:3, :3]}")
        print(f"    cD: shape={cD.shape}, min={cD.min():.4f}, max={cD.max():.4f}, mean={cD.mean():.4f}, std={cD.std():.4f}")
        print(f"        Sample (top-left 3x3):\n{cD[:3, :3]}")
    
    # Convert coefficients to a single array
    arr, slices = pywt.coeffs_to_array(coeffs)
    
    # Normalize for saving as image
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    arr_norm = arr_norm.astype(np.uint8)
    
    # Save the DWT result
    output_file = f"{imageid}_dwt_{channel_name}_levels{levels}_{wavelet}.png"
    
    if channel_name == "Y":
        # Y channel: save as grayscale
        out_img = Image.fromarray(arr_norm)
    elif channel_name == "U":
        # U channel: visualize with constant Y=128, varying Cb, constant Cr=128
        out_img = ycbcr_to_rgb_channel(128, arr_norm, 128)
    elif channel_name == "V":
        # V channel: visualize with constant Y=128, constant Cb=128, varying Cr
        out_img = ycbcr_to_rgb_channel(128, 128, arr_norm)
    
    out_img.save(output_file)
    print(f"Saved {channel_name} channel DWT to {output_file}")
    
    return arr_norm

# Load image and convert to YCbCr
img_rgb = Image.open(f"kodak-dataset/{IMAGEID}.png")
img_ycbcr = img_rgb.convert("YCbCr")
y_channel, u_channel, v_channel = img_ycbcr.split()

# Save original channels
y_channel.save(f"{IMAGEID}_original_Y.png")
# Save U and V with color visualization
u_colored = ycbcr_to_rgb_channel(128, np.asarray(u_channel), 128)
v_colored = ycbcr_to_rgb_channel(128, 128, np.asarray(v_channel))
u_colored.save(f"{IMAGEID}_original_U.png")
v_colored.save(f"{IMAGEID}_original_V.png")
print(f"Saved original Y, U, V channels")

# Convert to numpy arrays
Y = np.asarray(y_channel, dtype=np.float32)
U = np.asarray(u_channel, dtype=np.float32)
V = np.asarray(v_channel, dtype=np.float32)

# Process each channel
print(f"\nProcessing DWT with {LEVELS} level(s) using {WAVELET} wavelet:")
y_dwt = save_dwt_channel(Y, "Y", IMAGEID, LEVELS, WAVELET)
u_dwt = save_dwt_channel(U, "U", IMAGEID, LEVELS, WAVELET)
v_dwt = save_dwt_channel(V, "V", IMAGEID, LEVELS, WAVELET)

print(f"\nAll channels processed successfully!")
