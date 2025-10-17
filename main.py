# pip install pywavelets pillow matplotlib
import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG: hard-code number of levels
LEVELS = 1        # change this to 1,2,3...
WAVELET = "db4"   # change to 'db1','db4','sym4', etc.
OUTPUT_FILE_DWT = f"dwt_levels{LEVELS}_{WAVELET}.png"
OUTPUT_FILE_ORIG = "original_grayscale.png"
# ---------------------------

# 1) Load an image (grayscale for simplicity)
img = Image.open("kodak-dataset/kodim01.png").convert("L")
A = np.asarray(img, dtype=np.float32)

# Save the original grayscale image
img.save(OUTPUT_FILE_ORIG)
print(f"Saved original grayscale image to {OUTPUT_FILE_ORIG}")

# 2) Multi-level 2D DWT
coeffs = pywt.wavedec2(A, wavelet=WAVELET, level=LEVELS)

# 3) Convert coefficients to a single array
arr, slices = pywt.coeffs_to_array(coeffs)

# Normalize for saving as image
arr_norm = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
arr_norm = arr_norm.astype(np.uint8)

# 4) Save the DWT result
out_img = Image.fromarray(arr_norm)
out_img.save(OUTPUT_FILE_DWT)
print(f"Saved multi-level DWT result to {OUTPUT_FILE_DWT}")

# 5) Display side-by-side
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(arr_norm, cmap="gray")
plt.title(f"{LEVELS}-Level DWT ({WAVELET})")
plt.axis("off")

plt.tight_layout()
plt.show()
