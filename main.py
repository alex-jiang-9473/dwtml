# pip install pywavelets pillow matplotlib
import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG: hard-code number of levels
LEVELS = 2        # change this to 1,2,3...
WAVELET = "db4"   # change to 'db1','db4','sym4', etc.
OUTPUT_FILE = f"dwt_levels{LEVELS}_{WAVELET}.png"
# ---------------------------

# 1) Load an image (grayscale for simplicity)
img = Image.open("kodak-dataset/kodim01.png").convert("L")
A = np.asarray(img, dtype=np.float32)

# 2) Multi-level 2D DWT
coeffs = pywt.wavedec2(A, wavelet=WAVELET, level=LEVELS)

# 3) Convert coefficients to a single array
arr, slices = pywt.coeffs_to_array(coeffs)

# Normalize for saving as image
arr_norm = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
arr_norm = arr_norm.astype(np.uint8)

# 4) Save the result
out_img = Image.fromarray(arr_norm)
out_img.save(OUTPUT_FILE)
print(f"Saved multi-level DWT result to {OUTPUT_FILE}")

# 5) Display as well
plt.figure(figsize=(8,8))
plt.imshow(arr_norm, cmap="gray")
plt.title(f"{LEVELS}-Level DWT ({WAVELET})")
plt.axis("off")
plt.show()
