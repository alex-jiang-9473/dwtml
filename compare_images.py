import numpy as np
from PIL import Image
import argparse
import sys


def calculate_psnr(img1_path, img2_path):
    """
    Calculate PSNR between two images.
    
    Args:
        img1_path: Path to first image (reference)
        img2_path: Path to second image (comparison)
    
    Returns:
        PSNR value in dB
    """
    # Load images
    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')
    
    # Convert to numpy arrays
    arr1 = np.asarray(img1, dtype=np.float32)
    arr2 = np.asarray(img2, dtype=np.float32)
    
    # Check if images have the same dimensions
    if arr1.shape != arr2.shape:
        print(f"Warning: Images have different dimensions!")
        print(f"Image 1: {arr1.shape}")
        print(f"Image 2: {arr2.shape}")
        # Resize second image to match first
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        arr2 = np.asarray(img2, dtype=np.float32)
        print(f"Resized image 2 to match image 1: {arr2.shape}")
    
    # Calculate MSE
    mse = np.mean((arr1 - arr2) ** 2)
    
    if mse == 0:
        return float('inf')  # Images are identical
    
    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def calculate_metrics(img1_path, img2_path):
    """
    Calculate various comparison metrics between two images.
    
    Args:
        img1_path: Path to first image (reference)
        img2_path: Path to second image (comparison)
    """
    # Load images
    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')
    
    # Convert to numpy arrays
    arr1 = np.asarray(img1, dtype=np.float32)
    arr2 = np.asarray(img2, dtype=np.float32)
    
    # Resize if needed
    if arr1.shape != arr2.shape:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        arr2 = np.asarray(img2, dtype=np.float32)
    
    # Calculate metrics
    mse = np.mean((arr1 - arr2) ** 2)
    mae = np.mean(np.abs(arr1 - arr2))
    max_diff = np.max(np.abs(arr1 - arr2))
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Print results
    print("=" * 60)
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    print("=" * 60)
    print(f"Image dimensions: {arr1.shape}")
    print(f"PSNR:          {psnr:.4f} dB")
    print(f"MSE:           {mse:.4f}")
    print(f"MAE:           {mae:.4f}")
    print(f"Max Diff:      {max_diff:.4f}")
    print(f"RMSE:          {np.sqrt(mse):.4f}")
    print("=" * 60)
    
    return psnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate PSNR and other metrics between two images')
    parser.add_argument('image1', type=str, help='Path to first image (reference)')
    parser.add_argument('image2', type=str, help='Path to second image (comparison)')
    parser.add_argument('--simple', action='store_true', help='Only show PSNR value')
    
    args = parser.parse_args()
    
    try:
        if args.simple:
            psnr = calculate_psnr(args.image1, args.image2)
            print(f"{psnr:.4f}")
        else:
            calculate_metrics(args.image1, args.image2)
    except FileNotFoundError as e:
        print(f"Error: Could not find image file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
