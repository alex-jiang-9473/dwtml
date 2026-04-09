"""
Parallel training version of dwt_siren_split_yuv_channels.py
Trains all 21 sub-bands in parallel to speed up training time.

Key features:
- Trains multiple bands simultaneously using ProcessPoolExecutor
- Models moved to CPU before transfer between processes (avoids CUDA IPC issues)
- PSNR calculation requires all 3 LL bands to be trained first
- Bands complete in non-sequential order based on training time

Configuration:
- NUM_PARALLEL_WORKERS: Number of concurrent training jobs (adjust for GPU memory)
- DEVICE_IDS: GPU device IDs for round-robin allocation
"""

import os
import time
import json
import numpy as np
import torch
import pywt
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import from the original file
from dwt_siren_split_yuv_channels import (
    rgb_to_yuv, yuv_to_rgb,
    train_channel_dwt_models,
    reconstruct_channel_from_models,
    IMAGEID, LOG_DIR, WAVELET, LEVELS,
    TOTAL_PARAM_BUDGET, Y_BUDGET_PERCENT, U_BUDGET_PERCENT, V_BUDGET_PERCENT,
    Y_HF_BUDGET_PERCENT, U_HF_BUDGET_PERCENT, V_HF_BUDGET_PERCENT
)
import util

# Configuration for parallel training
NUM_PARALLEL_WORKERS = 7  # Number of bands to train in parallel (adjust based on GPU memory)
DEVICE_IDS = [0]  # GPU device IDs to use (e.g., [0, 1] for 2 GPUs)


def train_single_band_worker(args):
    """Worker function to train a single band in a separate process
    
    Args:
        args: Tuple of (band_idx, channel_name, band_name, channel_data, coeffs, budget, hf_budget, device_id)
    
    Returns:
        Tuple of (band_idx, channel_name, band_name, models_dict, process_id)
    """
    band_idx, ch_name, band_name, ch_data, ch_coeffs, budget, hf_budget, device_id = args
    
    # Get process ID to identify which worker is handling this
    pid = os.getpid()
    
    # Set device for this worker
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    print(f"[PID {pid}] [Band {band_idx}/21] Training {ch_name}-{band_name} on {device}...")
    start = time.time()
    
    # Train this specific band
    try:
        models = train_channel_dwt_models(
            ch_data, ch_coeffs, ch_name, budget, hf_budget, device,
            train_only_band=band_name
        )
        elapsed = time.time() - start
        print(f"[PID {pid}] [Band {band_idx}/21] ✓ {ch_name}-{band_name} completed in {elapsed:.1f}s")
        
        # Move models to CPU to avoid CUDA IPC issues across processes
        for band_key, model_info in models.items():
            if 'model' in model_info:
                model_info['model'].cpu()
        
        return (band_idx, ch_name, band_name, models, pid)
    except Exception as e:
        print(f"[PID {pid}] [Band {band_idx}/21] ✗ {ch_name}-{band_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return (band_idx, ch_name, band_name, None, pid)


def reconstruct_and_calc_psnr(y_models, u_models, v_models, y_channel, u_channel, v_channel, 
                               y_coeffs, u_coeffs, v_coeffs, img_rgb, orig_h, orig_w, device):
    """Reconstruct image and calculate PSNR
    
    Returns None if any channel is missing its LL band (can't reconstruct yet)
    """
    # Check if all channels have at least LL band trained
    if not y_models or 'LL' not in y_models:
        return None
    if not u_models or 'LL' not in u_models:
        return None
    if not v_models or 'LL' not in v_models:
        return None
    
    try:
        # Move models to device if needed (they come from workers on CPU)
        for models in [y_models, u_models, v_models]:
            for model_info in models.values():
                if 'model' in model_info:
                    model_info['model'].to(device)
        
        # Reconstruct channels
        y_rec_coeffs = reconstruct_channel_from_models(y_models, y_coeffs, device)
        u_rec_coeffs = reconstruct_channel_from_models(u_models, u_coeffs, device)
        v_rec_coeffs = reconstruct_channel_from_models(v_models, v_coeffs, device)
        
        # Inverse DWT
        y_rec = pywt.waverec2(y_rec_coeffs, WAVELET)[:orig_h, :orig_w]
        u_rec = pywt.waverec2(u_rec_coeffs, WAVELET)[:orig_h, :orig_w]
        v_rec = pywt.waverec2(v_rec_coeffs, WAVELET)[:orig_h, :orig_w]
        
        # Convert to RGB
        yuv_rec = np.stack([y_rec, u_rec, v_rec], axis=2)
        rgb_rec = yuv_to_rgb(yuv_rec)
        
        # Calculate PSNRs
        y_psnr = util.get_clamped_psnr(torch.FloatTensor(y_channel.flatten()), torch.FloatTensor(y_rec.flatten()))
        u_psnr = util.get_clamped_psnr(torch.FloatTensor(u_channel.flatten()), torch.FloatTensor(u_rec.flatten()))
        v_psnr = util.get_clamped_psnr(torch.FloatTensor(v_channel.flatten()), torch.FloatTensor(v_rec.flatten()))
        rgb_psnr = util.get_clamped_psnr(torch.FloatTensor(np.array(img_rgb).flatten()), torch.FloatTensor(rgb_rec.flatten()))
        
        return {
            'y': float(y_psnr),
            'u': float(u_psnr),
            'v': float(v_psnr),
            'rgb': float(rgb_psnr)
        }
    except Exception as e:
        # Handle any reconstruction errors gracefully
        print(f"  [Warning] Reconstruction failed: {e}")
        return None


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Parallel workers: {NUM_PARALLEL_WORKERS}")
    
    # Create output directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Load image
    img_path = f"kodak-dataset/{IMAGEID}.png"
    img_rgb = Image.open(img_path)
    print(f"\nLoaded image: {img_path}")
    print(f"  Size: {img_rgb.size}")
    
    # Convert to YUV
    img_yuv = rgb_to_yuv(img_rgb)
    y_channel = img_yuv[:, :, 0]
    u_channel = img_yuv[:, :, 1]
    v_channel = img_yuv[:, :, 2]
    
    print(f"\nYUV Statistics:")
    print(f"  Y: mean={np.mean(y_channel):.2f}, std={np.std(y_channel):.2f}, range=[{np.min(y_channel):.2f}, {np.max(y_channel):.2f}]")
    print(f"  U: mean={np.mean(u_channel):.2f}, std={np.std(u_channel):.2f}, range=[{np.min(u_channel):.2f}, {np.max(u_channel):.2f}]")
    print(f"  V: mean={np.mean(v_channel):.2f}, std={np.std(v_channel):.2f}, range=[{np.min(v_channel):.2f}, {np.max(v_channel):.2f}]")
    
    # Perform DWT on each channel
    print(f"\nPerforming {LEVELS}-level DWT with {WAVELET} wavelet on each channel...")
    y_coeffs = pywt.wavedec2(y_channel, WAVELET, level=LEVELS)
    u_coeffs = pywt.wavedec2(u_channel, WAVELET, level=LEVELS)
    v_coeffs = pywt.wavedec2(v_channel, WAVELET, level=LEVELS)
    
    # Parameter allocation
    y_hf_budget = int(TOTAL_PARAM_BUDGET * Y_HF_BUDGET_PERCENT)
    u_hf_budget = int(TOTAL_PARAM_BUDGET * U_HF_BUDGET_PERCENT)
    v_hf_budget = int(TOTAL_PARAM_BUDGET * V_HF_BUDGET_PERCENT)
    
    y_budget = int(TOTAL_PARAM_BUDGET * Y_BUDGET_PERCENT) - y_hf_budget
    u_budget = int(TOTAL_PARAM_BUDGET * U_BUDGET_PERCENT) - u_hf_budget
    v_budget = int(TOTAL_PARAM_BUDGET * V_BUDGET_PERCENT) - v_hf_budget
    
    print(f"\nParameter Budget Allocation:")
    print(f"  Total Budget: {TOTAL_PARAM_BUDGET:,} parameters")
    print(f"  Y: LL {y_budget:,} + HF {y_hf_budget:,}")
    print(f"  U: LL {u_budget:,} + HF {u_hf_budget:,}")
    print(f"  V: LL {v_budget:,} + HF {v_hf_budget:,}")
    
    orig_h, orig_w = y_channel.shape
    
    # Define all 21 bands
    all_bands = [
        # LL bands (3 bands)
        ('Y', y_channel, y_coeffs, y_budget, y_hf_budget, 'LL'),
        ('U', u_channel, u_coeffs, u_budget, u_hf_budget, 'LL'),
        ('V', v_channel, v_coeffs, v_budget, v_hf_budget, 'LL'),
    ]
    
    # Level 1 HF bands (9 bands)
    for band_type in ['cH_L1', 'cV_L1', 'cD_L1']:
        all_bands.append(('Y', y_channel, y_coeffs, y_budget, y_hf_budget, band_type))
        all_bands.append(('U', u_channel, u_coeffs, u_budget, u_hf_budget, band_type))
        all_bands.append(('V', v_channel, v_coeffs, v_budget, v_hf_budget, band_type))
    
    # Level 2 HF bands (9 bands)
    for band_type in ['cH_L2', 'cV_L2', 'cD_L2']:
        all_bands.append(('Y', y_channel, y_coeffs, y_budget, y_hf_budget, band_type))
        all_bands.append(('U', u_channel, u_coeffs, u_budget, u_hf_budget, band_type))
        all_bands.append(('V', v_channel, v_coeffs, v_budget, v_hf_budget, band_type))
    
    print(f"\n{'='*70}")
    print(f"Training All 21 Sub-bands in Parallel")
    print(f"  {NUM_PARALLEL_WORKERS} worker processes (up to {NUM_PARALLEL_WORKERS} bands at once)")
    print(f"  Bands will complete in non-sequential order")
    print(f"{'='*70}")
    
    # Prepare worker arguments
    worker_args = []
    for idx, (ch_name, ch_data, ch_coeffs, budget, hf_budget, band_name) in enumerate(all_bands, 1):
        device_id = DEVICE_IDS[idx % len(DEVICE_IDS)]  # Round-robin device assignment
        worker_args.append((idx, ch_name, band_name, ch_data, ch_coeffs, budget, hf_budget, device_id))
    
    # Initialize model dictionaries
    y_models = {}
    u_models = {}
    v_models = {}
    band_psnrs = []
    
    # Track process IDs to assign friendly worker numbers and bands handled
    pid_to_worker = {}
    worker_to_bands = {}  # worker_num -> list of (band_idx, ch_name, band_name)
    
    # Train in parallel
    start_time = time.time()
    completed_count = 0
    
    with ProcessPoolExecutor(max_workers=NUM_PARALLEL_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(train_single_band_worker, args): args for args in worker_args}
        
        # Process results as they complete
        for future in as_completed(futures):
            band_idx, ch_name, band_name, models, pid = future.result()
            
            # Assign friendly worker number
            if pid not in pid_to_worker:
                pid_to_worker[pid] = len(pid_to_worker) + 1
            worker_num = pid_to_worker[pid]
            
            # Track which bands this worker handled
            if worker_num not in worker_to_bands:
                worker_to_bands[worker_num] = []
            worker_to_bands[worker_num].append((band_idx, ch_name, band_name))
            
            if models is not None:
                # Update appropriate model dict
                if ch_name == 'Y':
                    y_models.update(models)
                elif ch_name == 'U':
                    u_models.update(models)
                else:  # V
                    v_models.update(models)
                
                completed_count += 1
                
                # Reconstruct and calculate PSNR after this band completes
                # (only works once all 3 LL bands are trained)
                psnrs = reconstruct_and_calc_psnr(
                    y_models, u_models, v_models,
                    y_channel, u_channel, v_channel,
                    y_coeffs, u_coeffs, v_coeffs,
                    img_rgb, orig_h, orig_w, device
                )
                
                if psnrs:
                    print(f"\n[Worker {worker_num}] [{completed_count}/21] After {ch_name}-{band_name}:")
                    print(f"  PSNR: Y={psnrs['y']:.2f}, U={psnrs['u']:.2f}, V={psnrs['v']:.2f}, RGB={psnrs['rgb']:.2f} dB")
                    
                    band_psnrs.append({
                        'band': f"{ch_name}-{band_name}",
                        'y': psnrs['y'],
                        'u': psnrs['u'],
                        'v': psnrs['v'],
                        'rgb': psnrs['rgb']
                    })
                else:
                    print(f"\n[Worker {worker_num}] [{completed_count}/21] {ch_name}-{band_name} trained (PSNR pending - need all LL bands)")
    
    train_time = time.time() - start_time
    
    # Print worker utilization summary
    print(f"\n{'='*70}")
    print(f"Worker Utilization Summary:")
    print(f"  Total Workers Used: {len(worker_to_bands)} / {NUM_PARALLEL_WORKERS}")
    
    # Map PID to worker number for display
    pid_to_worker_for_display = {pid: worker for pid, worker in pid_to_worker.items()}
    
    for worker_num in sorted(worker_to_bands.keys()):
        bands = worker_to_bands[worker_num]
        # Find the PID for this worker
        pid = [pid for pid, wn in pid_to_worker.items() if wn == worker_num][0]
        
        # Sort bands by band_idx
        bands_sorted = sorted(bands, key=lambda x: x[0])
        band_list = [f"{ch}-{bn}" for _, ch, bn in bands_sorted]
        
        print(f"  Worker {worker_num} (PID {pid}): {len(bands)} bands")
        print(f"    Bands: {', '.join(band_list)}")
    print(f"{'='*70}")
    
    # Calculate actual total parameters
    actual_params_y = sum(info['params'] for info in y_models.values())
    actual_params_u = sum(info['params'] for info in u_models.values())
    actual_params_v = sum(info['params'] for info in v_models.values())
    total_actual_params = actual_params_y + actual_params_u + actual_params_v
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"  Total Time: {train_time:.2f} seconds")
    print(f"  Actual Total Parameters: {total_actual_params:,}")
    print(f"    Y: {actual_params_y:,}")
    print(f"    U: {actual_params_u:,}")
    print(f"    V: {actual_params_v:,}")
    print(f"  Budget Efficiency: {100*total_actual_params/TOTAL_PARAM_BUDGET:.1f}%")
    print(f"{'='*70}")
    
    # Save models
    print("\nSaving models...")
    models_dir = os.path.join(LOG_DIR, "models_parallel")
    os.makedirs(models_dir, exist_ok=True)
    
    total_model_size_bytes = 0
    
    for ch_name, models in [('y', y_models), ('u', u_models), ('v', v_models)]:
        for band_name, info in models.items():
            model_path = os.path.join(models_dir, f"{ch_name}_{band_name.lower()}_model.pt")
            torch.save(info['model'].state_dict(), model_path)
            model_size = os.path.getsize(model_path)
            total_model_size_bytes += model_size
    
    total_model_size_kb = total_model_size_bytes / 1024
    total_model_size_mb = total_model_size_kb / 1024
    
    print(f"  Total Model Size: {total_model_size_bytes:,} bytes ({total_model_size_kb:.2f} KB, {total_model_size_mb:.2f} MB)")
    print(f"  Saved to: {models_dir}")
    
    # Final reconstruction and PSNR
    final_psnr = band_psnrs[-1] if band_psnrs else {'y': 0, 'u': 0, 'v': 0, 'rgb': 0}
    
    print(f"\n{'='*70}")
    print(f"Final PSNR: Y={final_psnr['y']:.2f}, U={final_psnr['u']:.2f}, V={final_psnr['v']:.2f}, RGB={final_psnr['rgb']:.2f} dB")
    print(f"{'='*70}")
    
    # Save results
    results = {
        'image_id': IMAGEID,
        'total_params': total_actual_params,
        'params_y': actual_params_y,
        'params_u': actual_params_u,
        'params_v': actual_params_v,
        'param_budget': TOTAL_PARAM_BUDGET,
        'y_psnr': final_psnr['y'],
        'u_psnr': final_psnr['u'],
        'v_psnr': final_psnr['v'],
        'rgb_psnr': final_psnr['rgb'],
        'band_psnrs': band_psnrs,
        'train_time': train_time,
        'parallel_workers': NUM_PARALLEL_WORKERS,
        'levels': LEVELS,
        'wavelet': WAVELET,
        'total_model_size_bytes': total_model_size_bytes,
        'total_model_size_kb': round(total_model_size_kb, 2),
        'total_model_size_mb': round(total_model_size_mb, 2),
        'y_models': {k: {'layers': v['layers'], 'hidden_size': v['hidden_size'], 
                         'params': v['params'], 'psnr': float(v['psnr'])} 
                     for k, v in y_models.items()},
        'u_models': {k: {'layers': v['layers'], 'hidden_size': v['hidden_size'], 
                         'params': v['params'], 'psnr': float(v['psnr'])} 
                     for k, v in u_models.items()},
        'v_models': {k: {'layers': v['layers'], 'hidden_size': v['hidden_size'], 
                         'params': v['params'], 'psnr': float(v['psnr'])} 
                     for k, v in v_models.items()}
    }
    
    results_path = os.path.join(LOG_DIR, f"{IMAGEID}_results_parallel.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    main()
