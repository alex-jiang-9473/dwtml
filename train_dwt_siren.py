"""
Train SIREN models on DWT coefficients for image compression.
Saves trained models for later reconstruction.

Based on dwt_siren_split_yuv_channels.py logic but split for clarity.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
import json
import time
import numpy as np
import pywt
import torch
from PIL import Image
from training import Trainer
from siren import Siren
import util
from resource_monitor import get_memory_stats, reset_cuda_memory
from experiment_config import (
    BandTrainingConfig,
    COMPARE_CONFIGS,
    IMAGEID,
    LEVELS,
    MODEL_DIR,
    SKIP_HF_TRAINING,
    TRAIN_HF_BANDS,
    build_band_checkpoint_name,
    format_band_config,
    get_candidate_configs,
    get_filter_threshold,
    WAVELET,
)
from dwt_siren_common import (
    calculate_iterations_for_params,
    calculate_model_params,
    find_model_size_for_budget,
    make_full_coords,
    make_norm_coords,
    rgb_to_yuv,
)

def allocate_parameters_per_channel(channel_coeffs, total_budget, hf_budget, channel_name):
    """Allocate parameters across bands for a channel"""
    ll_coeffs = channel_coeffs[0]
    
    if channel_name == 'Y':
        ll_pixels = ll_coeffs.size
    else:
        threshold = get_filter_threshold(channel_name, 'LL') * np.std(ll_coeffs)
        sparse_mask = np.abs(ll_coeffs) > threshold
        ll_pixels = np.sum(sparse_mask)
    
    band_info = []
    if ll_pixels > 0:
        band_info.append(('LL', ll_coeffs, ll_pixels, 0))
    
    allocations = {}
    layers, hidden_size = find_model_size_for_budget(total_budget, strict_under=True)
    actual_params = calculate_model_params(layers, hidden_size)
    
    allocations['LL'] = {
        'layers': layers,
        'hidden_size': hidden_size,
        'params': actual_params,
        'num_coeffs': ll_pixels
    }
    
    if SKIP_HF_TRAINING:
        return allocations
    
    # HF allocation (simplified for this version)
    for level_idx, (cH, cV, cD) in enumerate(channel_coeffs[1:], start=1):
        for band_name, coeffs in [('cH', cH), ('cV', cV), ('cD', cD)]:
            band_full_name = f"{band_name}_L{level_idx}"
            threshold = get_filter_threshold(channel_name, band_full_name) * np.std(coeffs)
            sparse_mask = np.abs(coeffs) > threshold
            num_sparse = np.sum(sparse_mask)
            
            if num_sparse > 0:
                band_info.append((band_full_name, coeffs, num_sparse, level_idx))
    
    return allocations

def train_single_band_model(coeffs, coords, band_name, layers, hidden_size, device, 
                            iterations, w0=30.0, dim_out=1, lr=2e-4):
    """Train a single band model"""
    
    # Normalize coefficients
    if coeffs.ndim == 1:
        coeffs = coeffs.reshape(-1, 1)
    
    coeff_mean = np.mean(coeffs, axis=0)
    coeff_std = np.std(coeffs, axis=0)
    
    if np.any(coeff_std < 1e-6):
        coeffs_norm = coeffs - coeff_mean
        coeff_std = np.where(coeff_std < 1e-6, 1.0, coeff_std)
    else:
        coeffs_norm = (coeffs - coeff_mean) / coeff_std
    
    # Tensors
    coords_tensor = torch.FloatTensor(coords).to(device)
    coeffs_tensor = torch.FloatTensor(coeffs_norm).to(device)
    if coeffs_tensor.ndim == 1:
        coeffs_tensor = coeffs_tensor.unsqueeze(1)
    
    # Reset memory
    if device.type == 'cuda':
        reset_cuda_memory()
        torch.cuda.synchronize()
    
    # Model
    dim_in = coords.shape[1]
    model = Siren(
        dim_in=dim_in,
        dim_hidden=hidden_size,
        dim_out=dim_out,
        num_layers=layers,
        final_activation=None,
        w0_initial=w0,
        w0=w0
    ).to(device)
    
    # Train
    start_time = time.time()
    trainer = Trainer(model, lr=lr)
    trainer.train(coords_tensor, coeffs_tensor, num_iters=iterations)
    training_time = time.time() - start_time
    
    # Calculate PSNR
    model.eval()
    with torch.no_grad():
        pred_norm = model(coords_tensor).cpu().numpy()
        pred = pred_norm * (coeff_std + 1e-8) + coeff_mean
        true_flat = coeffs.flatten()
        pred_flat = pred.flatten()
        true_tensor = torch.FloatTensor(true_flat)
        pred_tensor = torch.FloatTensor(pred_flat)
        psnr = util.get_clamped_psnr(true_tensor, pred_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_mem = get_memory_stats('cuda')
    mem_peak = end_mem['peak_allocated_mb'] if end_mem else None
    
    return model, coeff_mean, coeff_std, psnr, training_time, mem_peak

def train_channel_models(channel_data, channel_coeffs, channel_name, param_budget, 
                         hf_budget, device):
    """Train all models for a channel (Y, U, or V)"""
    
    print(f"\n{'='*70}")
    print(f"Training {channel_name} Channel Models")
    print(f"  LL Band Budget: {param_budget:,}")
    print(f"  HF Bands Budget: {hf_budget:,}")
    print(f"  Total Budget: {param_budget + hf_budget:,}")
    print(f"{'='*70}")
    
    allocations = allocate_parameters_per_channel(channel_coeffs, param_budget, hf_budget, channel_name)
    
    total_allocated = sum(info['params'] for info in allocations.values())
    print(f"\nParameter Allocation for {channel_name} Channel:")
    print(f"  {'Band':<12} {'Layers':<8} {'Hidden':<8} {'Params':<10} {'Coeffs':<10}")
    print(f"  {'-'*60}")
    for band_name, info in allocations.items():
        print(f"  {band_name:<12} {info['layers']:<8} {info['hidden_size']:<8} "
              f"{info['params']:<10,} {info['num_coeffs']:<10}")
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<12} {'':<8} {'':<8} {total_allocated:<10,}")
    print()
    
    models = {}
    
    # Train LL band
    ll_coeffs = channel_coeffs[0]
    h, w = ll_coeffs.shape
    ll_config = allocations['LL']
    
    if channel_name == 'Y':
        # Y LL: all pixels
        coords = make_full_coords(h, w)
        coeffs_to_train = ll_coeffs.flatten()
        num_train = ll_coeffs.size
        sparse_mask = None
        
        iterations = calculate_iterations_for_params(ll_config['params'])
        print(f"Training LL band: {h}x{w} all pixels, {iterations} iterations")
    else:
        # U/V LL: sparse
        threshold = get_filter_threshold(channel_name, 'LL') * np.std(ll_coeffs)
        sparse_mask = np.abs(ll_coeffs) > threshold
        sparse_coords_idx = np.argwhere(sparse_mask)
        num_train = len(sparse_coords_idx)
        
        coords = make_norm_coords(sparse_coords_idx, h, w)
        
        coeffs_to_train = ll_coeffs[sparse_mask]
        iterations = calculate_iterations_for_params(ll_config['params'])
        sparsity_pct = 100.0 * num_train / (h * w)
        print(f"Training LL band: {num_train}/{h*w} coeffs ({sparsity_pct:.1f}%), {iterations} iterations")
    
    model_ll, ll_mean, ll_std, ll_psnr, train_time, mem_peak = train_single_band_model(
        coeffs_to_train, coords, 'LL',
        ll_config['layers'], ll_config['hidden_size'],
        device, iterations, w0=30.0, dim_out=1
    )
    
    models['LL'] = {
        'model': model_ll,
        'mean': ll_mean,
        'std': ll_std,
        'psnr': ll_psnr,
        'shape': (h, w),
        'sparse_mask': sparse_mask,
        'num_coeffs': num_train,
        'layers': ll_config['layers'],
        'hidden_size': ll_config['hidden_size'],
        'params': ll_config['params']
    }
    
    print(f"  ✓ LL: PSNR={ll_psnr:.2f} dB, params={ll_config['params']:,}, time={train_time:.1f}s, mem={mem_peak:.1f}MB")
    
    if SKIP_HF_TRAINING:
        print(f"\nSkipping HF bands (SKIP_HF_TRAINING=True)")
    
    return models

def save_models(models, channel_name):
    """Save trained models to disk"""
    channel_dir = os.path.join(MODEL_DIR, channel_name)
    os.makedirs(channel_dir, exist_ok=True)
    
    for band_name, model_info in models.items():
        model = model_info['model']
        
        # Save model state dict and metadata
        save_data = {
            'state_dict': model.state_dict(),
            'coeff_mean': model_info['mean'],
            'coeff_std': model_info['std'],
            'shape': model_info['shape'],
            'layers': model_info['layers'],
            'hidden_size': model_info['hidden_size'],
            'psnr': model_info['psnr'],
            'params': model_info['params'],
            'num_coeffs': model_info['num_coeffs'],
            'sparse_mask': model_info['sparse_mask']
        }
        
        model_path = os.path.join(channel_dir, f"{band_name}_model.pt")
        torch.save(save_data, model_path)
        print(f"  Saved {channel_name} {band_name}: {model_path}")


def build_band_tasks(channel_name, channel_coeffs):
    """Build dense LL and sparse HF training tasks for a channel."""
    tasks = []

    ll_coeffs = channel_coeffs[0]
    h, w = ll_coeffs.shape
    if channel_name == 'Y':
        ll_coords = make_full_coords(h, w)
        ll_values = ll_coeffs.reshape(-1)
        ll_sparse_mask = None
        ll_threshold_factor = None
    else:
        threshold_factor = get_filter_threshold(channel_name, 'LL')
        threshold = threshold_factor * np.std(ll_coeffs)
        ll_sparse_mask = np.abs(ll_coeffs) > threshold
        sparse_coords_idx = np.argwhere(ll_sparse_mask)
        ll_coords = make_norm_coords(sparse_coords_idx, h, w)
        ll_values = ll_coeffs[ll_sparse_mask]
        ll_threshold_factor = float(threshold_factor)

    tasks.append({
        'channel_name': channel_name,
        'band_name': 'LL',
        'band_id': f'{channel_name}_LL',
        'shape': (h, w),
        'coords': ll_coords,
        'values': ll_values,
        'sparse_mask': ll_sparse_mask,
        'dense': channel_name == 'Y',
        'role': 'y_ll' if channel_name == 'Y' else 'uv_ll',
        'threshold_factor': ll_threshold_factor,
    })

    if not TRAIN_HF_BANDS:
        return tasks

    for level_idx, (cH, cV, cD) in enumerate(channel_coeffs[1:], start=1):
        for band_name, coeffs in [('cH', cH), ('cV', cV), ('cD', cD)]:
            band_full_name = f'{band_name}_L{level_idx}'
            threshold_factor = get_filter_threshold(channel_name, band_full_name)
            threshold = threshold_factor * np.std(coeffs)
            sparse_mask = np.abs(coeffs) > threshold
            if not np.any(sparse_mask):
                continue

            sparse_coords_idx = np.argwhere(sparse_mask)
            coords = make_norm_coords(sparse_coords_idx, *coeffs.shape)
            values = coeffs[sparse_mask]

            tasks.append({
                'channel_name': channel_name,
                'band_name': band_full_name,
                'band_id': f'{channel_name}_{band_full_name}',
                'shape': coeffs.shape,
                'coords': coords,
                'values': values,
                'sparse_mask': sparse_mask,
                'dense': False,
                'role': 'hf',
                'threshold_factor': float(threshold_factor),
            })

    return tasks


def train_band_candidate(task, config, device, output_dir):
    """Train one band with one config and save a checkpoint."""
    values = task['values']
    coords = task['coords']
    values_array = np.asarray(values)

    model, coeff_mean, coeff_std, psnr, training_time, mem_peak = train_single_band_model(
        values_array,
        coords,
        task['band_id'],
        config.layers,
        config.hidden_size,
        device,
        config.iterations,
        w0=config.w0,
        dim_out=1,
        lr=config.lr,
    )

    checkpoint = {
        'state_dict': model.state_dict(),
    }

    checkpoint_name = build_band_checkpoint_name(task['channel_name'], task['band_name'], config)
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)

    return {
        'config': config.to_dict(),
        'config_label': format_band_config(config),
        'training_psnr': float(psnr),
        'training_time_sec': float(training_time),
        'memory_peak_mb': None if mem_peak is None else float(mem_peak),
        'checkpoint_path': checkpoint_path,
        'checkpoint_name': checkpoint_name,
        'params': calculate_model_params(config.layers, config.hidden_size, dim_in=coords.shape[1], dim_out=1),
        'num_coeffs': int(values_array.size),
        'band_metadata': {
            'channel_name': task['channel_name'],
            'band_name': task['band_name'],
            'band_id': task['band_id'],
            'shape': tuple(task['shape']),
            'sparse_mask': task['sparse_mask'],
            'coeff_mean': np.asarray(coeff_mean),
            'coeff_std': np.asarray(coeff_std),
            'dim_in': coords.shape[1],
            'dim_out': 1,
        },
    }


def write_band_comparison_csv(band_summary, output_path):
    """Write one row per candidate for a single-band comparison report."""
    fieldnames = [
        'channel_name',
        'band_name',
        'band_id',
        'role',
        'dense',
        'shape_h',
        'shape_w',
        'candidate_count',
        'best_config_label',
        'best_training_psnr',
        'candidate_rank',
        'candidate_is_best',
        'config_label',
        'layers',
        'hidden_size',
        'iterations',
        'lr',
        'w0',
        'training_psnr',
        'training_time_sec',
        'memory_peak_mb',
        'params',
        'num_coeffs',
        'checkpoint_name',
        'checkpoint_path',
    ]

    shape = band_summary.get('shape') or [None, None]
    shape_h = shape[0] if len(shape) > 0 else None
    shape_w = shape[1] if len(shape) > 1 else None
    best_label = band_summary.get('best_config_label')

    rows = []
    for rank, candidate in enumerate(band_summary.get('candidates', []), start=1):
        config = candidate.get('config', {})
        rows.append({
            'channel_name': band_summary.get('channel_name'),
            'band_name': band_summary.get('band_name'),
            'band_id': band_summary.get('band_id'),
            'role': band_summary.get('role'),
            'dense': band_summary.get('dense'),
            'shape_h': shape_h,
            'shape_w': shape_w,
            'candidate_count': band_summary.get('candidate_count'),
            'best_config_label': best_label,
            'best_training_psnr': band_summary.get('best_training_psnr'),
            'candidate_rank': rank,
            'candidate_is_best': candidate.get('config_label') == best_label,
            'config_label': candidate.get('config_label'),
            'layers': config.get('layers'),
            'hidden_size': config.get('hidden_size'),
            'iterations': config.get('iterations'),
            'lr': config.get('lr'),
            'w0': config.get('w0'),
            'training_psnr': candidate.get('training_psnr'),
            'training_time_sec': candidate.get('training_time_sec'),
            'memory_peak_mb': candidate.get('memory_peak_mb'),
            'params': candidate.get('params'),
            'num_coeffs': candidate.get('num_coeffs'),
            'checkpoint_name': candidate.get('checkpoint_name'),
            'checkpoint_path': candidate.get('checkpoint_path'),
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest_csv(manifest, output_path):
    """Write one row per candidate across all bands in the training manifest."""
    fieldnames = [
        'image_id',
        'levels',
        'wavelet',
        'compare_configs',
        'train_hf_bands',
        'band_key',
        'channel_name',
        'band_name',
        'band_id',
        'role',
        'dense',
        'shape_h',
        'shape_w',
        'candidate_count',
        'best_config_label',
        'best_training_psnr',
        'candidate_rank',
        'candidate_is_best',
        'config_label',
        'layers',
        'hidden_size',
        'iterations',
        'lr',
        'w0',
        'training_psnr',
        'training_time_sec',
        'memory_peak_mb',
        'params',
        'num_coeffs',
        'checkpoint_name',
        'checkpoint_path',
    ]

    rows = []
    for band_key, band_summary in (manifest.get('bands') or {}).items():
        shape = band_summary.get('shape') or [None, None]
        shape_h = shape[0] if len(shape) > 0 else None
        shape_w = shape[1] if len(shape) > 1 else None
        best_label = band_summary.get('best_config_label')

        for rank, candidate in enumerate(band_summary.get('candidates', []), start=1):
            config = candidate.get('config', {})
            rows.append({
                'image_id': manifest.get('image_id'),
                'levels': manifest.get('levels'),
                'wavelet': manifest.get('wavelet'),
                'compare_configs': manifest.get('compare_configs'),
                'train_hf_bands': manifest.get('train_hf_bands'),
                'band_key': band_key,
                'channel_name': band_summary.get('channel_name'),
                'band_name': band_summary.get('band_name'),
                'band_id': band_summary.get('band_id'),
                'role': band_summary.get('role'),
                'dense': band_summary.get('dense'),
                'shape_h': shape_h,
                'shape_w': shape_w,
                'candidate_count': band_summary.get('candidate_count'),
                'best_config_label': best_label,
                'best_training_psnr': band_summary.get('best_training_psnr'),
                'candidate_rank': rank,
                'candidate_is_best': candidate.get('config_label') == best_label,
                'config_label': candidate.get('config_label'),
                'layers': config.get('layers'),
                'hidden_size': config.get('hidden_size'),
                'iterations': config.get('iterations'),
                'lr': config.get('lr'),
                'w0': config.get('w0'),
                'training_psnr': candidate.get('training_psnr'),
                'training_time_sec': candidate.get('training_time_sec'),
                'memory_peak_mb': candidate.get('memory_peak_mb'),
                'params': candidate.get('params'),
                'num_coeffs': candidate.get('num_coeffs'),
                'checkpoint_name': candidate.get('checkpoint_name'),
                'checkpoint_path': candidate.get('checkpoint_path'),
            })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_band_experiments(task, device):
    """Train a band using one or more configs and return the best checkpoint metadata."""
    band_dir = os.path.join(MODEL_DIR, task['channel_name'], task['band_name'])
    os.makedirs(band_dir, exist_ok=True)

    candidate_configs = get_candidate_configs(task['channel_name'], task['band_name'])
    if not COMPARE_CONFIGS:
        candidate_configs = candidate_configs[:1]

    print(f"\n{'='*70}")
    print(f"Training {task['band_id']}")
    print(f"  Samples: {len(task['values'])} | Role: {task['role']} | Dense: {task['dense']}")
    print(f"  Candidate configs: {', '.join(format_band_config(cfg) for cfg in candidate_configs)}")
    print(f"{'='*70}")

    candidate_results = []
    band_metadata = None
    for config in candidate_configs:
        print(f"\n→ Config {format_band_config(config)} | lr={config.lr:.1e}")
        result = train_band_candidate(task, config, device, band_dir)
        if band_metadata is None:
            band_metadata = result.pop('band_metadata')
        else:
            result.pop('band_metadata', None)
        candidate_results.append(result)

    best_result = max(candidate_results, key=lambda item: item['training_psnr'])

    if band_metadata is None:
        raise ValueError(f"Band metadata missing for {task['band_id']}")
    band_metadata_path = os.path.join(band_dir, 'band_metadata.pt')
    torch.save(band_metadata, band_metadata_path)

    best_checkpoint = torch.load(best_result['checkpoint_path'], map_location='cpu', weights_only=False)
    best_checkpoint_path = os.path.join(band_dir, 'best_model.pt')
    torch.save(best_checkpoint, best_checkpoint_path)

    band_summary = {
        'channel_name': task['channel_name'],
        'band_name': task['band_name'],
        'band_id': task['band_id'],
        'role': task['role'],
        'shape': list(task['shape']),
        'dense': task['dense'],
        'candidate_count': len(candidate_results),
        'candidates': candidate_results,
        'band_metadata_path': band_metadata_path,
        'best_checkpoint': best_checkpoint_path,
        'best_config': best_result['config'],
        'best_config_label': best_result['config_label'],
        'best_training_psnr': best_result['training_psnr'],
    }

    summary_path = os.path.join(band_dir, 'comparison.json')
    with open(summary_path, 'w') as f:
        json.dump(band_summary, f, indent=2)

    summary_csv_path = os.path.join(band_dir, 'comparison.csv')
    write_band_comparison_csv(band_summary, summary_csv_path)

    print(f"  Best config: {best_result['config_label']} | PSNR={best_result['training_psnr']:.2f} dB")
    print(f"  Saved best checkpoint: {best_checkpoint_path}")

    return band_summary

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    start_time = time.time()

    os.makedirs(MODEL_DIR, exist_ok=True)

    img_path = f"kodak-dataset/{IMAGEID}.png"
    img_rgb = Image.open(img_path)
    print(f"\nLoaded image: {img_path} ({img_rgb.size})")

    img_yuv = rgb_to_yuv(img_rgb)
    y_channel = img_yuv[:, :, 0]
    u_channel = img_yuv[:, :, 1]
    v_channel = img_yuv[:, :, 2]

    print(f"\nYUV Statistics:")
    print(f"  Y: mean={np.mean(y_channel):.2f}, std={np.std(y_channel):.2f}")
    print(f"  U: mean={np.mean(u_channel):.2f}, std={np.std(u_channel):.2f}")
    print(f"  V: mean={np.mean(v_channel):.2f}, std={np.std(v_channel):.2f}")

    print(f"\nPerforming {LEVELS}-level DWT with {WAVELET} wavelet...")
    y_coeffs = pywt.wavedec2(y_channel, WAVELET, level=LEVELS)
    u_coeffs = pywt.wavedec2(u_channel, WAVELET, level=LEVELS)
    v_coeffs = pywt.wavedec2(v_channel, WAVELET, level=LEVELS)

    channel_tasks = {
        'Y': build_band_tasks('Y', y_coeffs),
        'U': build_band_tasks('U', u_coeffs),
        'V': build_band_tasks('V', v_coeffs),
    }

    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Compare configs: {COMPARE_CONFIGS}")
    print(f"HF training: {TRAIN_HF_BANDS}")
    for channel_name, tasks in channel_tasks.items():
        print(f"  {channel_name}: {len(tasks)} bands queued")

    manifest = {
        'image_id': IMAGEID,
        'levels': LEVELS,
        'wavelet': WAVELET,
        'compare_configs': COMPARE_CONFIGS,
        'train_hf_bands': TRAIN_HF_BANDS,
        'bands': {},
        'channels': {},
    }

    for channel_name, tasks in channel_tasks.items():
        channel_dir = os.path.join(MODEL_DIR, channel_name)
        os.makedirs(channel_dir, exist_ok=True)
        manifest['channels'][channel_name] = {'band_count': len(tasks), 'bands': []}

        for task in tasks:
            band_summary = train_band_experiments(task, device)
            band_key = f"{channel_name}_{task['band_name']}"
            manifest['bands'][band_key] = band_summary
            manifest['channels'][channel_name]['bands'].append(band_key)

    manifest_path = os.path.join(MODEL_DIR, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    manifest_csv_path = os.path.join(MODEL_DIR, 'manifest.csv')
    write_manifest_csv(manifest, manifest_csv_path)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Manifest CSV saved to: {manifest_csv_path}")
    print(f"Total bands trained: {len(manifest['bands'])}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Models stored under: {MODEL_DIR}")

if __name__ == "__main__":
    main()
