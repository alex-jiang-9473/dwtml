"""
Reconstruct images from trained SIREN models on DWT coefficients.
Loads models trained by train_dwt_siren.py and reconstructs full image.

Based on dwt_siren_split_yuv_channels.py logic.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
import json
from itertools import product
import numpy as np
import pywt
import torch
from PIL import Image
import util
from experiment_config import IMAGEID, LEVELS, MODEL_DIR, OUTPUT_DIR, WAVELET
from siren import Siren
from dwt_siren_common import (
    load_siren_checkpoint,
    rgb_to_yuv,
    yuv_to_rgb,
)


# ============================================================================== 
# CONFIGURATION
# ==============================================================================

# Mode: Choose between random combinations or specified custom models
USE_CUSTOM_BAND_MODELS = True  # Set to True to use specified models, False for combinations

# Single band reconstruction mode
USE_ONLY_LL_BAND = True  # Set to True to reconstruct using only LL bands with zeros for HF
USE_Y_LL_ONLY = False  # If True: use only Y_LL (U,V = zeros). If False: use Y_LL, U_LL, V_LL
USE_HYBRID_LL_ORIGINAL_HF = True  # If True (requires USE_ONLY_LL_BAND=True): use reconstructed LL + original HF bands

# Evaluate every checkpoint combination across all sub-bands (only used if USE_CUSTOM_BAND_MODELS=False)
MAX_COMBINATIONS = 10  # Cap runtime by sampling this many combinations.
SAMPLE_RANDOM_COMBINATIONS = True
RANDOM_SAMPLE_SEED = 2
SAVE_COMBINATION_IMAGES = True

# Custom band models (optional): Specify exact model paths per band.
# Only used if USE_CUSTOM_BAND_MODELS = True
# Format: 'CHANNEL_BAND': '/path/to/model.pt'
# Leave empty dict {} if not using custom models.
CUSTOM_BAND_MODELS = {
    # Y channel
    'Y_LL': 'results/dwt_siren_models/kodim08/Y/LL/best_model.pt',
    'Y_cH_L1': 'results/dwt_siren_models/kodim08/Y/cH_L1/best_model.pt',
    'Y_cV_L1': 'results/dwt_siren_models/kodim08/Y/cV_L1/best_model.pt',
    'Y_cD_L1': 'results/dwt_siren_models/kodim08/Y/cD_L1/best_model.pt',
    'Y_cH_L2': 'results/dwt_siren_models/kodim08/Y/cH_L2/best_model.pt',
    'Y_cV_L2': 'results/dwt_siren_models/kodim08/Y/cV_L2/best_model.pt',
    'Y_cD_L2': 'results/dwt_siren_models/kodim08/Y/cD_L2/best_model.pt',
    
    # U channel
    'U_LL': 'results/dwt_siren_models/kodim08/U/LL/best_model.pt',
    'U_cH_L1': 'results/dwt_siren_models/kodim08/U/cH_L1/best_model.pt',
    'U_cV_L1': 'results/dwt_siren_models/kodim08/U/cV_L1/best_model.pt',
    'U_cD_L1': 'results/dwt_siren_models/kodim08/U/cD_L1/best_model.pt',
    'U_cH_L2': 'results/dwt_siren_models/kodim08/U/cH_L2/best_model.pt',
    'U_cV_L2': 'results/dwt_siren_models/kodim08/U/cV_L2/best_model.pt',
    'U_cD_L2': 'results/dwt_siren_models/kodim08/U/cD_L2/best_model.pt',
    
    # V channel
    'V_LL': 'results/dwt_siren_models/kodim08/V/LL/best_model.pt',
    'V_cH_L1': 'results/dwt_siren_models/kodim08/V/cH_L1/best_model.pt',
    'V_cV_L1': 'results/dwt_siren_models/kodim08/V/cV_L1/best_model.pt',
    'V_cD_L1': 'results/dwt_siren_models/kodim08/V/cD_L1/best_model.pt',
    'V_cH_L2': 'results/dwt_siren_models/kodim08/V/cH_L2/best_model.pt',
    'V_cV_L2': 'results/dwt_siren_models/kodim08/V/cV_L2/best_model.pt',
    'V_cD_L2': 'results/dwt_siren_models/kodim08/V/cD_L2/best_model.pt',
}

# ============================================================================== 
# UTILITIES
# ============================================================================== 

def reconstruct_band_from_model(model, model_data, band_shape, sparse_mask=None, device='cuda'):
    """Reconstruct a single band using trained model
    
    Args:
        model: Trained SIREN model
        model_data: Dictionary with mean, std, shape
        band_shape: (h, w) of band
        sparse_mask: Optional mask for sparse reconstruction
        device: torch device
        
    Returns:
        Reconstructed coefficient array
    """
    h, w = band_shape
    
    if sparse_mask is not None:
        # Sparse reconstruction: only reconstruct at sparse locations
        sparse_coords_idx = np.argwhere(sparse_mask)
        
        # Normalize coordinates
        coords = sparse_coords_idx.astype(np.float32)
        coords[:, 0] = (coords[:, 0] / (h - 1)) * 2 - 1 if h > 1 else 0
        coords[:, 1] = (coords[:, 1] / (w - 1)) * 2 - 1 if w > 1 else 0
        
        # Predict
        coords_tensor = torch.FloatTensor(coords).to(device)
        with torch.no_grad():
            pred_norm = model(coords_tensor).cpu().numpy()
        
        # Denormalize
        pred = pred_norm * (model_data['coeff_std'] + 1e-8) + model_data['coeff_mean']
        
        # Fill result
        coeffs = np.zeros((h, w), dtype=np.float32)
        for idx, (y, x) in enumerate(sparse_coords_idx):
            if y < h and x < w:
                coeffs[y, x] = pred[idx, 0] if pred.ndim > 1 else pred[idx]
    else:
        # Dense reconstruction: reconstruct all pixels
        y_coords = np.linspace(-1, 1, h)
        x_coords = np.linspace(-1, 1, w)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        coords = np.stack([yy.flatten(), xx.flatten()], axis=1).astype(np.float32)
        
        # Predict
        coords_tensor = torch.FloatTensor(coords).to(device)
        with torch.no_grad():
            pred_norm = model(coords_tensor).cpu().numpy()
        
        # Denormalize
        pred = pred_norm * (model_data['coeff_std'] + 1e-8) + model_data['coeff_mean']
        coeffs = pred.reshape(h, w)
    
    return coeffs

def load_training_manifest(model_dir):
    """Load the training manifest if it exists."""
    manifest_path = os.path.join(model_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        return None

    with open(manifest_path, 'r') as f:
        return json.load(f)


def resolve_checkpoint_path(checkpoint_path):
    """Resolve checkpoint path relative to current workspace if needed."""
    if checkpoint_path is None:
        return None
    if os.path.isabs(checkpoint_path):
        return checkpoint_path
    return os.path.abspath(checkpoint_path)


def calculate_checkpoint_sizes(checkpoint_path, checkpoint_dict):
    """Calculate checkpoint file size in KB and FP16 parameter size in KB.
    
    Returns:
        (file_size_kb, param_size_fp16_kb) or (None, None) if checkpoint is None
    """
    if checkpoint_path is None:
        return None, None

    file_size_kb = os.path.getsize(checkpoint_path) / 1024.0

    param_count = checkpoint_dict.get('params') if isinstance(checkpoint_dict, dict) else None
    if param_count is None:
        if isinstance(checkpoint_dict, dict) and 'state_dict' in checkpoint_dict:
            state_dict = checkpoint_dict['state_dict']
        elif isinstance(checkpoint_dict, dict):
            state_dict = checkpoint_dict
        else:
            state_dict = {}
        param_count = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))

    fp16_size_kb = (param_count * 2) / 1024.0 if param_count else None

    return file_size_kb, fp16_size_kb


def infer_siren_architecture(state_dict):
    """Infer SIREN architecture (num_layers, hidden_size) from state_dict."""
    # Find layer dimensions from the first linear layer
    if 'net.0.linear.weight' in state_dict:
        first_weight = state_dict['net.0.linear.weight']
        hidden_size = first_weight.shape[0]  # output dim of first layer
    else:
        return None, None
    
    # Count layers by finding the highest layer index
    max_layer_idx = -1
    for key in state_dict.keys():
        if key.startswith('net.') and '.linear.' in key:
            layer_idx = int(key.split('.')[1])
            max_layer_idx = max(max_layer_idx, layer_idx)
    
    # Number of layers includes input, hidden, and output
    num_layers = max_layer_idx + 1
    
    return num_layers, hidden_size


def load_band_options(channel_name, band_name, band_shape, manifest, device='cuda'):
    """Load all available candidate checkpoints for one band and precompute coefficients."""
    band_key = f'{channel_name}_{band_name}'
    
    # Check if custom model path is specified for this band
    if band_key in CUSTOM_BAND_MODELS:
        custom_model_path = resolve_checkpoint_path(CUSTOM_BAND_MODELS[band_key])
        if not os.path.exists(custom_model_path):
            raise FileNotFoundError(f"Custom model not found: {custom_model_path}")
        
        print(f"  Loading custom model for {band_key}: {custom_model_path}")
        
        # Try to load with metadata first
        band_records = manifest.get('bands', {})
        band_record = band_records.get(band_key)
        metadata_path = band_record.get('band_metadata_path') if band_record else None
        
        checkpoint_raw = torch.load(custom_model_path, map_location=device, weights_only=False)
        
        # Extract state_dict
        if isinstance(checkpoint_raw, dict) and 'state_dict' in checkpoint_raw:
            state_dict = checkpoint_raw['state_dict']
        else:
            state_dict = checkpoint_raw
        
        # Infer architecture from state_dict
        num_layers, hidden_size = infer_siren_architecture(state_dict)
        
        if num_layers is None or hidden_size is None:
            raise ValueError(f"Could not infer SIREN architecture from checkpoint: {custom_model_path}")
        
        if metadata_path and os.path.exists(resolve_checkpoint_path(metadata_path)):
            metadata_path = resolve_checkpoint_path(metadata_path)
            band_metadata = torch.load(metadata_path, map_location='cpu', weights_only=False)
            
            dim_in = int(band_metadata.get('dim_in', 2))
            dim_out = int(band_metadata.get('dim_out', 1))
            w0 = 30.0
            
            model = Siren(
                dim_in=dim_in,
                dim_hidden=int(hidden_size),
                dim_out=dim_out,
                num_layers=int(num_layers),
                final_activation=None,
                w0_initial=w0,
                w0=w0,
            ).to(device)
            
            model.load_state_dict(state_dict)
            model.eval()
            
            model_data = {
                'coeff_mean': np.asarray(band_metadata['coeff_mean']),
                'coeff_std': np.asarray(band_metadata['coeff_std']),
            }
            band_shape_to_use = tuple(band_metadata['shape'])
            sparse_mask_to_use = band_metadata.get('sparse_mask')
        else:
            # Fallback: load as legacy checkpoint
            model, legacy_checkpoint = load_siren_checkpoint(custom_model_path, device)
            coeff_mean = legacy_checkpoint.get('coeff_mean', legacy_checkpoint.get('mean'))
            coeff_std = legacy_checkpoint.get('coeff_std', legacy_checkpoint.get('std'))
            if coeff_mean is None or coeff_std is None:
                raise ValueError(f"Custom checkpoint missing normalization stats: {custom_model_path}")
            model_data = {
                'coeff_mean': np.asarray(coeff_mean),
                'coeff_std': np.asarray(coeff_std),
            }
            band_shape_to_use = tuple(legacy_checkpoint['shape'])
            sparse_mask_to_use = legacy_checkpoint.get('sparse_mask')
        
        coeffs = reconstruct_band_from_model(
            model,
            model_data,
            band_shape_to_use,
            sparse_mask_to_use,
            device,
        )
        
        file_size_kb, param_size_fp16_kb = calculate_checkpoint_sizes(custom_model_path, checkpoint_raw)
        
        return [{
            'band_key': band_key,
            'option_id': f'{band_key}_custom',
            'checkpoint_path': custom_model_path,
            'config_label': 'custom',
            'training_psnr': None,
            'is_best': False,
            'checkpoint_file_size_kb': file_size_kb,
            'param_size_fp16_kb': param_size_fp16_kb,
            'coeffs': coeffs,
        }]
    
    # Otherwise, use manifest-driven loading
    band_records = manifest.get('bands', {})
    band_record = band_records.get(band_key)

    if band_record is None:
        if band_name == 'LL':
            raise ValueError(f'LL band record missing in manifest for {channel_name}')
        return [{
            'band_key': band_key,
            'option_id': f'{band_key}_zeros',
            'checkpoint_path': None,
            'config_label': 'zeros',
            'training_psnr': None,
            'is_best': False,
            'coeffs': np.zeros(band_shape, dtype=np.float32),
        }]

    raw_candidates = band_record.get('candidates', [])
    if raw_candidates:
        option_sources = raw_candidates
    else:
        best_path = band_record.get('best_checkpoint')
        option_sources = [{
            'checkpoint_path': best_path,
            'config_label': band_record.get('best_config_label', 'best_model'),
            'training_psnr': band_record.get('best_training_psnr'),
        }]

    metadata_path = band_record.get('band_metadata_path')
    band_metadata = None
    if metadata_path:
        metadata_path = resolve_checkpoint_path(metadata_path)
        if metadata_path and os.path.exists(metadata_path):
            band_metadata = torch.load(metadata_path, map_location='cpu', weights_only=False)

    best_checkpoint_abs = resolve_checkpoint_path(band_record.get('best_checkpoint'))
    options = []

    for idx, candidate in enumerate(option_sources):
        checkpoint_path = resolve_checkpoint_path(candidate.get('checkpoint_path'))
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            continue

        checkpoint_raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if band_metadata is not None:
            config = candidate.get('config', {})
            layers = config.get('layers')
            hidden_size = config.get('hidden_size')
            if layers is None or hidden_size is None:
                raise ValueError(f"Config missing layers/hidden_size for {band_key} candidate {idx}")

            dim_in = int(band_metadata.get('dim_in', 2))
            dim_out = int(band_metadata.get('dim_out', 1))
            w0 = float(config.get('w0', 30.0))

            model = Siren(
                dim_in=dim_in,
                dim_hidden=int(hidden_size),
                dim_out=dim_out,
                num_layers=int(layers),
                final_activation=None,
                w0_initial=w0,
                w0=w0,
            ).to(device)

            if isinstance(checkpoint_raw, dict) and 'state_dict' in checkpoint_raw:
                state_dict = checkpoint_raw['state_dict']
            else:
                state_dict = checkpoint_raw
            model.load_state_dict(state_dict)
            model.eval()

            model_data = {
                'coeff_mean': np.asarray(band_metadata['coeff_mean']),
                'coeff_std': np.asarray(band_metadata['coeff_std']),
            }
            band_shape_to_use = tuple(band_metadata['shape'])
            sparse_mask_to_use = band_metadata.get('sparse_mask')
        else:
            # Backward compatibility for older full checkpoints.
            model, legacy_checkpoint = load_siren_checkpoint(checkpoint_path, device)
            coeff_mean = legacy_checkpoint.get('coeff_mean', legacy_checkpoint.get('mean'))
            coeff_std = legacy_checkpoint.get('coeff_std', legacy_checkpoint.get('std'))
            if coeff_mean is None or coeff_std is None:
                raise ValueError(f"Legacy checkpoint missing normalization stats: {checkpoint_path}")
            model_data = {
                'coeff_mean': np.asarray(coeff_mean),
                'coeff_std': np.asarray(coeff_std),
            }
            band_shape_to_use = tuple(legacy_checkpoint['shape'])
            sparse_mask_to_use = legacy_checkpoint.get('sparse_mask')
            config = candidate.get('config', {})

        if isinstance(checkpoint_raw, dict) and 'state_dict' in checkpoint_raw:
            checkpoint_for_size = checkpoint_raw
        else:
            checkpoint_for_size = checkpoint_raw

        coeffs = reconstruct_band_from_model(
            model,
            model_data,
            band_shape_to_use,
            sparse_mask_to_use,
            device,
        )

        file_size_kb, param_size_fp16_kb = calculate_checkpoint_sizes(checkpoint_path, checkpoint_for_size)

        options.append({
            'band_key': band_key,
            'option_id': f'{band_key}_{idx}',
            'checkpoint_path': checkpoint_path,
            'config_label': candidate.get('config_label', 'unknown'),
            'config': config,
            'training_psnr': candidate.get('training_psnr'),
            'is_best': checkpoint_path == best_checkpoint_abs,
            'checkpoint_file_size_kb': file_size_kb,
            'param_size_fp16_kb': param_size_fp16_kb,
            'coeffs': coeffs,
        })

    if not options:
        if band_name == 'LL':
            raise ValueError(f'No valid LL checkpoints found for {band_key}')
        options.append({
            'band_key': band_key,
            'option_id': f'{band_key}_zeros',
            'checkpoint_path': None,
            'config_label': 'zeros',
            'training_psnr': None,
            'is_best': False,
            'coeffs': np.zeros(band_shape, dtype=np.float32),
        })

    return options


def build_combination_axes(manifest, y_coeffs_orig, u_coeffs_orig, v_coeffs_orig, device='cuda'):
    """Build all per-band option axes for Cartesian product reconstruction."""
    channel_coeffs = {
        'Y': y_coeffs_orig,
        'U': u_coeffs_orig,
        'V': v_coeffs_orig,
    }

    axes = []
    for channel_name in ['Y', 'U', 'V']:
        coeffs = channel_coeffs[channel_name]
        ll_shape = coeffs[0].shape
        axes.append({
            'channel': channel_name,
            'band_name': 'LL',
            'options': load_band_options(channel_name, 'LL', ll_shape, manifest, device),
        })

        for level_idx in range(1, LEVELS + 1):
            cH_shape = coeffs[level_idx][0].shape
            for dir_name in ['cH', 'cV', 'cD']:
                band_name = f'{dir_name}_L{level_idx}'
                axes.append({
                    'channel': channel_name,
                    'band_name': band_name,
                    'options': load_band_options(channel_name, band_name, cH_shape, manifest, device),
                })

    return axes


def build_channel_coeffs_from_selection(original_coeffs, selected_options):
    """Assemble [LL, (cH,cV,cD), ...] coefficients from selected option objects."""
    coeffs_reconstructed = [selected_options['LL']['coeffs']]
    for level_idx in range(1, LEVELS + 1):
        hf = selected_options[f'cH_L{level_idx}']['coeffs']
        vf = selected_options[f'cV_L{level_idx}']['coeffs']
        df = selected_options[f'cD_L{level_idx}']['coeffs']
        coeffs_reconstructed.append((hf, vf, df))
    return coeffs_reconstructed


def sample_option_index_tuples(axis_sizes, sample_count, seed):
    """Sample unique option-index tuples without enumerating the full Cartesian product."""
    rng = np.random.default_rng(seed)
    sampled = set()
    sampled_list = []

    while len(sampled_list) < sample_count:
        idx_tuple = tuple(int(rng.integers(0, size)) for size in axis_sizes)
        if idx_tuple in sampled:
            continue
        sampled.add(idx_tuple)
        sampled_list.append(idx_tuple)

    return sampled_list


def write_combination_metrics_csv(results, output_path):
    """Write one row per evaluated combination with aggregate metrics."""
    fieldnames = [
        'combo_index',
        'image_path',
        'y_psnr',
        'u_psnr',
        'v_psnr',
        'rgb_psnr',
        'total_checkpoint_file_size_kb',
        'total_param_size_fp16_kb',
    ]

    rows = []
    for result in results:
        metrics = result.get('metrics', {})
        rows.append({
            'combo_index': result.get('combo_index'),
            'image_path': result.get('image_path'),
            'y_psnr': metrics.get('y_psnr'),
            'u_psnr': metrics.get('u_psnr'),
            'v_psnr': metrics.get('v_psnr'),
            'rgb_psnr': metrics.get('rgb_psnr'),
            'total_checkpoint_file_size_kb': result.get('total_checkpoint_file_size_kb'),
            'total_param_size_fp16_kb': result.get('total_param_size_fp16_kb'),
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_combination_selection_csv(results, output_path):
    """Write one row per selected sub-band option inside each evaluated combination."""
    fieldnames = [
        'combo_index',
        'band_key',
        'option_id',
        'config_label',
        'checkpoint_path',
        'training_psnr',
        'is_best',
        'checkpoint_file_size_kb',
        'param_size_fp16_kb',
    ]

    rows = []
    for result in results:
        combo_index = result.get('combo_index')
        for selected in result.get('selected_sub_bands', []):
            rows.append({
                'combo_index': combo_index,
                'band_key': selected.get('band_key'),
                'option_id': selected.get('option_id'),
                'config_label': selected.get('config_label'),
                'checkpoint_path': selected.get('checkpoint_path'),
                'training_psnr': selected.get('training_psnr'),
                'is_best': selected.get('is_best'),
                'checkpoint_file_size_kb': selected.get('checkpoint_file_size_kb'),
                'param_size_fp16_kb': selected.get('param_size_fp16_kb'),
            })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_best_per_band_csv(best_per_band_config, output_path):
    """Write one row per sub-band for the best-per-band (training PSNR) selection."""
    fieldnames = [
        'band_key',
        'config_label',
        'training_psnr',
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_per_band_config)


def reconstruct_channel_from_manifest(channel_name, original_coeffs, manifest, device='cuda'):
    """Reconstruct all bands for a channel from manifest-driven checkpoints."""
    coeffs_reconstructed = []
    channel_band_keys = manifest.get('channels', {}).get(channel_name, {}).get('bands', [])
    band_records = manifest.get('bands', {})

    def load_band_checkpoint(band_key):
        band_record = band_records.get(band_key)
        if not band_record:
            return None, None

        checkpoint_path = band_record.get('best_checkpoint')
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return None, None

        model, checkpoint = load_siren_checkpoint(checkpoint_path, device)
        return model, checkpoint

    # LL band
    ll_key = f'{channel_name}_LL'
    ll_model, ll_checkpoint = load_band_checkpoint(ll_key)
    if ll_model is None:
        raise ValueError(f'LL band model not found for channel {channel_name}')

    ll_coeffs = reconstruct_band_from_model(
        ll_model,
        ll_checkpoint,
        tuple(ll_checkpoint['shape']),
        ll_checkpoint.get('sparse_mask'),
        device,
    )
    coeffs_reconstructed.append(ll_coeffs)

    # HF bands in PyWavelets order: [LL, level-2 details, level-1 details]
    for level_idx in range(1, LEVELS + 1):
        original_hf = original_coeffs[level_idx]
        hf_h, hf_w = original_hf[0].shape

        hf_bands = []
        for dir_name in ['cH', 'cV', 'cD']:
            band_key = f'{channel_name}_{dir_name}_L{level_idx}'
            if band_key in channel_band_keys:
                model, checkpoint = load_band_checkpoint(band_key)
                if model is not None:
                    band_shape = tuple(checkpoint['shape'])
                    hf_band = reconstruct_band_from_model(
                        model,
                        checkpoint,
                        band_shape,
                        checkpoint.get('sparse_mask'),
                        device,
                    )
                    hf_bands.append(hf_band)
                    continue

            hf_bands.append(np.zeros((hf_h, hf_w), dtype=np.float32))

        coeffs_reconstructed.append(tuple(hf_bands))

    return coeffs_reconstructed

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate configuration
    if USE_CUSTOM_BAND_MODELS and not CUSTOM_BAND_MODELS:
        raise ValueError(
            "USE_CUSTOM_BAND_MODELS is True but CUSTOM_BAND_MODELS is empty. "
            "Either fill CUSTOM_BAND_MODELS or set USE_CUSTOM_BAND_MODELS=False"
        )
    
    print(f"\n{'='*70}")
    print(f"MODE: {'CUSTOM MODELS' if USE_CUSTOM_BAND_MODELS else 'RANDOM COMBINATIONS'}")
    print(f"{'='*70}")
    
    # Load original image for comparison
    img_path = f"kodak-dataset/{IMAGEID}.png"
    img_rgb = Image.open(img_path)
    print(f"\nLoaded original image: {img_path} ({img_rgb.size})")
    
    orig_h, orig_w = img_rgb.size[::-1]  # (H, W)
    
    # Convert to YUV
    img_yuv = rgb_to_yuv(img_rgb)
    y_channel = img_yuv[:, :, 0]
    u_channel = img_yuv[:, :, 1]
    v_channel = img_yuv[:, :, 2]
    
    # Perform DWT to get original coefficient shapes
    y_coeffs_orig = pywt.wavedec2(y_channel, WAVELET, level=LEVELS)
    u_coeffs_orig = pywt.wavedec2(u_channel, WAVELET, level=LEVELS)
    v_coeffs_orig = pywt.wavedec2(v_channel, WAVELET, level=LEVELS)
    
    print(f"\n{'='*70}")
    print("LOADING MODELS AND RECONSTRUCTING")
    print(f"{'='*70}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    manifest = load_training_manifest(MODEL_DIR)
    if manifest is None:
        raise FileNotFoundError(
            f"No manifest.json found in {MODEL_DIR}. Run train_dwt_siren.py first."
        )

    print(f"\nLoaded training manifest with {len(manifest.get('bands', {}))} trained bands")
    print(f"Compare configs: {manifest.get('compare_configs', False)}")
    print(f"HF training: {manifest.get('train_hf_bands', False)}")

    # ========================================================================
    # LL BAND ONLY RECONSTRUCTION MODE
    # ========================================================================
    if USE_ONLY_LL_BAND:
        # Determine mode description
        if USE_HYBRID_LL_ORIGINAL_HF:
            mode_desc = "Y_LL (reconstructed) + original HF" if USE_Y_LL_ONLY else "Y_LL, U_LL, V_LL (reconstructed) + original HF"
            hf_source = "original"
        else:
            mode_desc = "Y_LL ONLY (U,V = zeros)" if USE_Y_LL_ONLY else "Y_LL, U_LL, V_LL (+ zeros for HF)"
            hf_source = "zeros"
        
        print(f"\n{'='*70}")
        print(f"RECONSTRUCTING USING LL BANDS: {mode_desc}")
        print(f"HF bands: {hf_source}")
        print(f"{'='*70}")
        
        # Load Y_LL band
        y_ll_options = load_band_options('Y', 'LL', y_coeffs_orig[0].shape, manifest, device)
        y_ll_coeffs = y_ll_options[0]['coeffs']
        
        # Load or create U_LL and V_LL
        if USE_Y_LL_ONLY:
            u_ll_coeffs = np.zeros(u_coeffs_orig[0].shape, dtype=np.float32)
            v_ll_coeffs = np.zeros(v_coeffs_orig[0].shape, dtype=np.float32)
        else:
            u_ll_options = load_band_options('U', 'LL', u_coeffs_orig[0].shape, manifest, device)
            u_ll_coeffs = u_ll_options[0]['coeffs']
            
            v_ll_options = load_band_options('V', 'LL', v_coeffs_orig[0].shape, manifest, device)
            v_ll_coeffs = v_ll_options[0]['coeffs']
        
        # Build Y coefficients: LL + HF (original or zeros)
        y_coeffs_rec = [y_ll_coeffs]
        for level_idx in range(1, LEVELS + 1):
            if USE_HYBRID_LL_ORIGINAL_HF:
                # Use original HF bands
                y_coeffs_rec.append(y_coeffs_orig[level_idx])
            else:
                # Use zeros for HF
                hf_shape = y_coeffs_orig[level_idx][0].shape
                y_coeffs_rec.append((
                    np.zeros(hf_shape, dtype=np.float32),
                    np.zeros(hf_shape, dtype=np.float32),
                    np.zeros(hf_shape, dtype=np.float32),
                ))
        
        # Build U coefficients: LL + HF (original or zeros)
        u_coeffs_rec = [u_ll_coeffs]
        for level_idx in range(1, LEVELS + 1):
            if USE_HYBRID_LL_ORIGINAL_HF:
                # Use original HF bands
                u_coeffs_rec.append(u_coeffs_orig[level_idx])
            else:
                # Use zeros for HF
                hf_shape = u_coeffs_orig[level_idx][0].shape
                u_coeffs_rec.append((
                    np.zeros(hf_shape, dtype=np.float32),
                    np.zeros(hf_shape, dtype=np.float32),
                    np.zeros(hf_shape, dtype=np.float32),
                ))
        
        # Build V coefficients: LL + HF (original or zeros)
        v_coeffs_rec = [v_ll_coeffs]
        for level_idx in range(1, LEVELS + 1):
            if USE_HYBRID_LL_ORIGINAL_HF:
                # Use original HF bands
                v_coeffs_rec.append(v_coeffs_orig[level_idx])
            else:
                # Use zeros for HF
                hf_shape = v_coeffs_orig[level_idx][0].shape
                v_coeffs_rec.append((
                    np.zeros(hf_shape, dtype=np.float32),
                    np.zeros(hf_shape, dtype=np.float32),
                    np.zeros(hf_shape, dtype=np.float32),
                ))
        
        # Reconstruct image
        y_rec = pywt.waverec2(y_coeffs_rec, WAVELET)[:orig_h, :orig_w]
        u_rec = pywt.waverec2(u_coeffs_rec, WAVELET)[:orig_h, :orig_w]
        v_rec = pywt.waverec2(v_coeffs_rec, WAVELET)[:orig_h, :orig_w]
        
        yuv_rec = np.stack([y_rec, u_rec, v_rec], axis=2)
        rgb_rec = yuv_to_rgb(yuv_rec)
        
        # Save reconstructed image
        ll_only_image_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_reconstructed_ll_only.png")
        Image.fromarray(rgb_rec).save(ll_only_image_path)
        
        # Calculate metrics
        y_psnr = util.get_clamped_psnr(
            torch.FloatTensor(y_channel.flatten()),
            torch.FloatTensor(y_rec.flatten())
        )
        u_psnr = util.get_clamped_psnr(
            torch.FloatTensor(u_channel.flatten()),
            torch.FloatTensor(u_rec.flatten())
        )
        v_psnr = util.get_clamped_psnr(
            torch.FloatTensor(v_channel.flatten()),
            torch.FloatTensor(v_rec.flatten())
        )
        rgb_psnr = util.get_clamped_psnr(
            torch.FloatTensor(np.array(img_rgb).flatten()),
            torch.FloatTensor(rgb_rec.flatten())
        )
        
        # Save results
        log_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_ll_only_log.json")
        
        # Build mode key and description based on configuration
        if USE_HYBRID_LL_ORIGINAL_HF:
            mode_key = 'hybrid_ll_y_original_hf' if USE_Y_LL_ONLY else 'hybrid_ll_yuv_original_hf'
            description = 'Hybrid: Reconstructed Y_LL + original HF bands (U,V = zeros)' if USE_Y_LL_ONLY else 'Hybrid: Reconstructed Y_LL, U_LL, V_LL + original HF bands'
        else:
            mode_key = 'll_only_y' if USE_Y_LL_ONLY else 'll_only_yuv'
            description = 'LL-only: Y_LL only (U,V = zeros)' if USE_Y_LL_ONLY else 'LL-only: Y_LL, U_LL, V_LL bands with zeros for all HF bands'
        
        ll_only_result = {
            'image_id': IMAGEID,
            'levels': LEVELS,
            'wavelet': WAVELET,
            'mode': mode_key,
            'description': description,
            'image_path': ll_only_image_path,
            'metrics': {
                'y_psnr': float(y_psnr),
                'u_psnr': float(u_psnr),
                'v_psnr': float(v_psnr),
                'rgb_psnr': float(rgb_psnr),
            },
        }
        
        with open(log_path, 'w') as f:
            json.dump(ll_only_result, f, indent=2)
        
        print(f"\n{'='*70}")
        print("LL-ONLY RECONSTRUCTION COMPLETE")
        print(f"{'='*70}")
        print(f"Image saved to: {ll_only_image_path}")
        print(f"Results saved to: {log_path}")
        print(f"\nMetrics ({mode_desc}):")
        print(f"  Y PSNR: {y_psnr:.2f} dB")
        print(f"  U PSNR: {u_psnr:.2f} dB")
        print(f"  V PSNR: {v_psnr:.2f} dB")
        print(f"  RGB PSNR: {rgb_psnr:.2f} dB")
        return

    # ========================================================================
    # CUSTOM MODELS MODE
    # ========================================================================
    if USE_CUSTOM_BAND_MODELS:
        print(f"\n{'='*70}")
        print("LOADING CUSTOM SPECIFIED MODELS")
        print(f"{'='*70}")
        
        selected_by_channel = {'Y': {}, 'U': {}, 'V': {}}
        
        # Load all custom band models
        for band_key, model_path in CUSTOM_BAND_MODELS.items():
            parts = band_key.split('_', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid band key format: {band_key}. Expected 'CHANNEL_BAND'")
            channel_name, band_name = parts
            
            if channel_name not in ['Y', 'U', 'V']:
                raise ValueError(f"Invalid channel: {channel_name}. Must be Y, U, or V")
            
            # Determine band shape from original coefficients
            if band_name == 'LL':
                if channel_name == 'Y':
                    band_shape = y_coeffs_orig[0].shape
                elif channel_name == 'U':
                    band_shape = u_coeffs_orig[0].shape
                else:
                    band_shape = v_coeffs_orig[0].shape
            else:
                # HF band: extract level from band_name (e.g., 'cH_L1' -> level 1)
                level_match = band_name.split('_L')
                if len(level_match) != 2:
                    raise ValueError(f"Invalid HF band format: {band_name}. Expected 'cX_LN'")
                try:
                    level_idx = int(level_match[1])
                except ValueError:
                    raise ValueError(f"Invalid level in band: {band_name}")
                
                if channel_name == 'Y':
                    band_shape = y_coeffs_orig[level_idx][0].shape
                elif channel_name == 'U':
                    band_shape = u_coeffs_orig[level_idx][0].shape
                else:
                    band_shape = v_coeffs_orig[level_idx][0].shape
            
            # Load band options (will return single custom option)
            options = load_band_options(channel_name, band_name, band_shape, manifest, device)
            selected = options[0]
            selected_by_channel[channel_name][band_name] = selected
            print(f"  Loaded {band_key}: {model_path}")
        
        # Reconstruct from custom models
        print(f"\nReconstructing image from custom models...")
        y_coeffs_rec = build_channel_coeffs_from_selection(y_coeffs_orig, selected_by_channel['Y'])
        u_coeffs_rec = build_channel_coeffs_from_selection(u_coeffs_orig, selected_by_channel['U'])
        v_coeffs_rec = build_channel_coeffs_from_selection(v_coeffs_orig, selected_by_channel['V'])

        y_rec = pywt.waverec2(y_coeffs_rec, WAVELET)[:orig_h, :orig_w]
        u_rec = pywt.waverec2(u_coeffs_rec, WAVELET)[:orig_h, :orig_w]
        v_rec = pywt.waverec2(v_coeffs_rec, WAVELET)[:orig_h, :orig_w]

        yuv_rec = np.stack([y_rec, u_rec, v_rec], axis=2)
        rgb_rec = yuv_to_rgb(yuv_rec)

        # Save reconstructed image
        custom_image_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_reconstructed_custom.png")
        Image.fromarray(rgb_rec).save(custom_image_path)
        
        # Calculate metrics
        y_psnr = util.get_clamped_psnr(
            torch.FloatTensor(y_channel.flatten()),
            torch.FloatTensor(y_rec.flatten())
        )
        u_psnr = util.get_clamped_psnr(
            torch.FloatTensor(u_channel.flatten()),
            torch.FloatTensor(u_rec.flatten())
        )
        v_psnr = util.get_clamped_psnr(
            torch.FloatTensor(v_channel.flatten()),
            torch.FloatTensor(v_rec.flatten())
        )
        rgb_psnr = util.get_clamped_psnr(
            torch.FloatTensor(np.array(img_rgb).flatten()),
            torch.FloatTensor(rgb_rec.flatten())
        )
        
        # Build selection log
        selection_log = []
        total_checkpoint_file_size_kb = 0.0
        total_param_size_fp16_kb = 0.0
        for channel_name in ['Y', 'U', 'V']:
            for band_name, selected in selected_by_channel[channel_name].items():
                selection_log.append({
                    'band_key': selected['band_key'],
                    'option_id': selected['option_id'],
                    'config_label': selected['config_label'],
                    'checkpoint_path': selected['checkpoint_path'],
                    'training_psnr': selected.get('training_psnr'),
                    'is_best': selected['is_best'],
                    'checkpoint_file_size_kb': selected.get('checkpoint_file_size_kb'),
                    'param_size_fp16_kb': selected.get('param_size_fp16_kb'),
                })
                checkpoint_size = selected.get('checkpoint_file_size_kb')
                if checkpoint_size is not None:
                    total_checkpoint_file_size_kb += float(checkpoint_size)
                fp16_size = selected.get('param_size_fp16_kb')
                if fp16_size is not None:
                    total_param_size_fp16_kb += float(fp16_size)
        
        # Save results
        log_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_custom_models_log.json")
        custom_result = {
            'image_id': IMAGEID,
            'levels': LEVELS,
            'wavelet': WAVELET,
            'mode': 'custom_models',
            'image_path': custom_image_path,
            'metrics': {
                'y_psnr': float(y_psnr),
                'u_psnr': float(u_psnr),
                'v_psnr': float(v_psnr),
                'rgb_psnr': float(rgb_psnr),
            },
            'total_checkpoint_file_size_kb': float(total_checkpoint_file_size_kb),
            'total_param_size_fp16_kb': float(total_param_size_fp16_kb),
            'selected_sub_bands': selection_log,
        }
        
        with open(log_path, 'w') as f:
            json.dump(custom_result, f, indent=2)
        
        print(f"\n{'='*70}")
        print("CUSTOM MODELS RECONSTRUCTION COMPLETE")
        print(f"{'='*70}")
        print(f"Image saved to: {custom_image_path}")
        print(f"Results saved to: {log_path}")
        print(f"\nMetrics:")
        print(f"  Y PSNR: {y_psnr:.2f} dB")
        print(f"  U PSNR: {u_psnr:.2f} dB")
        print(f"  V PSNR: {v_psnr:.2f} dB")
        print(f"  RGB PSNR: {rgb_psnr:.2f} dB")
        print(f"  Total file size: {total_checkpoint_file_size_kb:.1f} KB")
        print(f"  Total FP16 param size: {total_param_size_fp16_kb:.1f} KB")
        return

    # ========================================================================
    # COMBINATION SAMPLING MODE
    # ========================================================================
    print(f"\n{'='*70}")
    print("LOADING SUB-BAND OPTIONS")
    print(f"{'='*70}")
    axes = build_combination_axes(manifest, y_coeffs_orig, u_coeffs_orig, v_coeffs_orig, device)

    total_combinations = 1
    for axis in axes:
        option_count = len(axis['options'])
        total_combinations *= option_count
        print(f"{axis['channel']}_{axis['band_name']}: {option_count} options")

    if MAX_COMBINATIONS is None:
        combinations_to_run = total_combinations
    else:
        combinations_to_run = min(total_combinations, MAX_COMBINATIONS)

    print(f"\nTotal combinations: {total_combinations:,}")
    if MAX_COMBINATIONS is not None and total_combinations > MAX_COMBINATIONS:
        print(f"Running capped subset: {combinations_to_run:,}")

    combinations_dir = os.path.join(OUTPUT_DIR, f"{IMAGEID}_combo_images")
    if SAVE_COMBINATION_IMAGES:
        os.makedirs(combinations_dir, exist_ok=True)

    all_results = []
    best_result = None
    worst_result = None
    best_rgb_rec = None
    worst_rgb_rec = None

    axis_sizes = [len(axis['options']) for axis in axes]
    if (
        SAMPLE_RANDOM_COMBINATIONS
        and MAX_COMBINATIONS is not None
        and total_combinations > combinations_to_run
    ):
        option_index_sequences = sample_option_index_tuples(
            axis_sizes,
            combinations_to_run,
            RANDOM_SAMPLE_SEED,
        )
        print(f"Sampling mode: random ({combinations_to_run:,} combinations, seed={RANDOM_SAMPLE_SEED})")
    else:
        index_ranges = [range(size) for size in axis_sizes]
        option_index_sequences = product(*index_ranges)
        if MAX_COMBINATIONS is None:
            print("Sampling mode: exhaustive")
        else:
            print(f"Sampling mode: first {combinations_to_run:,} combinations")

    for combo_idx, option_indices in enumerate(option_index_sequences):
        if MAX_COMBINATIONS is not None and combo_idx >= combinations_to_run:
            break

        selected_by_channel = {
            'Y': {},
            'U': {},
            'V': {},
        }
        selection_log = []
        total_checkpoint_file_size_kb = 0.0
        total_param_size_fp16_kb = 0.0

        for axis, opt_idx in zip(axes, option_indices):
            selected = axis['options'][opt_idx]
            selected_by_channel[axis['channel']][axis['band_name']] = selected
            selection_log.append({
                'band_key': selected['band_key'],
                'option_id': selected['option_id'],
                'config_label': selected['config_label'],
                'checkpoint_path': selected['checkpoint_path'],
                'training_psnr': selected['training_psnr'],
                'is_best': selected['is_best'],
                'checkpoint_file_size_kb': selected.get('checkpoint_file_size_kb'),
                'param_size_fp16_kb': selected.get('param_size_fp16_kb'),
            })

            checkpoint_size = selected.get('checkpoint_file_size_kb')
            if checkpoint_size is not None:
                total_checkpoint_file_size_kb += float(checkpoint_size)

            fp16_size = selected.get('param_size_fp16_kb')
            if fp16_size is not None:
                total_param_size_fp16_kb += float(fp16_size)

        y_coeffs_rec = build_channel_coeffs_from_selection(y_coeffs_orig, selected_by_channel['Y'])
        u_coeffs_rec = build_channel_coeffs_from_selection(u_coeffs_orig, selected_by_channel['U'])
        v_coeffs_rec = build_channel_coeffs_from_selection(v_coeffs_orig, selected_by_channel['V'])

        y_rec = pywt.waverec2(y_coeffs_rec, WAVELET)[:orig_h, :orig_w]
        u_rec = pywt.waverec2(u_coeffs_rec, WAVELET)[:orig_h, :orig_w]
        v_rec = pywt.waverec2(v_coeffs_rec, WAVELET)[:orig_h, :orig_w]

        yuv_rec = np.stack([y_rec, u_rec, v_rec], axis=2)
        rgb_rec = yuv_to_rgb(yuv_rec)

        image_path = None
        if SAVE_COMBINATION_IMAGES:
            image_path = os.path.join(combinations_dir, f"{IMAGEID}_combo_{combo_idx:06d}.png")
            Image.fromarray(rgb_rec).save(image_path)

        y_psnr = util.get_clamped_psnr(
            torch.FloatTensor(y_channel.flatten()),
            torch.FloatTensor(y_rec.flatten())
        )
        u_psnr = util.get_clamped_psnr(
            torch.FloatTensor(u_channel.flatten()),
            torch.FloatTensor(u_rec.flatten())
        )
        v_psnr = util.get_clamped_psnr(
            torch.FloatTensor(v_channel.flatten()),
            torch.FloatTensor(v_rec.flatten())
        )
        rgb_psnr = util.get_clamped_psnr(
            torch.FloatTensor(np.array(img_rgb).flatten()),
            torch.FloatTensor(rgb_rec.flatten())
        )

        result = {
            'combo_index': combo_idx,
            'image_path': image_path,
            'metrics': {
                'y_psnr': float(y_psnr),
                'u_psnr': float(u_psnr),
                'v_psnr': float(v_psnr),
                'rgb_psnr': float(rgb_psnr),
            },
            'total_checkpoint_file_size_kb': float(total_checkpoint_file_size_kb),
            'total_param_size_fp16_kb': float(total_param_size_fp16_kb),
            'selected_sub_bands': selection_log,
        }
        all_results.append(result)

        if best_result is None or result['metrics']['rgb_psnr'] > best_result['metrics']['rgb_psnr']:
            best_result = result
            best_rgb_rec = rgb_rec.copy()

        if worst_result is None or result['metrics']['rgb_psnr'] < worst_result['metrics']['rgb_psnr']:
            worst_result = result
            worst_rgb_rec = rgb_rec.copy()

        if (combo_idx + 1) % 25 == 0 or combo_idx + 1 == combinations_to_run:
            print(f"Processed {combo_idx + 1:,}/{combinations_to_run:,} combinations")

    print(f"\nBuilding best-per-band reconstruction from training PSNR...")
    best_per_band_result = None
    best_per_band_rgb = None
    best_per_band_config = []

    selected_best_per_band = {'Y': {}, 'U': {}, 'V': {}}
    for axis in axes:
        if not axis['options']:
            continue
        best_option = max(
            axis['options'],
            key=lambda opt: opt.get('training_psnr') or -float('inf')
        )
        selected_best_per_band[axis['channel']][axis['band_name']] = best_option
        best_per_band_config.append({
            'band_key': best_option['band_key'],
            'config_label': best_option['config_label'],
            'training_psnr': best_option.get('training_psnr'),
        })

    y_coeffs_best = build_channel_coeffs_from_selection(y_coeffs_orig, selected_best_per_band['Y'])
    u_coeffs_best = build_channel_coeffs_from_selection(u_coeffs_orig, selected_best_per_band['U'])
    v_coeffs_best = build_channel_coeffs_from_selection(v_coeffs_orig, selected_best_per_band['V'])

    y_rec_best = pywt.waverec2(y_coeffs_best, WAVELET)[:orig_h, :orig_w]
    u_rec_best = pywt.waverec2(u_coeffs_best, WAVELET)[:orig_h, :orig_w]
    v_rec_best = pywt.waverec2(v_coeffs_best, WAVELET)[:orig_h, :orig_w]

    yuv_rec_best = np.stack([y_rec_best, u_rec_best, v_rec_best], axis=2)
    rgb_rec_best = yuv_to_rgb(yuv_rec_best)
    best_per_band_rgb = rgb_rec_best.copy()

    y_psnr_best = util.get_clamped_psnr(
        torch.FloatTensor(y_channel.flatten()),
        torch.FloatTensor(y_rec_best.flatten())
    )
    u_psnr_best = util.get_clamped_psnr(
        torch.FloatTensor(u_channel.flatten()),
        torch.FloatTensor(u_rec_best.flatten())
    )
    v_psnr_best = util.get_clamped_psnr(
        torch.FloatTensor(v_channel.flatten()),
        torch.FloatTensor(v_rec_best.flatten())
    )
    rgb_psnr_best = util.get_clamped_psnr(
        torch.FloatTensor(np.array(img_rgb).flatten()),
        torch.FloatTensor(rgb_rec_best.flatten())
    )

    best_per_band_result = {
        'reconstruction_type': 'best_per_band',
        'metrics': {
            'y_psnr': float(y_psnr_best),
            'u_psnr': float(u_psnr_best),
            'v_psnr': float(v_psnr_best),
            'rgb_psnr': float(rgb_psnr_best),
        },
        'selected_sub_bands': best_per_band_config,
    }

    log_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_combination_log.json")
    best_image_path = None
    worst_image_path = None
    best_per_band_path = None

    if best_result is not None and best_rgb_rec is not None:
        best_image_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_best_combination.png")
        Image.fromarray(best_rgb_rec).save(best_image_path)

    if worst_result is not None and worst_rgb_rec is not None:
        worst_image_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_worst_combination.png")
        Image.fromarray(worst_rgb_rec).save(worst_image_path)

    if best_per_band_rgb is not None:
        best_per_band_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_reconstructed_best.png")
        Image.fromarray(best_per_band_rgb).save(best_per_band_path)

    summary = {
        'image_id': IMAGEID,
        'levels': LEVELS,
        'wavelet': WAVELET,
        'total_combinations_available': total_combinations,
        'total_combinations_evaluated': len(all_results),
        'max_combinations_cap': MAX_COMBINATIONS,
        'sample_random_combinations': SAMPLE_RANDOM_COMBINATIONS,
        'random_sample_seed': RANDOM_SAMPLE_SEED,
        'save_combination_images': SAVE_COMBINATION_IMAGES,
        'best_combination': {
            'image_path': best_image_path,
            'result': best_result,
        },
        'worst_combination': {
            'image_path': worst_image_path,
            'result': worst_result,
        },
        'best_per_band_training': {
            'image_path': best_per_band_path,
            'result': best_per_band_result,
        },
        'results': all_results,
    }
    with open(log_path, 'w') as f:
        json.dump(summary, f, indent=2)

    combo_metrics_csv_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_combination_metrics.csv")
    combo_selection_csv_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_combination_selections.csv")
    best_per_band_csv_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_best_per_band_training.csv")

    write_combination_metrics_csv(all_results, combo_metrics_csv_path)
    write_combination_selection_csv(all_results, combo_selection_csv_path)
    write_best_per_band_csv(best_per_band_config, best_per_band_csv_path)

    print(f"\n{'='*70}")
    print("COMBINATION EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Combination log saved to: {log_path}\n")
    print(f"Combination metrics CSV saved to: {combo_metrics_csv_path}")
    print(f"Combination selections CSV saved to: {combo_selection_csv_path}")
    print(f"Best-per-band CSV saved to: {best_per_band_csv_path}\n")

    if best_per_band_result is not None:
        print(f"BEST PER-BAND (from training PSNR):")
        print(f"  Image: {best_per_band_path}")
        print(f"  RGB PSNR: {best_per_band_result['metrics']['rgb_psnr']:.2f} dB\n")

    if best_result is not None:
        print(f"BEST COMBINATION (from {combinations_to_run:,} samples):")
        print(f"  Index: {best_result['combo_index']}")
        print(f"  Image: {best_image_path}")
        print(f"  RGB PSNR: {best_result['metrics']['rgb_psnr']:.2f} dB\n")

    if worst_result is not None:
        print(f"WORST COMBINATION (from {combinations_to_run:,} samples):")
        print(f"  Index: {worst_result['combo_index']}")
        print(f"  Image: {worst_image_path}")
        print(f"  RGB PSNR: {worst_result['metrics']['rgb_psnr']:.2f} dB")

if __name__ == "__main__":
    main()
