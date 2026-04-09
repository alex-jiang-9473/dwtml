"""
Common test configuration shared between benchmark_architectures.py and reconstruct_images.py
Set TEST_MODE = True for quick validation with minimal configs, False for full operations.
"""

# Master test mode flag - change this to switch between testing and full operation
TEST_MODE = True

# Per-band best configurations from benchmark results - for testing reconstruction quality
TEST_BAND_CONFIGS = {
    'U_LL': {'arch_config': '3L_24H', 'hyperparam_strategy': 'default', 'iterations': 5000},
    'U_cD_L1': {'arch_config': '3L_24H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'U_cD_L2': {'arch_config': '4L_18H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'U_cH_L1': {'arch_config': '2L_36H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'U_cH_L2': {'arch_config': '3L_24H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'U_cV_L1': {'arch_config': '3L_24H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'U_cV_L2': {'arch_config': '3L_24H', 'hyperparam_strategy': 'aggressive', 'iterations': 1000},
    'V_LL': {'arch_config': '3L_24H', 'hyperparam_strategy': 'conservative', 'iterations': 5000},
    'V_cD_L1': {'arch_config': '3L_24H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'V_cD_L2': {'arch_config': '4L_18H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'V_cH_L1': {'arch_config': '2L_36H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'V_cH_L2': {'arch_config': '3L_24H', 'hyperparam_strategy': 'aggressive', 'iterations': 1000},
    'V_cV_L1': {'arch_config': '3L_24H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'V_cV_L2': {'arch_config': '3L_24H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'Y_LL': {'arch_config': '3L_18H', 'hyperparam_strategy': 'conservative', 'iterations': 5000},
    'Y_cD_L1': {'arch_config': '3L_24H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'Y_cD_L2': {'arch_config': '4L_18H', 'hyperparam_strategy': 'aggressive', 'iterations': 1000},
    'Y_cH_L1': {'arch_config': '2L_36H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'Y_cH_L2': {'arch_config': '3L_24H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'Y_cV_L1': {'arch_config': '2L_36H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
    'Y_cV_L2': {'arch_config': '3L_24H', 'hyperparam_strategy': 'aggressive', 'iterations': 5000},
}

# Test mode architecture configs for benchmark (derived from TEST_BAND_CONFIGS unique architectures)
# Extract unique architectures: (3, 24), (4, 18), (2, 36), (3, 18)
TEST_LL_DENSE_CONFIGS = [(3, 24), (3, 18)]
TEST_LL_SPARSE_CONFIGS = [(3, 24), (2, 36), (4, 18), (3, 18)]
TEST_HF_SPARSE_CONFIGS = [(3, 24), (2, 36), (4, 18)]
TEST_ITERATION_CONFIGS = [5000]

# Helper function to parse arch_config strings and extract unique configs from TEST_BAND_CONFIGS
def _parse_arch_config(config_str):
    """Parse '3L_24H' -> (3, 24)"""
    parts = config_str.split('_')
    if len(parts) != 2:
        return None
    layers = int(parts[0][0])  # Extract digit before 'L'
    hidden = int(parts[1][:-1])  # Extract all digits except 'H'
    return (layers, hidden)

def _extract_unique_configs_from_band_configs():
    """Automatically derive unique (layers, hidden_size) and strategies from TEST_BAND_CONFIGS"""
    unique_ll_dense = set()
    unique_ll_sparse = set()
    unique_hf_sparse = set()
    unique_strategies = set()
    
    for band_name, config in TEST_BAND_CONFIGS.items():
        arch_tuple = _parse_arch_config(config['arch_config'])
        if arch_tuple is None:
            continue
        
        unique_strategies.add(config['hyperparam_strategy'])
        
        if 'LL' in band_name:
            if band_name == 'Y_LL':
                # Y_LL in your config is Sparse with 3L_18H
                unique_ll_sparse.add(arch_tuple)
            else:
                # U_LL and V_LL are typically dense
                unique_ll_dense.add(arch_tuple)
        else:
            # HF bands are sparse
            unique_hf_sparse.add(arch_tuple)
    
    return sorted(unique_ll_dense), sorted(unique_ll_sparse), sorted(unique_hf_sparse), sorted(unique_strategies)

# Auto-derive configs from TEST_BAND_CONFIGS
_ll_dense, _ll_sparse, _hf_sparse, _strategies = _extract_unique_configs_from_band_configs()

# Override with auto-derived values
TEST_LL_DENSE_CONFIGS = _ll_dense if _ll_dense else TEST_LL_DENSE_CONFIGS
TEST_LL_SPARSE_CONFIGS = _ll_sparse if _ll_sparse else TEST_LL_SPARSE_CONFIGS
TEST_HF_SPARSE_CONFIGS = _hf_sparse if _hf_sparse else TEST_HF_SPARSE_CONFIGS

# Test mode reconstruction config (uniform method)
TEST_ARCH_CONFIG_KEY = "3L_24H"
TEST_HYPERPARAM_STRATEGY = "default"
TEST_ITERATIONS = 5000

if TEST_MODE:
    print("⚠️  TEST MODE ENABLED - Using best-config per-band for validation")
