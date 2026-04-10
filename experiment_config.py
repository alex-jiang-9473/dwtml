"""Experiment presets and threshold configuration for the split DWT SIREN pipeline."""

from dataclasses import asdict, dataclass
from itertools import product
from typing import Dict, List


# ------------------------------------------------------------------------------
# Pipeline defaults used by both training and reconstruction
# ------------------------------------------------------------------------------
LEVELS = 2
WAVELET = "db4"
IMAGEID = "kodim08"
MODEL_DIR = f"results/dwt_siren_models/{IMAGEID}"
OUTPUT_DIR = "results/reconstructed_images"

TRAIN_HF_BANDS = True
COMPARE_CONFIGS = True
SKIP_HF_TRAINING = False


@dataclass(frozen=True)
class BandTrainingConfig:
    """Single fully-expanded training config for one band."""

    layers: int
    hidden_size: int
    iterations: int
    lr: float = 2e-4
    w0: float = 30.0

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class NetworkConfig:
    """Network architecture config group."""

    layers: int
    hidden_size: int


@dataclass(frozen=True)
class HyperParamConfig:
    """Optimizer / SIREN hyperparameter config group."""

    lr: float = 2e-4
    w0: float = 30.0


NETWORK_CONFIG_GROUPS: Dict[str, List[NetworkConfig]] = {
    "y_ll": [
        NetworkConfig(layers=3, hidden_size=6),
        NetworkConfig(layers=3, hidden_size=12),
        NetworkConfig(layers=6, hidden_size=12),
        NetworkConfig(layers=3, hidden_size=21),
        NetworkConfig(layers=6, hidden_size=30),
        NetworkConfig(layers=6, hidden_size=42),
        NetworkConfig(layers=8, hidden_size=56),
        NetworkConfig(layers=10, hidden_size=56),
    ],
    "uv_ll": [
        NetworkConfig(layers=3, hidden_size=6),
        NetworkConfig(layers=3, hidden_size=12),
        NetworkConfig(layers=6, hidden_size=12),
        NetworkConfig(layers=3, hidden_size=21),
        NetworkConfig(layers=6, hidden_size=30),
        NetworkConfig(layers=6, hidden_size=42),
        NetworkConfig(layers=8, hidden_size=56),
        NetworkConfig(layers=10, hidden_size=56),
    ],
    "hf": [
        NetworkConfig(layers=3, hidden_size=6),
        NetworkConfig(layers=3, hidden_size=12),
        NetworkConfig(layers=6, hidden_size=12),
        NetworkConfig(layers=3, hidden_size=21),
        NetworkConfig(layers=6, hidden_size=30),
        NetworkConfig(layers=6, hidden_size=42),
        NetworkConfig(layers=8, hidden_size=56),
        NetworkConfig(layers=10, hidden_size=56),
    ],
}

ITERATION_CONFIG_GROUPS: Dict[str, List[int]] = {
    "y_ll": [100, 300, 500, 1000, 2000, 5000, 10000],
    "uv_ll": [100, 300, 500, 1000, 2000, 5000, 10000],
    "hf": [100, 300, 500, 1000, 2000, 5000, 10000],

}

HYPERPARAM_CONFIG_GROUPS: Dict[str, List[HyperParamConfig]] = {
    "y_ll": [
        HyperParamConfig(lr=2e-4, w0=30.0),
        HyperParamConfig(lr=1.5e-4, w0=30.0),
    ],
    "uv_ll": [
        HyperParamConfig(lr=2e-4, w0=30.0),
    ],
    "hf": [
        HyperParamConfig(lr=2e-4, w0=40.0),
        HyperParamConfig(lr=1.5e-4, w0=60.0),
    ],
}

ROLE_FILTER_THRESHOLDS: Dict[str, float] = {
    "uv_ll": 1.0,
    "hf": 2.0,
}

# Optional overrides by exact band id, e.g. "Y_cH_L1", "U_LL", "V_cD_L2"
BAND_FILTER_THRESHOLDS: Dict[str, float] = {}


def get_band_role(channel_name: str, band_name: str) -> str:
    """Map a channel/band pair to a preset role."""
    if band_name == "LL":
        return "y_ll" if channel_name == "Y" else "uv_ll"
    return "hf"


def get_candidate_configs(channel_name: str, band_name: str) -> List[BandTrainingConfig]:
    """Return band candidates from combined config groups."""
    role = get_band_role(channel_name, band_name)
    networks = NETWORK_CONFIG_GROUPS.get(role, [])
    iterations = ITERATION_CONFIG_GROUPS.get(role, [])
    hyperparams = HYPERPARAM_CONFIG_GROUPS.get(role, [])

    candidates = []
    for net_cfg, iters, hp_cfg in product(networks, iterations, hyperparams):
        candidates.append(
            BandTrainingConfig(
                layers=net_cfg.layers,
                hidden_size=net_cfg.hidden_size,
                iterations=iters,
                lr=hp_cfg.lr,
                w0=hp_cfg.w0,
            )
        )

    return candidates


def format_band_config(config: BandTrainingConfig) -> str:
    """Compact human-readable config label."""
    lr_text = f"{config.lr:.0e}".replace("+", "")
    w0_text = str(int(config.w0)) if float(config.w0).is_integer() else str(config.w0).replace(".", "p")
    return f"L{config.layers}_H{config.hidden_size}_I{config.iterations}_LR{lr_text}_W0{w0_text}"


def build_band_checkpoint_name(channel_name: str, band_name: str, config: BandTrainingConfig) -> str:
    """Build a stable filename for a trained band checkpoint."""
    return f"{channel_name}_{band_name}_{format_band_config(config)}.pt"


def get_filter_threshold(channel_name: str, band_name: str, default_threshold: float = 1.5) -> float:
    """Resolve threshold factor using band override -> role default -> provided default."""
    band_id = f"{channel_name}_{band_name}"
    if band_id in BAND_FILTER_THRESHOLDS:
        return BAND_FILTER_THRESHOLDS[band_id]

    role = get_band_role(channel_name, band_name)
    if role in ROLE_FILTER_THRESHOLDS:
        return ROLE_FILTER_THRESHOLDS[role]

    return default_threshold