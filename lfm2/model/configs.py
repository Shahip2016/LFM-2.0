"""Model configurations for all LFM2 released sizes.

Based on the LFM2 Technical Report Table 1. Available dense models:
- LFM2-350M: 16 layers, d=1024, 10 conv + 6 GQA
- LFM2-700M: 24 layers, d=1536
- LFM2-1.2B: 20 layers, d=2048
- LFM2-2.6B: 28 layers, d=2560

MoE variant:
- LFM2-8B-A1B: 8.3B total, 1.5B active, 32 experts, top-k=4
"""

from .backbone import LFM2Config


def _make_pattern(n_layers: int, attn_positions: list[int]) -> list[str]:
    """Create block pattern from attention positions.

    Args:
        n_layers: Total number of layers.
        attn_positions: Layer indices that should be GQA attention blocks.

    Returns:
        List of "conv" and "attn" strings.
    """
    pattern = ["conv"] * n_layers
    for pos in attn_positions:
        pattern[pos] = "attn"
    return pattern


# ──────────────────────────── Dense Models ────────────────────────────

LFM2_350M = LFM2Config(
    n_layers=16,
    d_model=1024,
    n_heads=16,
    n_kv_heads=4,
    head_dim=64,
    mlp_hidden_dim=2816,
    conv_kernel_size=4,
    vocab_size=65536,
    max_seq_len=32768,
    block_pattern=_make_pattern(16, [2, 5, 8, 10, 13, 15]),
    norm_eps=1e-6,
)

LFM2_700M = LFM2Config(
    n_layers=24,
    d_model=1536,
    n_heads=24,
    n_kv_heads=4,
    head_dim=64,
    mlp_hidden_dim=4096,
    conv_kernel_size=4,
    vocab_size=65536,
    max_seq_len=32768,
    block_pattern=_make_pattern(24, [3, 7, 11, 14, 18, 21, 23]),
    norm_eps=1e-6,
)

LFM2_1_2B = LFM2Config(
    n_layers=20,
    d_model=2048,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    mlp_hidden_dim=5504,
    conv_kernel_size=4,
    vocab_size=65536,
    max_seq_len=32768,
    block_pattern=_make_pattern(20, [3, 6, 9, 12, 15, 18]),
    norm_eps=1e-6,
)

LFM2_2_6B = LFM2Config(
    n_layers=28,
    d_model=2560,
    n_heads=40,
    n_kv_heads=8,
    head_dim=64,
    mlp_hidden_dim=6912,
    conv_kernel_size=4,
    vocab_size=65536,
    max_seq_len=32768,
    block_pattern=_make_pattern(28, [3, 7, 11, 14, 18, 21, 24, 27]),
    norm_eps=1e-6,
)


# ──────────────────────────── MoE Model ────────────────────────────

LFM2_8B_A1B = LFM2Config(
    n_layers=24,
    d_model=2048,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    mlp_hidden_dim=1408,  # per-expert hidden dim (smaller than dense)
    conv_kernel_size=4,
    vocab_size=65536,
    max_seq_len=32768,
    block_pattern=_make_pattern(24, [3, 7, 11, 14, 18, 21, 23]),
    norm_eps=1e-6,
)
# MoE-specific settings (used by the MoE model builder)
LFM2_8B_A1B_MOE_CONFIG = {
    "n_experts": 32,
    "top_k": 4,
    "expert_hidden_dim": 1408,
    "dense_layers": [0, 1],  # First 2 layers remain dense for stability
}


# ──────────────────────────── Config Registry ────────────────────────────

CONFIGS = {
    "lfm2-350m": LFM2_350M,
    "lfm2-700m": LFM2_700M,
    "lfm2-1.2b": LFM2_1_2B,
    "lfm2-2.6b": LFM2_2_6B,
    "lfm2-8b-a1b": LFM2_8B_A1B,
}


def get_config(name: str) -> LFM2Config:
    """Get a model configuration by name.

    Args:
        name: Model name (e.g., ``"lfm2-350m"``, ``"lfm2-2.6b"``).

    Returns:
        The corresponding ``LFM2Config``.

    Raises:
        KeyError: If the name is not recognized.
    """
    name = name.lower()
    if name not in CONFIGS:
        available = ", ".join(sorted(CONFIGS.keys()))
        raise KeyError(f"Unknown config '{name}'. Available: {available}")
    return CONFIGS[name]
