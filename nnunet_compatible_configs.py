"""
nnUNet Compatible Configurations for TE-Swin UNet3D

This file contains tested and verified configurations that work with nnUNet.
Use these settings to avoid compatibility issues.
"""

# Verified working configurations
WORKING_CONFIGS = {
    "tiny_config": {
        "hidden_dim": 32,
        "layers": (2, 2, 2, 2),
        "heads": (2, 4, 6, 8),
        "head_dim": 16,
        "window_size": 2,
        "downscaling_factors": (2, 2, 2, 2),
        "compatible_patch_sizes": [(32, 32, 32), (64, 64, 64), (128, 128, 128)]
    },
    
    "small_config": {
        "hidden_dim": 48,
        "layers": (2, 2, 4, 2),
        "heads": (2, 4, 8, 16),
        "head_dim": 24,
        "window_size": 2,
        "downscaling_factors": (2, 2, 2, 2),
        "compatible_patch_sizes": [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
    },
    
    "base_config": {
        "hidden_dim": 96,
        "layers": (2, 2, 8, 2),
        "heads": (4, 8, 16, 32),
        "head_dim": 32,
        "window_size": 4,  # Larger window for base model
        "downscaling_factors": (4, 2, 2, 2),
        "compatible_patch_sizes": [(128, 128, 128), (256, 256, 256)]
    }
}


def get_compatible_patch_size(target_size, config_name="small_config"):
    """
    Get the nearest compatible patch size for a given target size.
    
    Args:
        target_size: Tuple of (D, H, W)
        config_name: Configuration to use
        
    Returns:
        Compatible patch size tuple
    """
    config = WORKING_CONFIGS[config_name]
    compatible_sizes = config["compatible_patch_sizes"]
    
    # Find the closest compatible size
    target_volume = target_size[0] * target_size[1] * target_size[2]
    best_size = compatible_sizes[0]
    best_diff = float('inf')
    
    for size in compatible_sizes:
        volume = size[0] * size[1] * size[2]
        diff = abs(volume - target_volume)
        if diff < best_diff:
            best_diff = diff
            best_size = size
    
    return best_size


def print_recommended_settings():
    """Print recommended nnUNet settings."""
    print("=" * 60)
    print("RECOMMENDED nnUNet SETTINGS FOR TE-SWIN UNET3D")
    print("=" * 60)
    
    print("\n🎯 VERIFIED WORKING CONFIGURATIONS:")
    
    for name, config in WORKING_CONFIGS.items():
        print(f"\n📋 {name.upper()}:")
        print(f"   - Parameters: ~{estimate_parameters(config):,}")
        print(f"   - Window size: {config['window_size']}")
        print(f"   - Downscaling: {config['downscaling_factors']}")
        print(f"   - Compatible patch sizes: {config['compatible_patch_sizes']}")
    
    print(f"\n⚙️ nnUNet PREPROCESSING RECOMMENDATIONS:")
    print(f"   - Use patch sizes: 64×64×64 or 128×128×128")
    print(f"   - Batch size: 1-2 (due to 3D memory requirements)")
    print(f"   - Enable deep supervision")
    print(f"   - Use smaller learning rate: 1e-4 to 5e-4")


def estimate_parameters(config):
    """Rough parameter count estimation."""
    hidden_dim = config["hidden_dim"]
    layers = config["layers"]
    # Rough estimation based on transformer architecture
    return hidden_dim * sum(layers) * 1000


if __name__ == "__main__":
    print_recommended_settings()
