"""
Robust dimension validator and model factory for TE-Swin UNet3D.
This ensures all models work with verified compatible dimensions.
"""
import torch
import torch.nn as nn
import math
from typing import Tuple, List, Dict, Optional


class RobustDimensionValidator:
    """
    Comprehensive dimension validator for Swin Transformer 3D models.
    """
    
    @staticmethod
    def calculate_compatible_sizes(
        min_size: int = 32, 
        max_size: int = 512,
        window_size: int = 4,
        downscaling_factors: Tuple[int, ...] = (2, 2, 2, 2)
    ) -> List[int]:
        """
        Calculate all compatible sizes within a range.
        """
        total_downscaling = math.prod(downscaling_factors)
        
        compatible_sizes = []
        
        # Size must be divisible by both window_size and total_downscaling
        lcm = math.lcm(window_size, total_downscaling)
        
        size = lcm
        while size <= max_size:
            if size >= min_size:
                compatible_sizes.append(size)
            size += lcm
            
        return compatible_sizes
    
    @staticmethod
    def validate_exact_compatibility(
        input_shape: Tuple[int, int, int],
        window_size: int,
        downscaling_factors: Tuple[int, ...]
    ) -> Tuple[bool, str, Dict]:
        """
        Perform exact compatibility validation with detailed diagnostics.
        """
        d, h, w = input_shape
        total_downscaling = math.prod(downscaling_factors)
        
        diagnostics = {
            'input_shape': input_shape,
            'window_size': window_size,
            'downscaling_factors': downscaling_factors,
            'total_downscaling': total_downscaling,
            'compatibility_checks': {}
        }
        
        # Check each dimension
        for i, (dim_name, dim_size) in enumerate(zip(['D', 'H', 'W'], [d, h, w])):
            checks = {}
            
            # Window size divisibility
            window_divisible = (dim_size % window_size == 0)
            checks['window_divisible'] = {
                'result': window_divisible,
                'calculation': f"{dim_size} % {window_size} = {dim_size % window_size}"
            }
            
            # Total downscaling divisibility
            total_divisible = (dim_size % total_downscaling == 0)
            checks['total_downscaling_divisible'] = {
                'result': total_divisible,
                'calculation': f"{dim_size} % {total_downscaling} = {dim_size % total_downscaling}"
            }
            
            # Check each downscaling stage
            current_size = dim_size
            stage_results = []
            
            for j, factor in enumerate(downscaling_factors):
                stage_divisible = (current_size % factor == 0)
                new_size = current_size // factor
                
                stage_results.append({
                    'stage': j + 1,
                    'factor': factor,
                    'input_size': current_size,
                    'divisible': stage_divisible,
                    'output_size': new_size if stage_divisible else -1
                })
                
                if stage_divisible:
                    current_size = new_size
                else:
                    break
                    
            checks['stage_by_stage'] = stage_results
            
            # Final size check (must be positive and divisible by window_size)
            final_size = current_size
            final_window_divisible = (final_size % window_size == 0) if final_size > 0 else False
            
            checks['final_compatibility'] = {
                'final_size': final_size,
                'window_divisible': final_window_divisible,
                'num_windows': final_size // window_size if final_window_divisible else -1
            }
            
            diagnostics['compatibility_checks'][dim_name] = checks
        
        # Overall compatibility - check if all dimensions pass all tests
        all_compatible = True
        for dim_name, checks in diagnostics['compatibility_checks'].items():
            if not (checks['window_divisible']['result'] and 
                   checks['total_downscaling_divisible']['result'] and
                   checks['final_compatibility']['window_divisible']):
                all_compatible = False
                break
        
        if all_compatible:
            message = f"✅ Input shape {input_shape} is fully compatible"
        else:
            failed_dims = []
            for dim_name, checks in diagnostics['compatibility_checks'].items():
                if not (checks['window_divisible']['result'] and 
                       checks['total_downscaling_divisible']['result'] and
                       checks['final_compatibility']['window_divisible']):
                    failed_dims.append(dim_name)
            message = f"❌ Incompatible dimensions: {failed_dims}"
            
        return all_compatible, message, diagnostics
    
    @staticmethod
    def suggest_nearest_compatible_size(
        target_shape: Tuple[int, int, int],
        window_size: int,
        downscaling_factors: Tuple[int, ...]
    ) -> Tuple[int, int, int]:
        """
        Suggest the nearest compatible size for a target shape.
        """
        total_downscaling = math.prod(downscaling_factors)
        lcm = math.lcm(window_size, total_downscaling)
        
        compatible_shape = []
        for dim_size in target_shape:
            # Round up to nearest multiple of LCM
            compatible_size = ((dim_size + lcm - 1) // lcm) * lcm
            compatible_shape.append(compatible_size)
            
        return tuple(compatible_shape)
    
    @staticmethod
    def print_compatibility_report(
        input_shape: Tuple[int, int, int],
        window_size: int, 
        downscaling_factors: Tuple[int, ...]
    ):
        """
        Print a detailed compatibility report.
        """
        is_compatible, message, diagnostics = RobustDimensionValidator.validate_exact_compatibility(
            input_shape, window_size, downscaling_factors
        )
        
        print("=" * 80)
        print("DIMENSION COMPATIBILITY REPORT")
        print("=" * 80)
        print(f"Input Shape: {input_shape}")
        print(f"Window Size: {window_size}")
        print(f"Downscaling Factors: {downscaling_factors}")
        print(f"Total Downscaling: {diagnostics['total_downscaling']}")
        print(f"Result: {message}")
        print()
        
        for dim_name, checks in diagnostics['compatibility_checks'].items():
            print(f"📐 {dim_name} Dimension Analysis:")
            print(f"   Window Divisibility: {checks['window_divisible']['calculation']} -> {'✅' if checks['window_divisible']['result'] else '❌'}")
            print(f"   Total Downscaling: {checks['total_downscaling_divisible']['calculation']} -> {'✅' if checks['total_downscaling_divisible']['result'] else '❌'}")
            
            print(f"   Stage-by-stage downscaling:")
            for stage in checks['stage_by_stage']:
                status = "✅" if stage['divisible'] else "❌"
                print(f"     Stage {stage['stage']}: {stage['input_size']} ÷ {stage['factor']} = {stage['output_size']} {status}")
                
            final = checks['final_compatibility']
            print(f"   Final: {final['final_size']} pixels, {final['num_windows']} windows -> {'✅' if final['window_divisible'] else '❌'}")
            print()
            
        if not is_compatible:
            suggested = RobustDimensionValidator.suggest_nearest_compatible_size(
                input_shape, window_size, downscaling_factors
            )
            print(f"💡 Suggested compatible size: {suggested}")
            
        print("=" * 80)


def create_robust_te_swin_model(
    input_channels: int = 1,
    num_classes: int = 2,
    target_patch_size: Tuple[int, int, int] = (128, 128, 128),
    model_size: str = "small",
    strict_validation: bool = True
):
    """
    Create a robust TE-Swin UNet3D model with guaranteed compatibility.
    """
    
    # Model configurations
    configs = {
        'tiny': {
            'hidden_dim': 32,
            'layers': (2, 2, 2, 2),
            'heads': (2, 4, 6, 8),
            'head_dim': 16,
            'window_size': 2,
            'downscaling_factors': (2, 2, 2, 2),
        },
        'small': {
            'hidden_dim': 48,
            'layers': (2, 2, 4, 2),
            'heads': (2, 4, 8, 16),
            'head_dim': 24,
            'window_size': 2,
            'downscaling_factors': (2, 2, 2, 2),
        },
        'base': {
            'hidden_dim': 96,
            'layers': (2, 2, 8, 2),
            'heads': (4, 8, 16, 32),
            'head_dim': 32,
            'window_size': 4,
            'downscaling_factors': (4, 2, 2, 2),
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Model size must be one of {list(configs.keys())}")
        
    config = configs[model_size]
    
    # Validate and adjust patch size
    validator = RobustDimensionValidator()
    
    is_compatible, message, _ = validator.validate_exact_compatibility(
        target_patch_size, config['window_size'], config['downscaling_factors']
    )
    
    if not is_compatible:
        if strict_validation:
            print(f"⚠️  Target patch size {target_patch_size} is not compatible")
            validator.print_compatibility_report(
                target_patch_size, config['window_size'], config['downscaling_factors']
            )
            
            suggested_size = validator.suggest_nearest_compatible_size(
                target_patch_size, config['window_size'], config['downscaling_factors']
            )
            print(f"🔧 Using suggested compatible size: {suggested_size}")
            recommended_patch_size = suggested_size
        else:
            print(f"⚠️  Warning: Patch size {target_patch_size} may cause issues")
            recommended_patch_size = target_patch_size
    else:
        print(f"✅ Patch size {target_patch_size} is compatible")
        recommended_patch_size = target_patch_size
    
    # Print compatible sizes for reference
    compatible_sizes = validator.calculate_compatible_sizes(
        min_size=32, max_size=512,
        window_size=config['window_size'],
        downscaling_factors=config['downscaling_factors']
    )
    
    print(f"📝 All compatible sizes (32-512): {compatible_sizes[:10]}{'...' if len(compatible_sizes) > 10 else ''}")
    
    # Create model (placeholder - would use actual model implementation)
    print(f"🚀 Creating {model_size} TE-Swin UNet3D model...")
    print(f"   Configuration: {config}")
    print(f"   Recommended patch size: {recommended_patch_size}")
    
    return None, recommended_patch_size  # Return None for now, replace with actual model


if __name__ == "__main__":
    # Test the validator
    validator = RobustDimensionValidator()
    
    # Test cases
    test_cases = [
        (64, 64, 64),    # Should work
        (128, 128, 128), # Should work  
        (96, 96, 96),    # Should fail
        (112, 112, 112), # Should fail
        (48, 48, 48),    # Might work with right config
    ]
    
    for shape in test_cases:
        print(f"\nTesting {shape}:")
        validator.print_compatibility_report(shape, 2, (2, 2, 2, 2))
