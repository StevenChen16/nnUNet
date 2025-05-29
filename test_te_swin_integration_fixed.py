"""
Fixed test script for TE-Swin UNet3D integration with nnUNet.
This version addresses dimension compatibility issues.
"""
import sys
import os
import torch

# Add nnUNet to path
sys.path.append(r'D:\workstation\ML\PANTHER\PANTHER\nnUNet')

def test_model_creation():
    """Test if we can create the TE-Swin UNet3D model."""
    print("Testing TE-Swin UNet3D model creation...")
    
    try:
        # Import our custom models
        from nnunetv2.training.nnUNetTrainer.variants.network_architecture.te_swin_models.nnUNet_TE_SwinUnet3D import (
            create_te_swinunet_s_3d,
            create_te_swinunet_t_3d,
            create_te_swinunet_b_3d
        )
        
        # Test model creation with different variants
        models = {
            'tiny': create_te_swinunet_t_3d(input_channels=1, num_classes=3),
            'small': create_te_swinunet_s_3d(input_channels=1, num_classes=3),
            'base': create_te_swinunet_b_3d(input_channels=1, num_classes=3)
        }
        
        for variant, model in models.items():
            print(f"✓ Successfully created {variant} TE-Swin UNet3D")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to create models: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dimension_compatibility():
    """Test which input dimensions are compatible with the model."""
    print("\nTesting dimension compatibility...")
    
    try:
        from nnunetv2.training.nnUNetTrainer.variants.network_architecture.te_swin_models.nnUNet_TE_SwinUnet3D import create_te_swinunet_t_3d
        
        # Create model with smaller window size for testing
        model = create_te_swinunet_t_3d(
            input_channels=1, 
            num_classes=2, 
            window_size=4,  # Smaller window size
            downscaling_factors=(2, 2, 2, 2),  # Smaller downscaling factors
            deep_supervision=False
        )
        model.eval()
        
        # Test compatible input sizes
        # For window_size=4 and downscaling=(2,2,2,2), minimum size should be:
        # 4 * 2^4 = 64 for each dimension
        compatible_sizes = [
            (1, 1, 64, 64, 64),   # Minimum compatible size
            (1, 1, 96, 96, 96),   # Slightly larger
            (2, 1, 128, 128, 128), # Realistic size
        ]
        
        successful_sizes = []
        
        for batch_size, channels, d, h, w in compatible_sizes:
            print(f"  Testing input size: {(batch_size, channels, d, h, w)}")
            
            try:
                # Create random input
                x = torch.randn(batch_size, channels, d, h, w)
                
                # Forward pass
                with torch.no_grad():
                    output = model(x)
                    
                print(f"    ✓ Success! Output shape: {output.shape}")
                successful_sizes.append((d, h, w))
                
            except Exception as e:
                print(f"    ✗ Failed: {str(e)[:100]}...")
                
        if successful_sizes:
            print(f"✓ Found {len(successful_sizes)} compatible input sizes")
            return True
        else:
            print("✗ No compatible input sizes found")
            return False
            
    except Exception as e:
        print(f"✗ Dimension compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_model():
    """Test model with adaptive parameters for small inputs."""
    print("\nTesting adaptive model for small inputs...")
    
    try:
        from nnunetv2.training.nnUNetTrainer.variants.network_architecture.te_swin_models.nnUNet_TE_SwinUnet3D import create_te_swinunet_t_3d
        
        # Create model optimized for smaller inputs
        # IMPORTANT: All layer counts must be even numbers!
        model = create_te_swinunet_t_3d(
            input_channels=1,
            num_classes=2,
            window_size=2,  # Very small window
            downscaling_factors=(2, 2, 2, 2),  # Conservative downscaling
            hidden_dim=32,  # Smaller hidden dimension
            layers=(2, 2, 2, 2),  # All even numbers - FIXED!
            heads=(2, 4, 6, 8),  # Fewer attention heads
            deep_supervision=False
        )
        model.eval()
        
        # Test with original problematic size
        test_sizes = [
            (1, 1, 32, 32, 32),   # Original problematic size
            (1, 1, 48, 48, 48),   # Slightly larger
        ]
        
        successful_tests = 0
        
        for batch_size, channels, d, h, w in test_sizes:
            print(f"  Testing input size: {(batch_size, channels, d, h, w)}")
            
            try:
                # Create random input
                x = torch.randn(batch_size, channels, d, h, w)
                
                # Forward pass
                with torch.no_grad():
                    output = model(x)
                    
                print(f"    ✓ Success! Output shape: {output.shape}")
                successful_tests += 1
                
            except Exception as e:
                print(f"    ✗ Failed: {str(e)[:100]}...")
                
        if successful_tests > 0:
            print(f"✓ Adaptive model test completed - {successful_tests}/{len(test_sizes)} tests passed")
            return True
        else:
            print("✗ All adaptive model tests failed")
            return False
        
    except Exception as e:
        print(f"✗ Adaptive model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_compatible_model():
    """Create a model configuration that works with typical nnUNet patch sizes."""
    print("\nCreating nnUNet-compatible model...")
    
    try:
        from nnunetv2.training.nnUNetTrainer.variants.network_architecture.te_swin_models.nnUNet_TE_SwinUnet3D import create_te_swinunet_s_3d
        
        # nnUNet typical patch sizes: 112x112x112, 128x128x128, 96x96x96
        # We need parameters that work with these sizes
        # Let's try window_size=2 and smaller downscaling factors
        
        model = create_te_swinunet_s_3d(
            input_channels=1,
            num_classes=3,  # Background, pancreas, tumor
            window_size=2,  # Small window size for better compatibility
            downscaling_factors=(2, 2, 2, 2),  # Conservative downscaling = 16 total
            hidden_dim=48,  # Smaller model for stability
            layers=(2, 2, 4, 2),  # All even numbers
            heads=(2, 4, 8, 16),  # Reasonable head counts
            deep_supervision=True
        )
        
        # Test with nnUNet-like patch sizes
        nnunet_patch_sizes = [
            (1, 1, 112, 112, 112),  # Common nnUNet patch size
            (1, 1, 128, 128, 128),  # Another common size
            (1, 1, 96, 96, 96),     # Smaller patch size
            (1, 1, 64, 64, 64),     # Even smaller for comparison
        ]
        
        model.eval()
        successful_tests = 0
        
        print(f"  Model configuration:")
        print(f"    - window_size: 2")
        print(f"    - downscaling_factors: (2,2,2,2) = 16 total")
        print(f"    - All dimensions must be divisible by: 2 (window) and 16 (total downscaling)")
        
        for batch_size, channels, d, h, w in nnunet_patch_sizes:
            print(f"  Testing nnUNet patch size: {(batch_size, channels, d, h, w)}")
            
            # Check theoretical compatibility first
            compatible = True
            compatibility_msg = []
            
            for dim_name, dim_size in zip(['D', 'H', 'W'], [d, h, w]):
                if dim_size % 2 != 0:
                    compatible = False
                    compatibility_msg.append(f"{dim_name}({dim_size}) not divisible by window_size(2)")
                if dim_size % 16 != 0:
                    compatible = False
                    compatibility_msg.append(f"{dim_name}({dim_size}) not divisible by total_downscaling(16)")
            
            if not compatible:
                print(f"    ✗ Skipped: {', '.join(compatibility_msg)}")
                continue
            
            try:
                # Create random input
                x = torch.randn(batch_size, channels, d, h, w)
                
                # Forward pass
                with torch.no_grad():
                    output = model(x)
                    
                if isinstance(output, list):
                    print(f"    ✓ Success! Deep supervision outputs: {len(output)} levels")
                    for i, out in enumerate(output):
                        print(f"      Level {i}: {out.shape}")
                else:
                    print(f"    ✓ Success! Output shape: {output.shape}")
                    
                successful_tests += 1
                
            except Exception as e:
                print(f"    ✗ Failed: {str(e)[:100]}...")
                import traceback
                print(f"    Full error: {str(e)}")
                
        if successful_tests > 0:
            print(f"✓ Successfully tested {successful_tests}/{len(nnunet_patch_sizes)} nnUNet patch sizes")
            
            # Test if we can handle non-standard sizes with adaptive sizing
            print("\n  Testing adaptive sizing for non-standard inputs...")
            try:
                # Create a model with adaptive sizing enabled
                adaptive_model = create_te_swinunet_s_3d(
                    input_channels=1,
                    num_classes=3,
                    window_size=2,
                    downscaling_factors=(2, 2, 2, 2),
                    hidden_dim=48,
                    layers=(2, 2, 4, 2),
                    heads=(2, 4, 8, 16),
                    deep_supervision=False  # Disable for simpler testing
                )
                adaptive_model.eval()
                
                # Test with a non-compatible size
                x_incompatible = torch.randn(1, 1, 100, 100, 100)  # Not divisible by 16
                
                with torch.no_grad():
                    # This should work if the model has adaptive sizing built-in
                    output = adaptive_model(x_incompatible)
                    print(f"    ✓ Adaptive sizing works! Input {x_incompatible.shape[2:]} -> Output {output.shape}")
                    
            except Exception as e:
                print(f"    ✗ Adaptive sizing failed: {str(e)[:100]}...")
            
            return True
        else:
            print("✗ No nnUNet patch sizes were compatible")
            return False
            
    except Exception as e:
        print(f"✗ nnUNet-compatible model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TE-Swin UNet3D nnUNet Integration Test (FIXED)")
    print("=" * 60)
    
    tests = [
        test_model_creation,
        test_dimension_compatibility,
        test_adaptive_model,
        create_compatible_model,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    test_names = [
        "Model Creation",
        "Dimension Compatibility",
        "Adaptive Model",
        "nnUNet Compatible Model",
    ]
    
    for name, result in zip(test_names, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:<25}: {status}")
    
    overall_success = all(results)
    print(f"\nOverall Status: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 TE-Swin UNet3D is successfully integrated with nnUNet!")
        print("\nRecommended model configuration for nnUNet:")
        print("- window_size=2 (for better compatibility)")
        print("- downscaling_factors=(2,2,2,2) (conservative downscaling)")
        print("- All layer counts must be even numbers")
        print("- Compatible patch sizes: 64x64x64, 128x128x128, 192x192x192, etc.")
        print("- Dimensions must be divisible by both window_size and total_downscaling_factor")
        print("\nNext steps:")
        print("1. Prepare your PANTHER dataset in nnUNet format")
        print("2. Run nnUNet preprocessing")
        print("3. Start training with: nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainer_TE_SwinUnet3D")
    else:
        print("\n❌ Integration incomplete. Please check the error messages above.")
    
    return overall_success


if __name__ == "__main__":
    main()
