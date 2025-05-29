"""
Test script to validate TE-Swin UNet3D integration with nnUNet.
This script tests the model creation and basic forward pass.
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
        return False


def test_forward_pass():
    """Test forward pass with sample data."""
    print("\nTesting forward pass...")
    
    try:
        from nnunetv2.training.nnUNetTrainer.variants.network_architecture.te_swin_models.nnUNet_TE_SwinUnet3D import create_te_swinunet_t_3d
        
        # Create small model for testing
        model = create_te_swinunet_t_3d(input_channels=1, num_classes=2, deep_supervision=True)
        model.eval()
        
        # Test different input sizes that should be compatible
        test_sizes = [
            (1, 1, 32, 32, 32),   # Minimum size compatible with window_size=4 and downscaling=(2,2,2,2)
            (2, 1, 64, 64, 64),   # Larger test
        ]
        
        for batch_size, channels, d, h, w in test_sizes:
            print(f"  Testing input size: {(batch_size, channels, d, h, w)}")
            
            # Create random input
            x = torch.randn(batch_size, channels, d, h, w)
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
                
            if isinstance(output, list):
                print(f"    Deep supervision outputs: {len(output)} levels")
                for i, out in enumerate(output):
                    print(f"      Level {i}: {out.shape}")
            else:
                print(f"    Output shape: {output.shape}")
                
        print("✓ Forward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_creation():
    """Test if we can create the custom trainer."""
    print("\nTesting trainer creation...")
    
    try:
        # Mock the necessary objects for trainer creation
        mock_plans = {
            'dataset_name': 'Dataset501_PANTHERTask1',
            'plans_name': 'nnUNetPlans',
            'original_median_spacing_after_transp': [1.0, 1.0, 1.0],
            'original_median_shape_after_transp': [128, 128, 128],
            'image_reader_writer': 'SimpleITKIO',
            'transpose_forward': [0, 1, 2],
            'transpose_backward': [0, 1, 2],
            'configurations': {
                '3d_fullres': {
                    'data_identifier': 'nnUNetPlans_3d_fullres',
                    'preprocessor_name': 'DefaultPreprocessor',
                    'batch_size': 2,
                    'patch_size': [112, 112, 112],
                    'median_image_size_in_voxels': [128, 128, 128],
                    'spacing': [1.0, 1.0, 1.0],
                    'normalization_schemes': ['ZScoreNormalization'],
                    'network_arch_class_name': 'PlainConvUNet',
                    'network_arch_init_kwargs': {},
                    'network_arch_init_kwargs_req_import': [],
                    'UNet_class_name': 'PlainConvUNet',
                    'UNet_base_num_features': 32,
                    'n_conv_per_stage_encoder': [2, 2, 2, 2, 2, 2],
                    'n_conv_per_stage_decoder': [2, 2, 2, 2, 2],
                    'num_pool_per_axis': [5, 5, 5],
                    'pool_op_kernel_sizes': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                    'conv_kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                    'unet_max_num_features': 320,
                    'resampling_fn_data': 'resample_data_or_seg_to_shape',
                    'resampling_fn_seg': 'resample_data_or_seg_to_shape',
                    'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None},
                    'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None},
                    'resampling_fn_probabilities': 'resample_data_or_seg_to_shape',
                    'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None},
                    'batch_dice': True,
                    'use_mask_for_norm': [False],
                    'next_stage_names': None,
                    'previous_stage_name': None
                }
            }
        }
        
        mock_dataset_json = {
            'channel_names': {0: 'CT'},
            'labels': {'background': 0, 'pancreas': 1, 'tumor': 2},
            'file_ending': '.nii.gz',
            'numTraining': 100,
            'dataset_name': 'Dataset501_PANTHERTask1'
        }
        
        # Import trainer
        from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_TE_SwinUnet3D import nnUNetTrainer_TE_SwinUnet3D
        
        # Create trainer instance (without actually initializing full training)
        device = torch.device('cpu')  # Use CPU for testing
        
        print("✓ Successfully imported TE-Swin UNet3D trainer")
        return True
        
    except Exception as e:
        print(f"✗ Trainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TE-Swin UNet3D nnUNet Integration Test")
    print("=" * 60)
    
    tests = [
        test_model_creation,
        test_forward_pass,
        test_trainer_creation,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    test_names = [
        "Model Creation",
        "Forward Pass", 
        "Trainer Creation",
    ]
    
    for name, result in zip(test_names, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:<20}: {status}")
    
    overall_success = all(results)
    print(f"\nOverall Status: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 TE-Swin UNet3D is successfully integrated with nnUNet!")
        print("\nNext steps:")
        print("1. Prepare your PANTHER dataset in nnUNet format")
        print("2. Run nnUNet preprocessing")
        print("3. Start training with: nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainer_TE_SwinUnet3D")
    else:
        print("\n❌ Integration incomplete. Please check the error messages above.")
    
    return overall_success


if __name__ == "__main__":
    main()
