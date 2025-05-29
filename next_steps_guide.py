"""
NEXT STEPS: Setting up TE-Swin UNet3D with nnUNet for PANTHER Challenge

This guide will walk you through the complete setup process from data preparation to training.
"""

import os
import sys

def print_next_steps():
    print("="*80)
    print("🚀 NEXT STEPS: TE-SWIN UNET3D + nnUNet FOR PANTHER CHALLENGE")
    print("="*80)
    
    print("""
📋 STEP 1: PREPARE PANTHER DATASET
==================================

1.1 Download PANTHER dataset:
    - Get dataset from Zenodo (record 15192302)
    - Extract to a working directory

1.2 Organize data structure:
    PANTHER_raw/
    ├── Task1_CT/                    # CT images (if available)
    ├── Task2_MRLinac/              # MR-Linac images (main focus)
    │   ├── images/
    │   │   ├── case_001.nii.gz
    │   │   ├── case_002.nii.gz
    │   │   └── ...
    │   └── labels/
    │       ├── case_001.nii.gz
    │       ├── case_002.nii.gz
    │       └── ...
    └── dataset.json

📋 STEP 2: CONVERT TO nnUNet FORMAT
===================================

2.1 Set nnUNet environment variables:
    export nnUNet_raw_data_base="/path/to/nnUNet_raw_data_base"
    export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
    export nnUNet_RESULTS_FOLDER="/path/to/nnUNet_results"

2.2 Create conversion script (save as convert_panther_to_nnunet.py):
""")
    
    print("""
2.3 Run conversion:
    python convert_panther_to_nnunet.py

📋 STEP 3: nnUNet PREPROCESSING
===============================

3.1 Plan and preprocess:
    nnUNetv2_plan_and_preprocess -d 602 --verify_dataset_integrity

3.2 Check preprocessing results:
    - Verify patch sizes are compatible (should be 64x64x64 or 128x128x128)
    - Check normalization parameters
    - Verify data augmentation settings

📋 STEP 4: START TRAINING
=========================

4.1 Train with TE-Swin UNet3D (Small variant recommended):
    nnUNetv2_train 602 3d_fullres 0 -tr nnUNetTrainer_TE_SwinUnet3D_small

4.2 Alternative training options:
    # Tiny variant (for limited GPU memory)
    nnUNetv2_train 602 3d_fullres 0 -tr nnUNetTrainer_TE_SwinUnet3D_tiny
    
    # Base variant (for high-end GPUs)
    nnUNetv2_train 602 3d_fullres 0 -tr nnUNetTrainer_TE_SwinUnet3D_base

4.3 Monitor training:
    - Check logs in nnUNet_RESULTS_FOLDER
    - Monitor GPU memory usage
    - Watch for convergence

📋 STEP 5: VALIDATION AND TESTING
==================================

5.1 Run inference on validation set:
    nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 602 -c 3d_fullres -f 0 -tr nnUNetTrainer_TE_SwinUnet3D_small

5.2 Evaluate results:
    nnUNetv2_evaluate_predictions -pred OUTPUT_FOLDER -gt GT_FOLDER -json

📋 STEP 6: TROUBLESHOOTING TIPS
===============================

6.1 Memory issues:
    - Use smaller batch size (batch_size=1)
    - Use tiny variant
    - Enable gradient checkpointing
    - Reduce patch size to 64x64x64

6.2 Training issues:
    - Start with lower learning rate (1e-4)
    - Use mixed precision training
    - Check data loading pipeline

6.3 Performance issues:
    - Experiment with different window sizes
    - Adjust texture loss weights
    - Fine-tune attention mechanisms

🔧 IMMEDIATE ACTION ITEMS:
=========================
""")

def create_conversion_script():
    conversion_script = '''"""
Convert PANTHER dataset to nnUNet format
"""
import os
import shutil
import json
from pathlib import Path
import nibabel as nib

def convert_panther_to_nnunet(panther_path, nnunet_raw_path, dataset_id=602):
    """
    Convert PANTHER dataset to nnUNet format.
    
    Args:
        panther_path: Path to PANTHER dataset
        nnunet_raw_path: Path to nnUNet raw data base
        dataset_id: nnUNet dataset ID (default: 602)
    """
    
    dataset_name = f"Dataset{dataset_id:03d}_PANTHER_Task2"
    output_dir = Path(nnunet_raw_path) / dataset_name
    
    # Create output directories
    (output_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (output_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (output_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    
    # Convert training data
    images_path = Path(panther_path) / "Task2_MRLinac" / "images"
    labels_path = Path(panther_path) / "Task2_MRLinac" / "labels"
    
    training_cases = []
    
    for img_file in images_path.glob("*.nii.gz"):
        case_id = img_file.stem.replace(".nii", "")
        
        # Copy image
        shutil.copy2(img_file, output_dir / "imagesTr" / f"{case_id}_0000.nii.gz")
        
        # Copy label if exists
        label_file = labels_path / img_file.name
        if label_file.exists():
            shutil.copy2(label_file, output_dir / "labelsTr" / f"{case_id}.nii.gz")
            training_cases.append(case_id)
    
    # Create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "T2"
        },
        "labels": {
            "background": 0,
            "pancreas": 1,
            "tumor": 2
        },
        "numTraining": len(training_cases),
        "file_ending": ".nii.gz",
        "dataset_name": dataset_name,
        "reference": "PANTHER Challenge - Task 2",
        "licence": "CC-BY-SA 4.0",
        "description": "Pancreatic tumor segmentation on MR-Linac images"
    }
    
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"✓ Converted {len(training_cases)} cases to nnUNet format")
    print(f"✓ Dataset saved to: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    # Adjust these paths according to your setup
    PANTHER_PATH = "/path/to/PANTHER_raw"
    NNUNET_RAW_PATH = os.environ.get("nnUNet_raw_data_base", "/path/to/nnUNet_raw_data_base")
    
    convert_panther_to_nnunet(PANTHER_PATH, NNUNET_RAW_PATH)
'''
    
    with open("D:\\workstation\\ML\\PANTHER\\PANTHER\\nnUNet\\convert_panther_to_nnunet.py", "w") as f:
        f.write(conversion_script)
    
    print("✓ Created convert_panther_to_nnunet.py")

def main():
    print_next_steps()
    
    print("""
1. 📁 CREATE DATA CONVERSION SCRIPT:
   Run this Python script to create the conversion tool:
""")
    
    create_conversion_script()
    
    print("""
2. 🔧 SET ENVIRONMENT VARIABLES:
   Add these to your ~/.bashrc or set them in your current session:
   
   export nnUNet_raw_data_base="/your/path/to/nnUNet_raw_data_base"
   export nnUNet_preprocessed="/your/path/to/nnUNet_preprocessed"  
   export nnUNet_RESULTS_FOLDER="/your/path/to/nnUNet_results"

3. 📊 DOWNLOAD PANTHER DATASET:
   Visit: https://zenodo.org/records/15192302
   Extract to a working directory

4. 🚀 START WITH CONVERSION:
   python convert_panther_to_nnunet.py

5. ⚡ BEGIN TRAINING:
   nnUNetv2_plan_and_preprocess -d 602 --verify_dataset_integrity
   nnUNetv2_train 602 3d_fullres 0 -tr nnUNetTrainer_TE_SwinUnet3D_small

🎯 EXPECTED TIMELINE:
   - Data preparation: 1-2 hours
   - Preprocessing: 30 minutes - 2 hours  
   - Training: 6-24 hours (depending on GPU and data size)
   - Validation: 1-2 hours

💡 NEED HELP?
   - Check logs in nnUNet_RESULTS_FOLDER
   - Monitor GPU memory with nvidia-smi
   - Adjust batch size if out of memory
   - Use tiny variant for limited resources

Ready to start? Begin with downloading the PANTHER dataset! 🚀
""")

if __name__ == "__main__":
    main()
