#!/usr/bin/env python3
"""
Test script to run segmentation on a single file and see detailed output
"""

import os
from master import process_eeg_file

def test_single_file():
    """Test segmentation on a single EEG file"""
    
    annotations_file = 'C:/bt23ece064/dataset/seizure_per_channel_Dec_2017.xlsx'
    output_dir = "test_results"
    
    # Test with eeg1
    eeg_file = 'C:/bt23ece064/dataset/eeg4.edf'
    
    if os.path.exists(eeg_file):
        print("Testing segmentation on eeg1...")
        print("=" * 60)
        
        success = process_eeg_file(
            eeg_file_path=eeg_file,
            annotations_file_path=annotations_file,
            output_dir=output_dir
        )
        
        if success:
            print("✅ Processing completed successfully")
        else:
            print("❌ Processing failed")
    else:
        print(f"⚠️  {eeg_file} not found")

if __name__ == "__main__":
    test_single_file() 