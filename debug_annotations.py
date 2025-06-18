#!/usr/bin/env python3
"""
Debug script to examine annotation structure and verify segmentation logic
"""

import pandas as pd
import numpy as np

def debug_annotations():
    """Debug the annotation structure"""
    
    # Load annotations
    annotations_file = 'C:/bt23ece064/dataset/seizure_per_channel_Dec_2017.xlsx'
    annotations = pd.read_excel(annotations_file)
    
    print("Annotation file structure:")
    print(f"Total annotations: {len(annotations)}")
    print(f"Columns: {list(annotations.columns)}")
    print(f"Unique FILE IDs: {sorted(annotations['FILE ID'].unique())}")
    
    # Check annotations per file
    print("\nAnnotations per file:")
    for file_id in sorted(annotations['FILE ID'].unique()):
        file_annotations = annotations[annotations['FILE ID'] == file_id]
        print(f"File {file_id}: {len(file_annotations)} annotations")
    
    # Check channel columns
    channel_cols = ['Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp1-F3', 'F3-C3', 'C3-P3',
                    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F7', 'F7-T3', 'T3-T5', 
                    'T5-O1', 'Fz-Cz', 'Cz-Pz']
    
    print(f"\nChannel columns: {len(channel_cols)}")
    print(f"Channel columns: {channel_cols}")
    
    # Check a few sample annotations
    print("\nSample annotations (first 3):")
    for i in range(min(3, len(annotations))):
        file_id = annotations['FILE ID'].iloc[i]
        start = annotations['Start'].iloc[i]
        stop = annotations['Stop'].iloc[i]
        
        seizure_channels = []
        non_seizure_channels = []
        
        for col in channel_cols:
            if annotations[col].iloc[i] == 1:
                seizure_channels.append(col)
            else:
                non_seizure_channels.append(col)
        
        print(f"  Annotation {i+1}: File {file_id}, Time {start:.1f}s-{stop:.1f}s")
        print(f"    Seizure channels: {len(seizure_channels)}")
        print(f"    Non-seizure channels: {len(non_seizure_channels)}")
        print(f"    Seizure: {seizure_channels[:3]}{'...' if len(seizure_channels) > 3 else ''}")
        print(f"    Non-seizure: {non_seizure_channels[:3]}{'...' if len(non_seizure_channels) > 3 else ''}")

if __name__ == "__main__":
    debug_annotations() 