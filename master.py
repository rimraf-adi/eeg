import numpy as np
import pandas as pd
import mne
import logging
import sys
import time
import os
import gc
from datetime import datetime
from scipy.signal import welch, butter, filtfilt
from sklearn.linear_model import LinearRegression
from pathlib import Path
from typing import Dict, Tuple

def process_eeg_file(
    eeg_file_path: str,
    annotations_file_path: str,
    output_dir: str = "results",
    frequency_bands: Dict[str, Tuple[float, float]] = None,
    fs: int = 256
) -> bool:
    """
    Process a single EEG file and extract features for all frequency bands
    
    Args:
        eeg_file_path: Path to the EDF file
        annotations_file_path: Path to the annotations Excel file
        output_dir: Output directory for results
        frequency_bands: Dictionary of band_name -> (low_freq, high_freq)
        fs: Sampling frequency
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Default frequency bands
    if frequency_bands is None:
        frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 100)
        }
    
    base_filename = os.path.splitext(os.path.basename(eeg_file_path))[0]
    
    try:
        # Setup directories
        os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)
        
        print(f"Processing: {base_filename}")
        
        # Bipolar pairs configuration
        bipolar_pairs = [
            ('EEG Fp1-REF', 'EEG F7-REF'), ('EEG F7-REF',  'EEG T3-REF'),
            ('EEG T3-REF',  'EEG T5-REF'), ('EEG T5-REF',  'EEG O1-REF'),
            ('EEG Fp1-REF', 'EEG F3-REF'), ('EEG F3-REF',  'EEG C3-REF'),
            ('EEG C3-REF',  'EEG P3-REF'), ('EEG P3-REF',  'EEG O1-REF'),
            ('EEG Fz-REF',  'EEG Cz-REF'), ('EEG Cz-REF',  'EEG Pz-REF'),
            ('EEG Fp2-REF', 'EEG F4-REF'), ('EEG F4-REF',  'EEG C4-REF'),
            ('EEG C4-REF',  'EEG P4-REF'), ('EEG P4-REF',  'EEG O2-REF'),
            ('EEG Fp2-REF', 'EEG F8-REF'), ('EEG F8-REF',  'EEG T4-REF'),
            ('EEG T4-REF',  'EEG T6-REF'), ('EEG T6-REF',  'EEG O2-REF'),
        ]
        
        desired_order = [
            'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
            'Fz-Cz', 'Cz-Pz',
        ]
        
        # Normalize and reorder pairs
        bipolar_pairs = [(a.strip().upper(), b.strip().upper()) for a, b in bipolar_pairs]
        name_to_pair = {f"{a.replace('EEG ', '').replace('-REF', '')}-{b.replace('EEG ', '').replace('-REF', '')}": (a, b) for a, b in bipolar_pairs}
        reordered_pairs = [name_to_pair[name.upper()] for name in desired_order]
        
        anode = [a for a, _ in reordered_pairs]
        cathode = [b for _, b in reordered_pairs]
        ch_names = [f"{a.replace('EEG ', '').replace('-REF', '').lower().capitalize()}-{b.replace('EEG ', '').replace('-REF', '').lower().capitalize()}" 
                   for a, b in reordered_pairs]
        
        # Load EEG data
        mne.set_log_level('ERROR')
        raw = mne.io.read_raw_edf(eeg_file_path, preload=True, verbose=False)
        raw.rename_channels(lambda ch: ch.upper())
        
        # Drop unnecessary channels
        drop_candidates = ['ECG EKG', 'RESP EFFORT','ECG EKG-REF','RESP EFFORT-REF']
        to_drop = [ch for ch in drop_candidates if ch in raw.ch_names]
        if to_drop:
            raw.drop_channels(to_drop)
        
        # Set bipolar reference
        raw = mne.set_bipolar_reference(raw, anode=anode, cathode=cathode, ch_name=ch_names, copy=True, verbose=False)
        eeg = raw.get_data()
        
        # Clean up raw data immediately
        del raw
        gc.collect()
        
        # Load annotations
        annotations = pd.read_excel(annotations_file_path)
        annotations = annotations.dropna(subset=['FILE ID'])  # Only drop rows where FILE ID is NaN
        annotations = annotations.sort_values(by=['FILE ID','Start'], ascending=[True,True])
        
        # Extract file ID from filename (e.g., 'eeg1' -> 1.0)
        file_id = float(base_filename.replace('eeg', ''))
        
        # Filter annotations for current file only
        file_annotations = annotations[annotations['FILE ID'] == file_id].copy()
        
        if file_annotations.empty:
            print(f"No annotations found for {base_filename}")
            return False
        
        print(f"Found {len(file_annotations)} annotations for {base_filename}")
        
        # Extract segments
        seizure_segments = []
        non_seizure_segments = []
        
        channel_cols = ['Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp1-F3', 'F3-C3', 'C3-P3',
                        'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F7', 'F7-T3', 'T3-T5', 
                        'T5-O1', 'Fz-Cz', 'Cz-Pz']
        
        total_seizure_channels = 0
        total_non_seizure_channels = 0
        
        for i in range(len(file_annotations)):
            start = int(file_annotations['Start'].iloc[i] * fs) 
            stop = int(file_annotations['Stop'].iloc[i] * fs)
            
            # Ensure indices are within bounds
            if start >= eeg.shape[1] or stop > eeg.shape[1]:
                print(f"Warning: Annotation indices out of bounds for {base_filename}, annotation {i}")
                continue
            
            annotation_seizure_count = 0
            annotation_non_seizure_count = 0
            
            for ch_idx, col in enumerate(channel_cols):
                if ch_idx < len(eeg):
                    segment = eeg[ch_idx][start:stop]
                    if file_annotations[col].iloc[i] == 1: 
                        seizure_segments.append(segment)
                        annotation_seizure_count += 1
                        total_seizure_channels += 1
                    else:  
                        non_seizure_segments.append(segment)
                        annotation_non_seizure_count += 1
                        total_non_seizure_channels += 1
            
            print(f"  Annotation {i+1}: {annotation_seizure_count} seizure, {annotation_non_seizure_count} non-seizure channels")
        
        print(f"Total seizure channels: {total_seizure_channels}, Total non-seizure channels: {total_non_seizure_channels}")
        
        # Clean up EEG data and annotations
        del eeg, file_annotations
        gc.collect()
        
        print(f"Seizure segments: {len(seizure_segments)}, Non-seizure: {len(non_seizure_segments)}")
        
        # Extract features for all frequency bands
        all_seizure_features = []
        all_non_seizure_features = []
        
        for band_name, (low_freq, high_freq) in frequency_bands.items():
            print(f"Processing {band_name} band...")
            
            # Process seizure segments
            seizure_features = extract_features_optimized(seizure_segments, (low_freq, high_freq), fs, band_name)
            if not seizure_features.empty:
                all_seizure_features.append(seizure_features)
            
            # Process non-seizure segments  
            non_seizure_features = extract_features_optimized(non_seizure_segments, (low_freq, high_freq), fs, band_name)
            if not non_seizure_features.empty:
                all_non_seizure_features.append(non_seizure_features)
            
            # Clean up intermediate results
            del seizure_features, non_seizure_features
            gc.collect()
        
        # Create separate CSV files
        if all_seizure_features:
            seizure_df = pd.concat(all_seizure_features, axis=1)
            seizure_df['segment_type'] = 'seizure'
            seizure_df['file_id'] = base_filename
            
            seizure_file = os.path.join(output_dir, "features", f"{base_filename}_seizure_features.csv")
            seizure_df.to_csv(seizure_file, index=False)
            print(f"Exported seizure features: {seizure_file}")
            del seizure_df
        
        if all_non_seizure_features:
            non_seizure_df = pd.concat(all_non_seizure_features, axis=1)
            non_seizure_df['segment_type'] = 'non_seizure'
            non_seizure_df['file_id'] = base_filename
            
            non_seizure_file = os.path.join(output_dir, "features", f"{base_filename}_non_seizure_features.csv")
            non_seizure_df.to_csv(non_seizure_file, index=False)
            print(f"Exported non-seizure features: {non_seizure_file}")
            del non_seizure_df
        
        # Create combined file
        if all_seizure_features or all_non_seizure_features:
            combined_features = []
            
            if all_seizure_features:
                seizure_combined = pd.concat(all_seizure_features, axis=1)
                seizure_combined['segment_type'] = 'seizure'
                seizure_combined['file_id'] = base_filename
                combined_features.append(seizure_combined)
            
            if all_non_seizure_features:
                non_seizure_combined = pd.concat(all_non_seizure_features, axis=1)
                non_seizure_combined['segment_type'] = 'non_seizure'
                non_seizure_combined['file_id'] = base_filename
                combined_features.append(non_seizure_combined)
            
            if combined_features:
                final_combined = pd.concat(combined_features, ignore_index=True)
                combined_file = os.path.join(output_dir, "features", f"{base_filename}_all_features.csv")
                final_combined.to_csv(combined_file, index=False)
                print(f"Exported combined features: {combined_file}")
                del final_combined, combined_features
        
        # Clean up all remaining variables
        del all_seizure_features, all_non_seizure_features, seizure_segments, non_seizure_segments
        del bipolar_pairs, reordered_pairs, anode, cathode, ch_names
        del base_filename, desired_order, channel_cols, name_to_pair
        
        # Force garbage collection
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Error processing {base_filename}: {str(e)}")
        
        # Clean up on error - delete specific variables if they exist
        cleanup_vars = ['eeg', 'raw', 'annotations', 'seizure_segments', 'non_seizure_segments', 
                       'all_features', 'final_features', 'bipolar_pairs', 'reordered_pairs']
        for var_name in cleanup_vars:
            try:
                if var_name in locals():
                    del locals()[var_name]
            except:
                pass
        gc.collect()
        
        return False


def extract_features_optimized(segments, band, fs, band_name):
    """Optimized feature extraction with memory management"""
    if not segments:
        return pd.DataFrame()
    
    n_segments = len(segments)
    slopes = np.zeros(n_segments)
    intercepts = np.zeros(n_segments)
    midbands = np.zeros(n_segments)
    
    midband_freq = (band[0] + band[1]) / 2.0
    
    for seg_idx, y in enumerate(segments):
        try:
            y = np.asarray(y)
            if y.size == 0:
                slopes[seg_idx] = np.nan
                intercepts[seg_idx] = np.nan
                midbands[seg_idx] = np.nan
                continue
            
            # Zero-pad if too small
            min_length = 512
            if y.size < min_length:
                y = np.pad(y, (0, min_length - y.size), mode='constant', constant_values=0)
            
            # Bandpass filter
            segment = bandpass_filter(y, band[0], band[1], fs)
            
            # PSD
            nperseg_val = min(512, segment.size)
            if nperseg_val < 4:
                slopes[seg_idx] = np.nan
                intercepts[seg_idx] = np.nan
                midbands[seg_idx] = np.nan
                continue
            
            f, Pxx = welch(segment, fs=fs, nperseg=nperseg_val)
            
            freq_mask = (f >= band[0]) & (f <= band[1])
            f_filtered = f[freq_mask]
            Pxx_filtered = Pxx[freq_mask]
            
            if len(f_filtered) < 2:
                slopes[seg_idx] = np.nan
                intercepts[seg_idx] = np.nan
                midbands[seg_idx] = np.nan
                continue
            
            log_Pxx = np.log10(Pxx_filtered + 1e-12)
            
            # Linear regression
            model = LinearRegression()
            model.fit(f_filtered.reshape(-1, 1), log_Pxx)
            
            slopes[seg_idx] = model.coef_[0]
            intercepts[seg_idx] = model.intercept_
            midbands[seg_idx] = model.predict([[midband_freq]])[0]
            
            # Clean up segment variables
            del y, segment, f, Pxx, f_filtered, Pxx_filtered, log_Pxx, model
            
        except Exception as e:
            slopes[seg_idx] = np.nan
            intercepts[seg_idx] = np.nan
            midbands[seg_idx] = np.nan
    
    return pd.DataFrame({
        f'slope_{band_name}': slopes,
        f'intercept_{band_name}': intercepts,
        f'midband_{band_name}': midbands
    })


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply bandpass filter to data"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)
    
success = process_eeg_file('C:/bt23ece064/dataset/eeg1.edf', 'C:/bt23ece064/dataset/seizure_per_channel_Dec_2017.xlsx', 'C:/bt23ece064/newannotations/results/')
print(success)