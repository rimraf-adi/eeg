import numpy as np
import pandas as pd
import mne
from scipy.signal import welch, butter, filtfilt

# generate bipplar pairs
bipolar_pairs = [
    ('EEG Fp1-REF', 'EEG F7-REF'),
    ('EEG F7-REF',  'EEG T3-REF'),
    ('EEG T3-REF',  'EEG T5-REF'),
    ('EEG T5-REF',  'EEG O1-REF'),
    ('EEG Fp1-REF', 'EEG F3-REF'),
    ('EEG F3-REF',  'EEG C3-REF'),
    ('EEG C3-REF',  'EEG P3-REF'),
    ('EEG P3-REF',  'EEG O1-REF'),
    ('EEG Fz-REF',  'EEG Cz-REF'),
    ('EEG Cz-REF',  'EEG Pz-REF'),
    ('EEG Fp2-REF', 'EEG F4-REF'),
    ('EEG F4-REF',  'EEG C4-REF'),
    ('EEG C4-REF',  'EEG P4-REF'),
    ('EEG P4-REF',  'EEG O2-REF'),
    ('EEG Fp2-REF', 'EEG F8-REF'),
    ('EEG F8-REF',  'EEG T4-REF'),
    ('EEG T4-REF',  'EEG T6-REF'),
    ('EEG T6-REF',  'EEG O2-REF'),
]

desired_order = [
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fz-Cz', 'Cz-Pz',
]

def normalize_channel(ch):
    return ch.strip().upper()

def pair_name(pair):
    left = pair[0].replace('EEG ', '').replace('-REF', '').strip().upper()
    right = pair[1].replace('EEG ', '').replace('-REF', '').strip().upper()
    return f"{left}-{right}"

def make_ch_names(pairs):
    def pretty(ch):
        clean = ch.replace('EEG ', '').replace('-REF', '').strip().lower()
        return clean.capitalize()
    return [f"{pretty(a)}-{pretty(b)}" for a, b in pairs]

bipolar_pairs = [(normalize_channel(a), normalize_channel(b)) for a, b in bipolar_pairs]

name_to_pair = {pair_name(p): p for p in bipolar_pairs}

reordered_pairs = [name_to_pair[name.upper()] for name in desired_order]

anode = [a for a, _ in reordered_pairs]
cathode = [b for _, b in reordered_pairs]
ch_names = make_ch_names(reordered_pairs)

for name, pair in zip(desired_order, reordered_pairs):
    print(f"{name}: {pair}")

print("\nAnode:", anode)
print("Cathode:", cathode)
print("Channel Names:", ch_names)

# load the dataset
def getArray(filename: str):
    raw = mne.io.read_raw_edf(filename, preload=True)
    raw.rename_channels(lambda ch: ch.upper())
    drop_candidates = ['ECG EKG', 'RESP EFFORT','ECG EKG-REF','RESP EFFORT-REF']
    available = set(raw.ch_names)
    to_drop = [ch for ch in drop_candidates if ch in available]
    if to_drop:
        raw.drop_channels(to_drop)
    raw = mne.set_bipolar_reference(raw, anode=anode, cathode=cathode, ch_name=ch_names, copy=True)
    array = raw.get_data()
    # raw.plot()
    return array

eeg = getArray("C:/bt23ece064/dataset/eeg1.edf")


# load the annotations
annotations = pd.read_excel('C:/bt23ece064/dataset/seizure_per_channel_Dec_2017.xlsx')
annotations = annotations.sort_values(by=['FILE ID','Start'], ascending=[True,True])
annotations = annotations.dropna(axis=0)

# we overwrite them for proper length
# seizured_segments = [[np.zeros_like(channel, dtype=float) for channel in eeg]]
# non_seizured_segments = [[np.zeros_like(channel, dtype=float) for channel in eeg]]
unordered_seizured_segments = []
unordered_non_seizured_segments = []


channel_cols = ['Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp1-F3', 'F3-C3', 'C3-P3',
                'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F7', 'F7-T3', 'T3-T5', 
                'T5-O1', 'Fz-Cz', 'Cz-Pz']
print(f"Starting processing {len(annotations)} annotations...")

for i in range(len(annotations)):
    patient = int(annotations['FILE ID'].iloc[i] - 1)
    start = int(annotations['Start'].iloc[i] * 256) 
    stop = int(annotations['Stop'].iloc[i] * 256)
    
    
    print(f"Processing annotation {i+1}/{len(annotations)} - Patient {patient+1}, Time: {annotations['Start'].iloc[i]:.1f}s-{annotations['Stop'].iloc[i]:.1f}s")
    
    seizure_count = 0
    non_seizure_count = 0
    
    for ch_idx, col in enumerate(channel_cols):
        if annotations[col].iloc[i] == 1: 
                # seizured_segments[patient][ch_idx][start:stop] = eeg_set[patient][ch_idx][start:stop]
                # non_seizured_segments[patient][ch_idx][start:stop] = 0
            unordered_seizured_segments.append(eeg[ch_idx][start:stop])
            seizure_count += 1
        else:  
                # seizured_segments[patient][ch_idx][start:stop] = 0
                # non_seizured_segments[patient][ch_idx][start:stop] = eeg_set[patient][ch_idx][start:stop]
            unordered_non_seizured_segments.append(eeg[ch_idx][start:stop])
            non_seizure_count += 1
    
    print(f"  Annotation {i+1}: {seizure_count} seizure channels, {non_seizure_count} non-seizure channels")

print("\nProcessing complete!")
print(f"Total annotations processed: {len(annotations)}")

def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
