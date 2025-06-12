import json
import glob
import os
import numpy as np
import pandas as pd
import mne
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, correlate
from scipy.stats import skew, kurtosis
from tabulate import tabulate
import pywt
from sklearn.linear_model import LinearRegression

imports = """import numpy as np
import pandas as pd
import mne
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, correlate
from scipy.stats import skew, kurtosis
from tabulate import tabulate
import pywt"""

# Global variables cell
global_vars_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["""bipolar_pairs = [
    ('EEG Fp1-REF', 'EEG F7-REF'),
    ('EEG F7-REF', 'EEG T3-REF'),
    ('EEG T3-REF', 'EEG T5-REF'),
    ('EEG T5-REF', 'EEG O1-REF'),
    ('EEG Fp1-REF', 'EEG F3-REF'),
    ('EEG F3-REF', 'EEG C3-REF'),
    ('EEG C3-REF', 'EEG P3-REF'),
    ('EEG P3-REF', 'EEG O1-REF'),
    ('EEG Fz-REF', 'EEG Cz-REF'),
    ('EEG Cz-REF', 'EEG Pz-REF'),
    ('EEG Fp2-REF', 'EEG F4-REF'),
    ('EEG F4-REF', 'EEG C4-REF'),
    ('EEG C4-REF', 'EEG P4-REF'),
    ('EEG P4-REF', 'EEG O2-REF'),
    ('EEG Fp2-REF', 'EEG F8-REF'),
    ('EEG F8-REF', 'EEG T4-REF'),
    ('EEG T4-REF', 'EEG T6-REF'),
    ('EEG T6-REF', 'EEG O2-REF'),
]

anode = [pair[0] for pair in bipolar_pairs]
cathode = [pair[1] for pair in bipolar_pairs]
ch_names = [f"{pair[0].replace('EEG ', '').replace('-REF', '')}-{pair[1].replace('EEG ', '').replace('-REF', '')}" 
           for pair in bipolar_pairs]"""]
}

# Function definition cell
function_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["""def getArray(filename :str):
    channels_to_drop = ['ECG EKG-REF', 'Resp Effort-REF']
    raw = mne.io.read_raw_edf(filename, preload=True)
    raw.drop_channels(channels_to_drop)
    
    raw = mne.set_bipolar_reference(raw, anode=anode, cathode=cathode, copy=True, ch_name=ch_names)
    raw.plot()
    array = raw.get_data()
    actual_ch_names = raw.ch_names
    return array, actual_ch_names"""]
}

# Annotations cell
annotations_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["""a = pd.read_csv('/Users/adityakinjawadekar/Documents/100xcode/py/annotations_2017_A_fixed.csv')
b = pd.read_csv('/Users/adityakinjawadekar/Documents/100xcode/py/annotations_2017_B.csv')
c = pd.read_csv('/Users/adityakinjawadekar/Documents/100xcode/py/annotations_2017_C.csv')"""]
}

# Interval calculation cell
interval_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["""sz = []
nsz = []
for i in range(len(array[0])):
    if a['1'].iloc[i//256] == 1:
        sz.append(i)
    else:
        nsz.append(i)
        
def calculate_intervals(numbers):
    numbers = sorted(numbers)
    intervals = []
    start = end = numbers[0]

    for num in numbers[1:]:
        if num == end + 1:
            end = num
        else:
            intervals.append([start, end])
            start = end = num

    intervals.append([start, end])  
    return intervals

int = calculate_intervals(sz)
nsi = calculate_intervals(nsz)

# Extract seizure and non-seizure segments for all channels
seizured_segments = []
non_seizured_segments = []

for j in range(len(array)):  # For each channel
    channel_seizure = []
    channel_non_seizure = []
    
    # Process seizure segments
    for interval in int:
        start, end = interval
        if start < len(array[j]) and end < len(array[j]):
            channel_seizure.append(array[j][start:end])
    
    # Process non-seizure segments
    for interval in nsi:
        start, end = interval
        if start < len(array[j]) and end < len(array[j]):
            channel_non_seizure.append(array[j][start:end])
    
    seizured_segments.append(channel_seizure)
    non_seizured_segments.append(channel_non_seizure)"""]
}

# Visualization cell
visualization_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["""n_channels = array.shape[0]
n_timepoints = array.shape[1]

# Initialize arrays for all channels
seizured_all = np.zeros_like(array)
non_seizured_all = np.zeros_like(array)

# Process each channel
for ch in range(n_channels):
    for i in range(n_timepoints):
        if a['1'].iloc[i//256] == 1:  # Seizure
            seizured_all[ch][i] = array[ch][i]
            non_seizured_all[ch][i] = 0
        elif a['1'].iloc[i//256] == 0:  # Non-seizure
            seizured_all[ch][i] = 0
            non_seizured_all[ch][i] = array[ch][i]

# Create the plot
fig, axes = plt.subplots(n_channels, 3, figsize=(18, 3*n_channels))
fig.suptitle(f'EEG {eeg_number}: Original, Seizure, and Non-Seizure Segments', fontsize=16)

# Plot each channel
for ch in range(n_channels):
    # Original signal
    axes[ch, 0].plot(array[ch], 'k-', linewidth=0.5)
    axes[ch, 0].set_title(f'{actual_ch_names[ch]} - Full Signal')
    axes[ch, 0].set_ylabel('Amplitude (µV)')
    
    # Seizure segments
    axes[ch, 1].plot(seizured_all[ch], 'r-', linewidth=0.5)
    axes[ch, 1].set_title(f'{actual_ch_names[ch]} - Seizure Segments')
    axes[ch, 1].set_ylabel('Amplitude (µV)')
    
    # Non-seizure segments  
    axes[ch, 2].plot(non_seizured_all[ch], 'orange', linewidth=0.5)
    axes[ch, 2].set_title(f'{actual_ch_names[ch]} - Non-Seizure Segments')
    axes[ch, 2].set_ylabel('Amplitude (µV)')
    
    # Add x-axis label only to bottom row
    if ch == n_channels - 1:
        axes[ch, 0].set_xlabel('Time (samples)')
        axes[ch, 1].set_xlabel('Time (samples)')
        axes[ch, 2].set_xlabel('Time (samples)')

plt.tight_layout()
plt.savefig(f'plots/eeg{eeg_number}/signal_segments_eeg{eeg_number}.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Print some statistics
print(f"Total channels: {n_channels}")
print(f"Total timepoints: {n_timepoints}")
print(f"Seizure samples: {np.sum(a['1'] == 1) * 256}")
print(f"Non-seizure samples: {np.sum(a['1'] == 0) * 256}")"""]
}

# Autocorrelation cell
autocorr_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["""# Create autocorrelation plots for all channels
for channel in range(len(array)):
    fig, axs = plt.subplots(5, 5, figsize=(15, 10))  # 5 rows x 5 columns of subplots
    axs = axs.flatten()  # Flatten to 1D array for easy indexing
    
    for idx, i in enumerate(seizured_segments[channel]):
        autocorr = correlate(i, i, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        axs[idx].plot(autocorr)  # Plot first 500 lags
        axs[idx].set_title(f"Segment {idx+1}", fontsize=8)
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
    
    plt.suptitle(f"Autocorrelation of Seizure Segments (Channel {channel+1}: {actual_ch_names[channel]})", fontsize=16)
    plt.tight_layout()
    plt.show()"""]
}

# Power Spectral Density cell
psd_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["""# Create Power Spectral Density plots for all channels
fs = 256  # Sampling frequency in Hz

for channel in range(len(array)):
    fig, axs = plt.subplots(5, 5, figsize=(15, 10))
    axs = axs.flatten()
    
    for idx, i in enumerate(seizured_segments[channel]):
        f, Pxx = welch(i, fs=fs, nperseg=1024)
        axs[idx].semilogy(f, Pxx)  # log scale on y-axis
        axs[idx].set_title(f"Segment {idx+1}", fontsize=8)
        axs[idx].set_xlim(0, 128)  # Show up to 128 Hz
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
    
    plt.suptitle(f"Power Spectral Density of Seizure Segments (Channel {channel+1}: {actual_ch_names[channel]})", fontsize=16)
    plt.tight_layout()
    plt.show()"""]
}

# Continuous Wavelet Transform cell
cwt_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["""wavelet = 'cmor'
scales = np.arange(1, 64)
sampling_rate = 256
sampling_period = 1 / sampling_rate

frequencies = pywt.scale2frequency(wavelet, scales) / sampling_period

for channel in range(len(array)):
    fig, axs = plt.subplots(5, 5, figsize=(20, 15))
    axs = axs.flatten()
    
    for i, segment in enumerate(seizured_segments[channel]):
        coefficients, _ = pywt.cwt(segment, scales, wavelet, sampling_period)
        
        ax = axs[i]
        t = np.arange(len(segment)) / sampling_rate
        im = ax.imshow(np.abs(coefficients), extent=[t[0], t[-1], frequencies[-1], frequencies[0]],
                      cmap='magma', aspect='auto', vmax=np.percentile(np.abs(coefficients), 99))
        ax.set_title(f'Segment {i+1}', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
    
    fig.subplots_adjust(right=0.9, hspace=0.4, wspace=0.3)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Magnitude')
    
    plt.suptitle(f'Continuous Wavelet Transform (Channel {channel+1}: {actual_ch_names[channel]})', fontsize=16)
    plt.savefig(f'plots/eeg{eeg_number}/cwt_channel_{channel+1}_eeg{eeg_number}.png', dpi=300, bbox_inches='tight')
    plt.show()"""]
}

# Statistics cell
stats_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["""import numpy as np
from tabulate import tabulate
from scipy.signal import welch

time_stats = []
psd_areas = []

fs = 256

for ch in range(len(array)):
    for seg_idx, segment in enumerate(seizured_segments[ch]):
        mean_val = np.mean(segment)
        var_val = np.var(segment)
        std_val = np.std(segment)
        
        time_stats.append([actual_ch_names[ch], seg_idx + 1, mean_val, var_val, std_val])
        
        f, Pxx = welch(segment, fs=fs, nperseg=1024)
        area = np.trapz(Pxx, f)
        psd_areas.append([actual_ch_names[ch], seg_idx + 1, area])

time_stats = np.array(time_stats)
psd_areas = np.array(psd_areas)

time_headers = ['Channel', 'Segment', 'Mean', 'Variance', 'Std Dev']
psd_headers = ['Channel', 'Segment', 'PSD Area']

print("Time Domain Statistics for Seizured Segments:")
print(tabulate(time_stats, headers=time_headers, tablefmt='grid', floatfmt='.4f'))
print("\\n")

print("Power Spectral Density Areas for Seizured Segments:")
print(tabulate(psd_areas, headers=psd_headers, tablefmt='grid', floatfmt='.4f'))

np.savetxt(f'stats/eeg{eeg_number}/time_stats_seizured_eeg{eeg_number}.csv', 
           time_stats, delimiter=',', header=','.join(time_headers), comments='', fmt='%s')
np.savetxt(f'stats/eeg{eeg_number}/psd_areas_seizured_eeg{eeg_number}.csv', 
           psd_areas, delimiter=',', header=','.join(psd_headers), comments='', fmt='%s')"""]
}

# PSD log-fit cell
psd_logfit_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["""from sklearn.linear_model import LinearRegression

psd_fit_results = []

fs = 256

# Create a figure for plotting PSD and linear fits
plt.figure(figsize=(15, 10))
plt.suptitle('Log PSD and Linear Fit for Seizure Segments', fontsize=16)

for ch in range(len(array)):
    for seg_idx, segment in enumerate(seizured_segments[ch]):
        f, Pxx = welch(segment, fs=fs, nperseg=1024)
        log_Pxx = np.log(Pxx + 1e-12)
        f_reshaped = f.reshape(-1, 1)
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(f_reshaped, log_Pxx)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Calculate fitted line
        fitted_line = model.predict(f_reshaped)
        
        # Plot log PSD and fitted line
        plt.subplot(5, 5, seg_idx + 1)
        plt.plot(f, log_Pxx, 'b-', label='Log PSD', alpha=0.5)
        plt.plot(f, fitted_line, 'r-', label='Linear Fit')
        plt.title(f'Ch {ch+1} Seg {seg_idx+1}\\nSlope: {slope:.2f}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('log(PSD)')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        psd_fit_results.append([actual_ch_names[ch], seg_idx + 1, slope, intercept])

plt.tight_layout()
plt.savefig(f'plots/eeg{eeg_number}/psd_linear_fit_eeg{eeg_number}.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a figure for plotting slopes and intercepts
plt.figure(figsize=(15, 10))
plt.suptitle('Slopes and Intercepts for Seizure Segments', fontsize=16)

# Plot slopes
plt.subplot(2, 1, 1)
slopes = [result[2] for result in psd_fit_results]
plt.plot(slopes, 'bo-')
plt.title('Slopes')
plt.xlabel('Segment Index')
plt.ylabel('Slope')
plt.grid(True, alpha=0.3)

# Plot intercepts
plt.subplot(2, 1, 2)
intercepts = [result[3] for result in psd_fit_results]
plt.plot(intercepts, 'ro-')
plt.title('Intercepts')
plt.xlabel('Segment Index')
plt.ylabel('Intercept')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'plots/eeg{eeg_number}/slopes_intercepts_eeg{eeg_number}.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results to CSV
psd_fit_results = np.array(psd_fit_results)
psd_fit_headers = ['Channel', 'Segment', 'Slope', 'Intercept']
np.savetxt(f'stats/eeg{eeg_number}/psd_logfit_seizured_eeg{eeg_number}.csv', 
           psd_fit_results, delimiter=',', header=','.join(psd_fit_headers), comments='', fmt='%s')"""]
}

# Get all eeg*.ipynb files in the processing directory
notebook_files = glob.glob("processing/eeg*.ipynb")

# Update each notebook
for notebook_file in notebook_files:
    notebook_number = os.path.basename(notebook_file).replace('eeg', '').replace('.ipynb', '')

    # Directory cell with the actual number
    dir_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"""eeg_number = '{notebook_number}'
export_dirs = {{
    'plots': f'plots/eeg{notebook_number}',
    'stats': f'stats/eeg{notebook_number}',
    'data': f'data/eeg{notebook_number}'
}}
for dir_path in export_dirs.values():
    os.makedirs(dir_path, exist_ok=True)"""]
    }

    current_array_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"array, actual_ch_names = getArray('../dataset/eeg{notebook_number}.edf')"]
    }
    
    # Create the notebook structure with all cells
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [imports]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["!pip install scikit-learn"]
            },
            dir_cell,
            global_vars_cell,
            function_cell,
            current_array_cell,
            annotations_cell,
            interval_cell,
            visualization_cell,
            stats_cell,
            psd_logfit_cell,
            autocorr_cell,
            psd_cell,
            cwt_cell,
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["!pip install scikit-learn"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(notebook_file, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Updated {notebook_file}")

print("All notebooks have been updated with the required imports, functions, and data loading code.")
