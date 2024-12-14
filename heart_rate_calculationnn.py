import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.ndimage import uniform_filter1d

# -----------------------------------------
# Title: Heart Rate Calculation from ECG Signals Using Signal Processing Techniques
# -----------------------------------------

st.title("Heart Rate Calculation from ECG Signals")

# Step 1: Load Data
file_path = r'C:\Users\asadn\Downloads\datanew.csv'  # Update to your actual path
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()  # Remove trailing spaces if any

print("Data Preview:")
print(data.head())
print(data.info())

# Verify that the required columns exist
if 'sample #' not in data.columns or 'MLII' not in data.columns:
    raise ValueError("Input CSV must contain 'sample #' and 'MLII' columns.")

if st.sidebar.button("Process ECG"):
    st.write("Processing ECG signal...")
# Step 2: Preprocessing - Bandpass Filtering
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

fs = 360  # Sampling frequency in Hz (adjust based on your dataset)

# Apply bandpass filter
filtered_signal = bandpass_filter(data['MLII'].values, 0.5, 50, fs)

# Remove the first second of data (fs samples) to avoid filter startup artifacts
filtered_signal = filtered_signal[fs:]
sample_numbers = data['sample #'].values[fs:]

# (Optional) Remove DC offset
filtered_signal = filtered_signal - np.mean(filtered_signal)

# Step 3: R-Peak Detection
def detect_r_peaks(signal, distance, height):
    """Detect peaks in the ECG signal that represent R-peaks."""
    peaks, _ = find_peaks(signal, distance=distance, height=height)
    return peaks

# Dynamic thresholding
dynamic_height = uniform_filter1d(filtered_signal, size=int(0.2 * fs))  # 200 ms smoothing
height = 0.6 * np.max(dynamic_height)  
distance = int(0.5 * fs)  # Min distance between peaks ~ 0.5s

r_peaks = detect_r_peaks(filtered_signal, distance, height)

# Step 4: Heart Rate Calculation
def calculate_heart_rate(peaks, fs):
    """Calculate heart rate (BPM) from R-peak indices."""
    rr_intervals = np.diff(peaks) / fs  # Convert sample differences to seconds
    bpm = 60 / rr_intervals
    return bpm, rr_intervals

if len(r_peaks) < 2:
    print("Not enough R-peaks detected to calculate heart rate.")
    exit()

bpm_all, rr_intervals = calculate_heart_rate(r_peaks, fs)

# Remove outliers using IQR
q1, q3 = np.percentile(rr_intervals, [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
valid_mask = (rr_intervals > lower_bound) & (rr_intervals < upper_bound)
valid_rr_intervals = rr_intervals[valid_mask]

if len(valid_rr_intervals) == 0:
    print("No valid RR intervals after outlier removal.")
    exit()

valid_time = sample_numbers[r_peaks[1:len(valid_rr_intervals) + 1]] / fs
bpm = 60 / valid_rr_intervals

print(f"Mean Heart Rate: {np.mean(bpm):.2f} BPM")
print(f"Min Heart Rate: {np.min(bpm):.2f} BPM")
print(f"Max Heart Rate: {np.max(bpm):.2f} BPM")



st.write("Displaying results...")
# Visualize the first 1000 samples to check filtering
plt.figure(figsize=(10, 4))
plt.plot(data['sample #'][:1000], data['MLII'][:1000], label='Original')
plt.plot(data['sample #'][:1000], 
         (bandpass_filter(data['MLII'].values, 0.5, 50, fs) - np.mean(bandpass_filter(data['MLII'].values, 0.5, 50, fs)))[:1000], 
         label='Filtered', alpha=0.7)
plt.legend()
plt.title('ECG Signal Preprocessing')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Plot detected R-peaks on a portion of the signal
plot_range = 2000
plt.figure(figsize=(10, 4))
plt.plot(sample_numbers[:plot_range], filtered_signal[:plot_range], label='Filtered Signal')
if len(r_peaks) > 0:
    peaks_to_plot = r_peaks[r_peaks < plot_range]
    plt.plot(sample_numbers[peaks_to_plot], filtered_signal[peaks_to_plot], 'ro', label='Detected R-Peaks')
plt.legend()
plt.title('R-Peak Detection')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Debug plot: Overlay detected peaks on entire signals
""" plt.figure(figsize=(10, 4))
plt.plot(sample_numbers, data['MLII'].values[fs:], label='Original Signal')
plt.plot(sample_numbers, filtered_signal, label='Filtered Signal', alpha=0.7)
if len(r_peaks) > 0:
    plt.plot(sample_numbers[r_peaks], filtered_signal[r_peaks], 'ro', label='R-Peaks')
plt.legend()
plt.title('R-Peak Detection Validation')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show() """

# Step 5: Visualization of Heart Rate Over Time
plt.figure(figsize=(10, 4))
plt.plot(valid_time, bpm, label='Heart Rate (BPM)')
plt.title('Refined Heart Rate Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Heart Rate (BPM)')
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Additional Analysis (HRV Time-Domain and Frequency-Domain)
# ------------------------------------------------------------

# Time-Domain HRV Metrics
# NN intervals are the same as RR intervals after removing artifacts.
nn_intervals = valid_rr_intervals  # often called NN intervals in HRV analysis

# Common time-domain metrics:
mean_nn = np.mean(nn_intervals)          # Mean of NN intervals (s)
sdnn = np.std(nn_intervals, ddof=1)      # Standard deviation of NN intervals
diff_nn = np.diff(nn_intervals)
rmssd = np.sqrt(np.mean(diff_nn**2))     # Root mean square of successive differences
nn50 = np.sum(np.abs(diff_nn) > 0.05)    # Count of NN interval differences > 50 ms
pnn50 = (nn50 / len(diff_nn)) * 100      # Percentage of differences > 50 ms

print("\nTime-Domain HRV Metrics:")
print(f"Mean NN Interval: {mean_nn:.4f} s")
print(f"SDNN: {sdnn:.4f} s")
print(f"RMSSD: {rmssd:.4f} s")
print(f"NN50: {nn50}")
print(f"pNN50: {pnn50:.2f} %")

# Frequency-Domain Analysis
# To perform frequency-domain analysis of heart rate variability, we need to:
# 1. Resample the NN intervals to create a uniformly sampled time series.
# 2. Compute power spectral density (PSD) using e.g., Welch's method.

# Interpolate NN intervals at a fixed rate (e.g., 4 Hz) for PSD analysis
# First, create a time vector corresponding to the NN intervals
nn_times = np.cumsum(nn_intervals)
target_fs = 4.0  # 4 Hz is a common choice
# Interpolation times
interp_times = np.arange(nn_times[0], nn_times[-1], 1/target_fs)

# Use linear interpolation
interp_nn = np.interp(interp_times, nn_times, nn_intervals)

# Compute PSD using Welch's method
f, pxx = welch(interp_nn, fs=target_fs, nperseg=256)

# Frequency bands for HRV: 
# VLF: 0.0033-0.04 Hz, LF: 0.04-0.15 Hz, HF: 0.15-0.40 Hz
vlf_band = (0.0033, 0.04)
lf_band = (0.04, 0.15)
hf_band = (0.15, 0.40)

def band_power(frequencies, power, band):
    mask = (frequencies >= band[0]) & (frequencies <= band[1])
    return np.trapz(power[mask], frequencies[mask])

vlf_power = band_power(f, pxx, vlf_band)
lf_power = band_power(f, pxx, lf_band)
hf_power = band_power(f, pxx, hf_band)
lf_hf_ratio = lf_power / hf_power if hf_power != 0 else np.inf

print("\nFrequency-Domain HRV Metrics:")
print(f"VLF Power: {vlf_power:.4f}")
print(f"LF Power: {lf_power:.4f}")
print(f"HF Power: {hf_power:.4f}")
print(f"LF/HF Ratio: {lf_hf_ratio:.4f}")

# Plot PSD
plt.figure(figsize=(10, 4))
plt.semilogy(f, pxx)
plt.title('HRV Frequency-Domain Analysis (PSD)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (s^2/Hz)')
plt.axvspan(*vlf_band, color='yellow', alpha=0.3, label='VLF Band')
plt.axvspan(*lf_band, color='green', alpha=0.3, label='LF Band')
plt.axvspan(*hf_band, color='red', alpha=0.3, label='HF Band')
plt.legend()
plt.tight_layout()
plt.show()