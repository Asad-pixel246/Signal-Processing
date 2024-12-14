import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.ndimage import uniform_filter1d

st.title("Heart Rate Calculation from ECG Signals")

st.write("This application processes an ECG signal to detect R-peaks and calculate heart rate, along with HRV metrics.")

# File upload
uploaded_file = st.file_uploader("Upload your ECG data CSV file", type="csv")

if uploaded_file is not None:
    # Step 1: Load Data
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()

    st.write("Data Preview:")
    st.write(data.head())
    st.write(data.info())

    # Verify required columns
    if 'sample #' not in data.columns or 'MLII' not in data.columns:
        st.error("Input CSV must contain 'sample #' and 'MLII' columns.")
        st.stop()

    # Step 2: Preprocessing - Bandpass Filtering
    def bandpass_filter(signal, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    fs = 360  # Sampling frequency (modify if needed)

    filtered_signal = bandpass_filter(data['MLII'].values, 0.5, 50, fs)
    # Remove the first second of data
    filtered_signal = filtered_signal[fs:]
    sample_numbers = data['sample #'].values[fs:]
    filtered_signal = filtered_signal - np.mean(filtered_signal)

    # Plot original vs filtered (first 1000 samples)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data['sample #'][:1000], data['MLII'][:1000], label='Original')
    ax.plot(data['sample #'][:1000], 
            (bandpass_filter(data['MLII'].values, 0.5, 50, fs) - np.mean(bandpass_filter(data['MLII'].values, 0.5, 50, fs)))[:1000], 
            label='Filtered', alpha=0.7)
    ax.legend()
    ax.set_title('ECG Signal Preprocessing')
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)

    # Step 3: R-Peak Detection
    def detect_r_peaks(signal, distance, height):
        peaks, _ = find_peaks(signal, distance=distance, height=height)
        return peaks

    dynamic_height = uniform_filter1d(filtered_signal, size=int(0.2 * fs))
    height = 0.6 * np.max(dynamic_height)
    distance = int(0.5 * fs)

    r_peaks = detect_r_peaks(filtered_signal, distance, height)

    # Plot R-peaks on a portion of the signal
    plot_range = 2000
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sample_numbers[:plot_range], filtered_signal[:plot_range], label='Filtered Signal')
    if len(r_peaks) > 0:
        peaks_to_plot = r_peaks[r_peaks < plot_range]
        ax.plot(sample_numbers[peaks_to_plot], filtered_signal[peaks_to_plot], 'ro', label='Detected R-Peaks')
    ax.legend()
    ax.set_title('R-Peak Detection')
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)

    # Step 4: Heart Rate Calculation
    def calculate_heart_rate(peaks, fs):
        rr_intervals = np.diff(peaks) / fs
        bpm = 60 / rr_intervals
        return bpm, rr_intervals

    if len(r_peaks) < 2:
        st.warning("Not enough R-peaks detected to calculate heart rate.")
        st.stop()

    bpm_all, rr_intervals = calculate_heart_rate(r_peaks, fs)

    # Remove outliers using IQR
    q1, q3 = np.percentile(rr_intervals, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    valid_mask = (rr_intervals > lower_bound) & (rr_intervals < upper_bound)
    valid_rr_intervals = rr_intervals[valid_mask]

    if len(valid_rr_intervals) == 0:
        st.warning("No valid RR intervals after outlier removal.")
        st.stop()

    valid_time = sample_numbers[r_peaks[1:len(valid_rr_intervals) + 1]] / fs
    bpm = 60 / valid_rr_intervals

    st.write(f"**Mean Heart Rate:** {np.mean(bpm):.2f} BPM")
    st.write(f"**Min Heart Rate:** {np.min(bpm):.2f} BPM")
    st.write(f"**Max Heart Rate:** {np.max(bpm):.2f} BPM")

    # Step 5: Visualization of Heart Rate Over Time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(valid_time, bpm, label='Heart Rate (BPM)')
    ax.set_title('Refined Heart Rate Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heart Rate (BPM)')
    ax.legend()
    st.pyplot(fig)

    # Additional Analysis (HRV)
    nn_intervals = valid_rr_intervals
    mean_nn = np.mean(nn_intervals)
    sdnn = np.std(nn_intervals, ddof=1)
    diff_nn = np.diff(nn_intervals)
    rmssd = np.sqrt(np.mean(diff_nn**2))
    nn50 = np.sum(np.abs(diff_nn) > 0.05)
    pnn50 = (nn50 / len(diff_nn)) * 100

    st.write("\n**Time-Domain HRV Metrics:**")
    st.write(f"Mean NN Interval: {mean_nn:.4f} s")
    st.write(f"SDNN: {sdnn:.4f} s")
    st.write(f"RMSSD: {rmssd:.4f} s")
    st.write(f"NN50: {nn50}")
    st.write(f"pNN50: {pnn50:.2f} %")

    # Frequency-Domain HRV Analysis
    nn_times = np.cumsum(nn_intervals)
    target_fs = 4.0
    interp_times = np.arange(nn_times[0], nn_times[-1], 1/target_fs)
    interp_nn = np.interp(interp_times, nn_times, nn_intervals)

    f, pxx = welch(interp_nn, fs=target_fs, nperseg=256)

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

    st.write("\n**Frequency-Domain HRV Metrics:**")
    st.write(f"VLF Power: {vlf_power:.4f}")
    st.write(f"LF Power: {lf_power:.4f}")
    st.write(f"HF Power: {hf_power:.4f}")
    st.write(f"LF/HF Ratio: {lf_hf_ratio:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(f, pxx)
    ax.set_title('HRV Frequency-Domain Analysis (PSD)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (s^2/Hz)')
    ax.axvspan(*vlf_band, color='yellow', alpha=0.3, label='VLF Band')
    ax.axvspan(*lf_band, color='green', alpha=0.3, label='LF Band')
    ax.axvspan(*hf_band, color='red', alpha=0.3, label='HF Band')
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Please upload a CSV file to proceed.")