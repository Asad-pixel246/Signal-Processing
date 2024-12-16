import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.ndimage import uniform_filter1d
import io  # Import io for StringIO

# -----------------------------------------
# Title: Heart Rate Calculation from ECG Signals
# -----------------------------------------

# Note: For custom theming (Step 6), create a file: .streamlit/config.toml in your repo:
# Example config.toml content:
# [theme]
# primaryColor = "#6C63FF"
# backgroundColor = "#F5F5F5"
# secondaryBackgroundColor = "#FFFFFF"
# textColor = "#333333"
# font = "sans serif"

st.set_page_config(page_title="Heart Rate Calculation", page_icon="❤️")

st.title("Heart Rate Calculation from ECG Signals")

st.write("This application processes one or more ECG signals to detect R-peaks, calculate heart rate, and perform HRV analysis.")
st.write("By: Muhammad Yousaf")
st.write("By: Ali Haider Gillani")
st.write("By: Syed Muhammad Asad")

# Step 7: Multiple File Upload
uploaded_files = st.file_uploader("Upload your ECG data CSV file(s)", type="csv", accept_multiple_files=True)

if uploaded_files:
    # If multiple files are uploaded, allow selection of one file to process
    file_names = [f.name for f in uploaded_files]
    selected_file_name = st.selectbox("Select a file to process:", file_names)
    selected_file = None
    for f in uploaded_files:
        if f.name == selected_file_name:
            selected_file = f
            break

    # Step 2: Interactive Controls (Sidebar)
    st.sidebar.title("Processing Parameters")
    fs = st.sidebar.slider("Sampling Frequency (Hz)", 100, 1000, 360)
    lowcut = st.sidebar.slider("Low Cutoff Frequency (Hz)", 0.1, 5.0, 0.5)
    highcut = st.sidebar.slider("High Cutoff Frequency (Hz)", 5.0, 100.0, 50.0)
    distance_ms = st.sidebar.slider("Min Distance Between R-Peaks (ms)", 200, 1000, 500)
    height_factor = st.sidebar.slider("R-Peak Height Threshold Factor", 0.1, 1.0, 0.6)

    # Step 1: Tabs for Better Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Preprocessing", "R-Peak Detection", "HRV Analysis"])

    with tab1:
        # Load the selected data
        data = pd.read_csv(selected_file)
        data.columns = data.columns.str.strip()

        # Check columns
        if 'sample #' not in data.columns or 'MLII' not in data.columns:
            st.error("Input CSV must contain 'sample #' and 'MLII' columns.")
            st.stop()

        st.write("Data Preview:")
        st.write(data.head())
        
        # Use StringIO to capture data.info() output
        buffer = io.StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

    with tab2:
        # Step 2 & Filtering
        def bandpass_filter(signal, lowcut_freq, highcut_freq, fs_rate, order=4):
            nyquist = 0.5 * fs_rate
            low = lowcut_freq / nyquist
            high = highcut_freq / nyquist
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, signal)

        # Preprocessing
        filtered_signal = bandpass_filter(data['MLII'].values, lowcut, highcut, fs)
        # Remove the first second of data
        filtered_signal = filtered_signal[fs:]
        sample_numbers = data['sample #'].values[fs:]
        # Remove DC offset
        filtered_signal = filtered_signal - np.mean(filtered_signal)

        st.write("### Filtered Signal (First 1000 samples)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['sample #'][:1000], data['MLII'][:1000], label='Original')
        test_filtered = bandpass_filter(data['MLII'].values, lowcut, highcut, fs) 
        test_filtered = test_filtered - np.mean(test_filtered)
        ax.plot(data['sample #'][:1000], test_filtered[:1000], label='Filtered', alpha=0.7)
        ax.legend()
        ax.set_title('ECG Signal Preprocessing')
        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Amplitude')
        st.pyplot(fig)

    with tab3:
        # R-Peak Detection
        def detect_r_peaks(signal, distance_samples, height):
            peaks, _ = find_peaks(signal, distance=distance_samples, height=height)
            return peaks

        dynamic_height = uniform_filter1d(filtered_signal, size=int(0.2 * fs))
        height_threshold = height_factor * np.max(dynamic_height)
        distance_samples = int((distance_ms/1000) * fs)

        r_peaks = detect_r_peaks(filtered_signal, distance_samples, height_threshold)

        st.write("### R-Peaks Detected")
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

    with tab4:
        # Heart Rate Calculation
        def calculate_heart_rate(peaks, fs_rate):
            rr_intervals = np.diff(peaks) / fs_rate
            bpm = 60 / rr_intervals
            return bpm, rr_intervals

        if len(r_peaks) < 2:
            st.warning("Not enough R-peaks detected to calculate heart rate.")
            st.stop()

        bpm_all, rr_intervals = calculate_heart_rate(r_peaks, fs)

        # Remove outliers
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

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(valid_time, bpm, label='Heart Rate (BPM)')
        ax.set_title('Refined Heart Rate Over Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heart Rate (BPM)')
        ax.legend()
        st.pyplot(fig)

        # Time-Domain HRV Metrics
        nn_intervals = valid_rr_intervals
        mean_nn = np.mean(nn_intervals)
        sdnn = np.std(nn_intervals, ddof=1)
        diff_nn = np.diff(nn_intervals)
        rmssd = np.sqrt(np.mean(diff_nn**2))
        nn50 = np.sum(np.abs(diff_nn) > 0.05)
        pnn50 = (nn50 / len(diff_nn)) * 100

        st.write("**Time-Domain HRV Metrics:**")
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

        st.write("**Frequency-Domain HRV Metrics:**")
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
    st.write("Please upload one or more CSV files to proceed.")