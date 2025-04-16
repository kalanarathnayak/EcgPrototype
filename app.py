import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from torch.utils.data import DataLoader
import wfdb
from wfdb import processing
from sklearn import preprocessing
import scipy.signal
import tempfile
import os

# Imports the model architecture
from model import TransformerECG, ECGDataset

# Sets the page config
st.set_page_config(page_title="ULTRA-ECG 🫀", layout="wide")

# Constants
WINDOW_SIZE = 180
FS = 360  # Sampling frequency


def load_model():
    """Loads the trained model"""
    model = TransformerECG(
        input_size=180,
        d_model=256,
        nhead=16,
        num_transformer_layers=3,
        dropout=0.15
    )
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_signal(signal):
    """Preprocess the ECG signal"""
    # Calculate median filter window sizes
    window_200ms = int(0.2 * FS)
    if window_200ms % 2 == 0:
        window_200ms += 1

    window_600ms = int(0.6 * FS)
    if window_600ms % 2 == 0:
        window_600ms += 1

    # Removes baseline wander using two median filters
    filt1 = scipy.signal.medfilt(signal, window_200ms)
    filt2 = scipy.signal.medfilt(filt1, window_600ms)
    signal = signal - filt2

    # Scales the data
    signal = preprocessing.scale(np.nan_to_num(signal))

    return signal


def detect_peaks(signal):
    """Detects R-peaks in the ECG signal"""
    qrs = processing.XQRS(sig=signal, fs=FS)
    qrs.detect()
    return qrs.qrs_inds


def extract_beats(signal, peaks):
    """Extracts the beats around R-peaks"""
    beats = []
    valid_peaks = []

    for peak in peaks:
        start = peak - WINDOW_SIZE // 2
        end = peak + WINDOW_SIZE // 2

        if start >= 0 and end < len(signal):
            beats.append(signal[start:end])
            valid_peaks.append(peak)

    return np.array(beats), valid_peaks


def get_anomaly_scores(model, beats):
    """Gets the anomaly scores for beats"""
    dataset = ECGDataset(beats)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    reconstruction_errors = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            error = torch.mean(torch.abs(output - batch), dim=1)
            reconstruction_errors.extend(error.numpy())

    return np.array(reconstruction_errors)


def plot_ecg_overview(signal, peaks, scores, threshold, current_start, segment_duration, fs=360):
    """An overview plot of the entire ECG with anomaly detection and segment indicator"""
    fig = go.Figure()

    # Time axis for full signal
    time_axis = np.arange(len(signal)) / fs

    # Plots the full ECG signal
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=signal,
        mode='lines',
        name='ECG Signal',
        line=dict(color='blue', width=1)
    ))

    # Splits the peaks into normal and anomalous
    normal_peaks = [peaks[i] for i in range(len(peaks)) if scores[i] <= threshold]
    anomaly_peaks = [peaks[i] for i in range(len(peaks)) if scores[i] > threshold]

    # Plots normal beats (smaller markers for overview)
    if normal_peaks:
        fig.add_trace(go.Scatter(
            x=np.array(normal_peaks) / fs,
            y=signal[normal_peaks],
            mode='markers',
            name='Normal Beats',
            marker=dict(color='green', size=4, symbol='circle')
        ))

    # Plots anomalous beats (smaller markers for overview)
    if anomaly_peaks:
        fig.add_trace(go.Scatter(
            x=np.array(anomaly_peaks) / fs,
            y=signal[anomaly_peaks],
            mode='markers',
            name='Anomalous Beats',
            marker=dict(color='red', size=4, symbol='circle')
        ))

    # Adds highlighted region for selected segment
    fig.add_vrect(
        x0=current_start,
        x1=current_start + segment_duration,
        fillcolor="rgba(128, 128, 128, 0.2)",
        layer="below",
        line_width=0,
        annotation_text="Viewing Window",
        annotation_position="top left"
    )

    fig.update_layout(
        title='ECG Overview with Anomaly Detection',
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        height=200,
        margin=dict(t=30, b=20, l=50, r=50),
        showlegend=True,
    )

    return fig


def plot_ecg_segment(signal, peaks, scores, threshold, start_time=0, fs=360):
    """Create a detailed plot of the selected ECG segment with annotations and grid"""
    fig = go.Figure()

    # Create time axis in seconds
    time_axis = np.arange(len(signal)) / fs + start_time

    # Plot the ECG signal
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=signal,
        mode='lines',
        name='ECG Signal',
        line=dict(color='blue', width=1.5)
    ))

    # Split peaks into normal and anomalous
    normal_peaks = [peaks[i] for i in range(len(peaks)) if scores[i] <= threshold]
    anomaly_peaks = [peaks[i] for i in range(len(peaks)) if scores[i] > threshold]

    # Plot normal beats
    if normal_peaks:
        fig.add_trace(go.Scatter(
            x=np.array(normal_peaks) / fs + start_time,
            y=signal[normal_peaks],
            mode='markers',
            name='Normal Beats',
            marker=dict(color='green', size=8, symbol='circle')
        ))

    # Plot anomalous beats
    if anomaly_peaks:
        fig.add_trace(go.Scatter(
            x=np.array(anomaly_peaks) / fs + start_time,
            y=signal[anomaly_peaks],
            mode='markers',
            name='Anomalous Beats',
            marker=dict(color='red', size=8, symbol='circle')
        ))

    # Update layout with grid
    fig.update_layout(
        title='Selected ECG Segment with Anomaly Detection',
        xaxis=dict(
            title='Time (seconds)',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128, 128, 128, 0.2)',
            minor=dict(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(128, 128, 128, 0.1)',
                dtick=0.2
            )
        ),
        yaxis=dict(
            title='Amplitude',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128, 128, 128, 0.2)',
            minor=dict(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(128, 128, 128, 0.1)',
                dtick=0.5
            )
        ),
        plot_bgcolor='white',
        height=400,
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Add minor grid lines
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor='rgba(128, 128, 128, 0.2)',
        mirror=True
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor='rgba(128, 128, 128, 0.2)',
        mirror=True
    )

    return fig

def app_footer():
    footer = """
    <div style="position: fixed; bottom: 0; width: 100%; background-color: #f0f2f6; padding: 10px; text-align: center; right: 0px">
        <p style="margin: 0; color: #666666; font-size: 14px;">
            ULTRA-ECG v2.10 | 
            Developed using PyTorch and Streamlit | 
            Model: Transformer-based Autoencoder  |
            Built with 🫀 by Kalana Ratnayake (w1903043/20211322)
        </p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

def main():
    st.title("ULTRA-ECG 🫀")
    st.subheader("Unsupervised Transformer based model for optimized ECG anomaly detection")
    st.write("Upload an ECG recording to detect anomalous beats")

    # Initializes the session state variables if they don't exist
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    def reset_analysis():
        """Resets the analysis state when files change"""
        st.session_state.processed_data = None
        st.session_state.analysis_done = False

    # File uploaders for both .dat and .hea files
    st.write(":red[Please upload both .dat and .hea files for your ECG record (files must have the same name):]")
    col1, col2 = st.columns(2)

    with col1:
        dat_file = st.file_uploader("Upload .dat file",
                                    type=['dat'],
                                    on_change=reset_analysis,
                                    disabled=st.session_state.analysis_done)
    with col2:
        hea_file = st.file_uploader("Upload .hea file",
                                    type=['hea'],
                                    on_change=reset_analysis,
                                    disabled=st.session_state.analysis_done)

    # Validates the file names
    files_valid = False
    if dat_file and hea_file:
        dat_name = dat_file.name.replace('.dat', '')
        hea_name = hea_file.name.replace('.hea', '')

        if dat_name != hea_name:
            st.error("The .dat and .hea files must have the same base name.")
        else:
            files_valid = True

    analyze_button = st.button("Analyze ECG",
                               disabled=not (dat_file and hea_file and files_valid) or
                                        st.session_state.analysis_done)

    if (dat_file and hea_file and files_valid) and \
            (analyze_button or st.session_state.analysis_done):
        try:
            # Creates a temporary directory to store the files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Gets the base name without extension
                base_name = dat_file.name.replace('.dat', '')

                # Saves the uploaded files to temporary directory
                dat_path = os.path.join(temp_dir, dat_file.name)
                hea_path = os.path.join(temp_dir, hea_file.name)

                with open(dat_path, 'wb') as f:
                    f.write(dat_file.getvalue())
                with open(hea_path, 'wb') as f:
                    f.write(hea_file.getvalue())

                # Reads the record using wfdb
                record = wfdb.rdrecord(os.path.join(temp_dir, base_name), smooth_frames=True)
                signal = record.p_signal[:, 0]  # Get lead II

                # Loads model
                model = load_model()

                # Processes the entire signal and store results in session state
                if st.session_state.processed_data is None:
                    with st.spinner("Processing entire ECG signal..."):
                        processed_signal = preprocess_signal(signal)
                        peaks = detect_peaks(processed_signal)
                        beats, valid_peaks = extract_beats(processed_signal, peaks)
                        scores = get_anomaly_scores(model, beats)
                        threshold = np.percentile(scores, 87)

                        # Stores all processed data in session state
                        st.session_state.processed_data = {
                            'processed_signal': processed_signal,
                            'valid_peaks': valid_peaks,
                            'scores': scores,
                            'threshold': threshold
                        }
                        st.session_state.analysis_done = True

                # Retrieves the processed data from session state
                processed_signal = st.session_state.processed_data['processed_signal']
                valid_peaks = st.session_state.processed_data['valid_peaks']
                scores = st.session_state.processed_data['scores']
                threshold = st.session_state.processed_data['threshold']

                # Adds segment selection for viewing
                st.subheader("Select Viewing Window")
                total_duration = len(signal) / FS  # Converts samples to seconds

                col1, col2 = st.columns(2)
                with col1:
                    segment_duration = st.slider(
                        "Window length (seconds)",
                        min_value=10,
                        max_value=int(total_duration),
                        value=min(30, int(total_duration)),
                        step=5
                    )

                with col2:
                    start_time = st.slider(
                        "Start time (seconds)",
                        min_value=0,
                        max_value=int(total_duration - segment_duration),
                        value=0,
                        step=5
                    )

                # Plots the overview of entire ECG with anomaly detection
                overview_fig = plot_ecg_overview(processed_signal, valid_peaks, scores, threshold,
                                                 start_time, segment_duration, fs=FS)
                st.plotly_chart(overview_fig, use_container_width=True, config={'displayModeBar': False})

                # Extracts the segment for detailed viewing
                start_sample = int(start_time * FS)
                segment_length = int(segment_duration * FS)
                end_sample = start_sample + segment_length

                # Gets the segment indices for peaks and scores
                segment_indices = [i for i, peak in enumerate(valid_peaks)
                                   if start_sample <= peak < end_sample]
                segment_peaks = [peak - start_sample for peak in valid_peaks
                                 if start_sample <= peak < end_sample]
                segment_scores = [scores[i] for i in segment_indices]

                # Plots the selected segment with annotations
                segment_fig = plot_ecg_segment(processed_signal[start_sample:end_sample],
                                               segment_peaks, segment_scores, threshold,
                                               start_time=start_time, fs=FS)
                st.plotly_chart(segment_fig, use_container_width=True)

                # Displays the statistics for the entire recording
                st.subheader("Detection Results (Entire Recording)")
                col1, col2, col3 = st.columns(3)

                total_beats = len(scores)
                anomalous_beats = sum(scores > threshold)
                normal_beats = total_beats - anomalous_beats

                col1.metric("Total Beats", total_beats)
                col2.metric("Normal Beats", normal_beats)
                col3.metric("Anomalous Beats", anomalous_beats)

                # Displays the confidence scores
                st.subheader("Confidence Scores")
                scores_df = pd.DataFrame({
                    'Beat Number': range(1, len(scores) + 1),
                    'Time (seconds)': [p / FS for p in valid_peaks],
                    'Confidence Score': scores,
                    'Status': ['Anomalous' if score > threshold else 'Normal' for score in scores]
                })
                st.dataframe(scores_df)
                app_footer()

        except Exception as e:
            st.error(f"Error processing the ECG file: {str(e)}")


if __name__ == "__main__":
    main()
