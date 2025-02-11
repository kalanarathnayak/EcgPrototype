import numpy as np
import pandas as pd
import wfdb
from wfdb import processing
from sklearn import preprocessing
import scipy.signal
from config import FS, WINDOW_SIZE

def preprocess_data(data_path, patients):
    """
    Preprocessing steps:
    1. Remove baseline wander using median filters.
    2. Extract heartbeats with ±50 ms around R-peaks.
    3. Scale the data.
    """
    beats = []
    labels = []
    patient_ids = []

    # Calculate median filter window sizes (ensure odd numbers)
    window_200ms = int(0.2 * FS)
    if window_200ms % 2 == 0:
        window_200ms += 1

    window_600ms = int(0.6 * FS)
    if window_600ms % 2 == 0:
        window_600ms += 1

    for patient in patients:
        print(f'Processing record number {patient}')

        # Read record and extract lead II
        record = wfdb.rdrecord(data_path + patient, smooth_frames=True)
        lead_II = record.p_signal[:, 0]

        # Remove baseline wander using two median filters
        filt1 = scipy.signal.medfilt(lead_II, window_200ms)
        filt2 = scipy.signal.medfilt(filt1, window_600ms)
        lead_II = lead_II - filt2

        # Scale the data (handling any NaNs)
        lead_II = preprocessing.scale(np.nan_to_num(lead_II))

        # QRS detection
        qrs = processing.XQRS(sig=lead_II, fs=FS)
        qrs.detect()
        peaks = qrs.qrs_inds

        # Extract beats (±50 ms around R-peaks)
        for peak in peaks[1:-1]:
            start = peak - WINDOW_SIZE // 2
            end = peak + WINDOW_SIZE // 2

            if start < 0 or end > len(lead_II):
                continue

            # Get annotation for this segment
            ann = wfdb.rdann(data_path + patient, extension='atr',
                           sampfrom=start, sampto=end,
                           return_label_elements=['symbol'])

            if len(ann.symbol) == 1:
                symbol = ann.symbol[0]
                # Label Normal beats as 1, others as 0
                if symbol == 'N':
                    labels.append(1)  # Normal
                else:
                    labels.append(0)  # Abnormal
                beats.append(lead_II[start:end])
                patient_ids.append(patient)

    return np.array(beats), np.array(labels), np.array(patient_ids)

def prepare_data(beats, labels, patient_ids):
    """
    Prepare data splits:
    1. Sample 18,824 normal beats.
    2. Split normals: 80% for training (further split 80:20 into train/val) and 20% for testing.
    3. Take 10% of abnormal beats for testing.
    """
    df = pd.DataFrame(beats)
    df['target'] = labels
    df['patient_id'] = patient_ids

    normal_df = df[df['target'] == 1]  # Normal beats
    abnormal_df = df[df['target'] != 1]  # Abnormal beats

    print(f"\nTotal Normal samples: {len(normal_df)}")
    print(f"Total Abnormal samples: {len(abnormal_df)}")

    # Sample 18,824 normal beats
    normal_df = normal_df.sample(n=18824, random_state=42)

    # Split normal data into 80% training and 20% testing
    train_val_size = int(0.8 * len(normal_df))
    train_val_df = normal_df.sample(n=train_val_size, random_state=42)
    test_normal = normal_df[~normal_df.index.isin(train_val_df.index)]

    # Further split training data into training and validation (80:20)
    train_size = int(0.8 * len(train_val_df))
    train_df = train_val_df.sample(n=train_size, random_state=42)
    val_df = train_val_df[~train_val_df.index.isin(train_df.index)]

    # Select 10% of abnormal beats for testing
    test_abnormal = abnormal_df.sample(n=int(0.1 * len(abnormal_df)), random_state=42)

    # Combine test normals and abnormals, then shuffle
    test_df = pd.concat([test_normal, test_abnormal])
    test_df = test_df.sample(frac=1, random_state=42)

    print(f"\nTraining samples (Normal only): {len(train_df)}")
    print(f"Validation samples (Normal only): {len(val_df)}")
    print(f"Test samples - Normal: {len(test_normal)}, Abnormal: {len(test_abnormal)}")

    return train_df, val_df, test_df