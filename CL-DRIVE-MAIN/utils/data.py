import os

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, lfilter

participant_id = [
    "1030",
    "1105",
    "1106",
    "1241",
    "1271",
    "1314",
    "1323",
    "1337",
    "1372",
    "1417",
    "1434",
    "1544",
    "1547",
    "1595",
    "1629",
    "1716",
    "1717",
    "1744",
    "1868",
    "1892",
    "1953",
]

complexity_levels = range(1, 10)

modality_columns = {
    "EEG": ["TP9", "AF7", "AF8", "TP10"],
    "ECG": [
        "ECG LL-RA RAW",
        "ECG LL-RA CAL",
        "ECG LA-RA RAW",
        "ECG LA-RA CAL",
        "ECG Vx-RL RAW",
        "ECG Vx-RL CAL",
    ],
    "EDA": [
        "GSR RAW",
        "GSR Resistance CAL",
        "GSR Conductance CAL",
        "GSR RAW.1",
        "GSR Resistance CAL.1",
        "GSR Conductance CAL.1",
    ],
    "Gaze": [
        "ET_PupilLeft",
        "ET_PupilRight",
        "ET_GazeLeftx",
        "ET_GazeLefty",
        "ET_GazeRightx",
        "ET_GazeRighty",
        "ET_Gaze3DX",
        "ET_Gaze3DY",
        "ET_Gaze3DZ",
        "ET_ValidityLeftEye",
        "ET_ValidityRightEye",
        "ET_GazeDirectionLeftX",
        "ET_GazeDirectionLeftY",
        "ET_GazeDirectionLeftZ",
        "ET_GazeDirectionRightX",
        "ET_GazeDirectionRightY",
        "ET_GazeDirectionRightZ",
        "ET_Distance3D",
    ],
}


def read_data_with_structure(
    root_folder,
    verbose=False,
    drop_na=True,
    load_baseline=False,
    participant=None,
    modality=None,
):
    if not os.path.isdir(root_folder):
        raise ValueError("Invalid dataset root folder.")
    if participant is None:
        raise ValueError("No participant ID specified.")
    if modality is None:
        raise ValueError("No modality specified.")

    modality_folder = os.path.join(root_folder, modality)
    if not os.path.isdir(modality_folder):
        raise ValueError(f"Modality '{modality}' does not exist.")

    participant_path = os.path.join(modality_folder, participant)
    if not os.path.isdir(participant_path):
        raise ValueError(f"Participant '{participant}' does not exist.")

    all_data = []
    for file in os.listdir(participant_path):
        file_path = os.path.join(participant_path, file)
        is_baseline = "baseline" in file

        if load_baseline and not is_baseline:
            continue
        if not load_baseline and is_baseline:
            continue

        data = pd.read_csv(file_path)
        data["Participant_ID"] = participant
        data["Modality"] = modality
        data["Complexity_Level"] = int(file.split("_")[-1].replace(".csv", ""))
        if verbose:
            print(f"Reading file: {file_path}")
            print(f"Shape: {data.shape}")
        all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)
    # Remove rows with timestamps less than 120 seconds and greater than 300 seconds,
    # as we don't have labels for them
    if load_baseline:
        combined_data = combined_data[(combined_data["Timestamp"] < 120)].copy()
    else:
        combined_data = combined_data[
            (combined_data["Timestamp"] >= 120) & (combined_data["Timestamp"] < 300)
        ].copy()

    if drop_na:
        columns_to_check = modality_columns.get(modality, [])
        original_row_count = len(combined_data)
        combined_data = combined_data.dropna(subset=columns_to_check, how="all").copy()

        dropped_rows = original_row_count - len(combined_data)
        dropped_row_percentage = (dropped_rows / original_row_count) * 100
        print(
            f"Dropped Rows: {dropped_rows}/{original_row_count} ({dropped_row_percentage:.2f}%)"
        )

    labels_folder = os.path.join(root_folder, "Labels")
    if not load_baseline and os.path.isdir(labels_folder):
        label_file = os.path.join(labels_folder, f"{participant}.csv")
        if os.path.isfile(label_file):
            labels = pd.read_csv(label_file)
            if verbose:
                print(f"Reading labels file: {label_file}")
            # Labels start from 0, and timestamps are from 120
            labels["time"] += 120
            labels = labels.sort_values(by="time")
            label_columns = [
                col for col in labels.columns if col not in ["time", "Participant_ID"]
            ]
            for col in label_columns:
                combined_data[col] = pd.NA
            for _, label_row in labels.iterrows():
                start_time = label_row["time"] - 10
                end_time = label_row["time"]
                time_filter = (
                    (combined_data["Timestamp"] >= start_time)
                    & (combined_data["Timestamp"] < end_time)
                    & (combined_data["Participant_ID"] == participant)
                )
                for col in label_columns:
                    combined_data.loc[time_filter, col] = label_row[col]
        else:
            print(f"No labels file found for participant: {participant}")

    # Add self-assessed level
    combined_data["Selfassessed_Level"] = combined_data.apply(
        lambda row: row.get(f"lvl_{int(row['Complexity_Level'])}", pd.NA), axis=1
    ).astype("Int64", errors="ignore")

    # Remove level columns
    combined_data = combined_data.drop(
        columns=[col for col in combined_data.columns if col.startswith("lvl_")],
        errors="ignore",
    )

    if verbose:
        print(f"Total Participants: {combined_data['Participant_ID'].nunique()}")
        print(f"Total Complexity Levels: {combined_data['Complexity_Level'].nunique()}")
        print(f"Total Rows: {len(combined_data)}")
        print(f"Total Columns: {combined_data.shape[1]}")

    return combined_data


def butter_lowpass(cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low")
    return b, a


def butter_highpass(cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high")
    return b, a


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order)
    return lfilter(b, a, data)


def highpass_filter(data, cutoff, fs, order=2):
    b, a = butter_highpass(cutoff, fs, order)
    return lfilter(b, a, data)


def bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


def notch_filter(data, notch_freq, fs, quality_factor):
    nyquist = 0.5 * fs
    notch = notch_freq / nyquist
    b, a = iirnotch(notch, quality_factor)
    return lfilter(b, a, data)


def prepare_segments(
    df=None,
    verbose=False,
    preprocess=False,
    modality=None,
    time_step=10,
):
    if df is None or df.empty:
        print("Warning: No data provided or DataFrame is empty. Skipping...")
        return []

    if modality not in ["EEG", "ECG", "EDA", "Gaze"]:
        raise ValueError("Modality must be one of: 'EEG', 'ECG', 'EDA', 'Gaze'")

    if "Timestamp" not in df.columns or df["Timestamp"].empty:
        print(
            "Warning: Missing or empty 'Timestamp' column. Skipping segment preparation..."
        )
        return []

    df = df.sort_values(by="Timestamp")
    fs = 256
    allowed_error = 2

    selected_columns = modality_columns[modality]

    start_time = df["Timestamp"].iloc[0]
    end_time = df["Timestamp"].iloc[-1]

    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")
    print(f"Time Step: {time_step}")
    print(f"Expected Entries: {fs * time_step}\n")

    if abs(start_time - 120) > allowed_error and start_time > allowed_error:
        print(f"Warning: Unexpected start time {start_time}. Skipping this data.")
        return []

    if abs(end_time - 300) > allowed_error and abs(end_time - 120) > allowed_error:
        print(f"Warning: Unexpected end time {end_time}. Skipping this data.")
        return []

    all_data = []

    for t in range(int(start_time), int(end_time), time_step):
        segment = df[(df["Timestamp"] >= t) & (df["Timestamp"] < t + time_step)]

        if segment.empty:
            print(f"Skipping empty segment at time {t}.")
            continue

        features = segment[selected_columns].values
        timestamps = segment["Timestamp"].values

        label = (
            segment["Selfassessed_Level"].iloc[0]
            if "Selfassessed_Level" in segment
            else np.nan
        )

        if verbose:
            print(f"Processing segment {t}...")
            print(f"Entries: {len(segment)}")
            print(f"Label: {label}\n")

        if abs(len(segment) - fs * time_step) > 10:
            if time_step < 120:  # Don't check for valid length of baseline data
                print(
                    f"Warning: Segment at time {t} has {len(segment)} entries (expected {fs * time_step}). Skipping..."
                )
                continue
            else:
                print(
                    f"Warning: Segment at time {t} has {len(segment)} entries (expected {fs * time_step})."
                )

        if (
            "Selfassessed_Level" in segment
            and len(segment["Selfassessed_Level"].unique()) > 1
        ):
            print(f"Warning: Multiple labels found in segment at time {t}. Skipping...")
            continue

        if preprocess:
            for col_idx in range(features.shape[1]):
                if modality == "EEG":
                    features[:, col_idx] = bandpass_filter(
                        features[:, col_idx], 0.4, 75, fs
                    )
                    features[:, col_idx] = notch_filter(
                        features[:, col_idx], 60, fs, 30
                    )
                elif modality == "ECG":
                    features[:, col_idx] = bandpass_filter(
                        features[:, col_idx], 5, 15, fs
                    )
                elif modality == "EDA":
                    features[:, col_idx] = lowpass_filter(features[:, col_idx], 3, fs)
                    features[:, col_idx] = highpass_filter(
                        features[:, col_idx], 0.05, fs
                    )
                elif modality == "Gaze":
                    raise NotImplementedError("Gaze preprocessing not implemented.")

        all_data.append(
            {"timestamps": timestamps, "features": features, "label": label}
        )

    return all_data


def load_tensor_concat(dataset, modality):
    tensor_data = []

    for participant in participant_id:
        participant_segments = []

        try:
            data = read_data_with_structure(
                dataset,
                drop_na=True,
                load_baseline=False,
                participant=participant,
                modality=modality,
            )
            baseline = read_data_with_structure(
                dataset,
                drop_na=True,
                load_baseline=True,
                participant=participant,
                modality=modality,
            )

            for level in complexity_levels:
                try:
                    level_data = data[data["Complexity_Level"] == level]
                    level_baseline = baseline[baseline["Complexity_Level"] == level]

                    segments = prepare_segments(
                        level_data, preprocess=True, modality=modality
                    )

                    baseline_segments = (
                        prepare_segments(
                            level_baseline,
                            preprocess=True,
                            modality=modality,
                            time_step=120,
                        )
                        if not level_baseline.empty
                        else []
                    )

                    valid_level_segments = []
                    for idx, seg in enumerate(segments):
                        n_entries = len(seg["features"])

                        if n_entries != 2560:
                            print(
                                f"Dropping segment: Participant {participant}, Level {level}, "
                                f"Segment {idx}, Entries: {n_entries}\n"
                            )
                            continue

                        valid_level_segments.append(seg)

                    valid_baseline_segments = [
                        seg for seg in baseline_segments if len(seg["features"]) == 120
                    ]

                    if not valid_level_segments and not valid_baseline_segments:
                        print(
                            f"Warning: No valid segments for Participant {participant}, Level {level}.\n"
                        )
                        continue

                    if valid_level_segments:
                        features = np.array(
                            [seg["features"] for seg in valid_level_segments]
                        )
                        labels = np.array(
                            [seg["label"] for seg in valid_level_segments]
                        )
                        labels_expanded = np.tile(
                            labels[:, np.newaxis, np.newaxis], (1, features.shape[1], 1)
                        )
                        level_segments = np.concatenate(
                            [features, labels_expanded], axis=-1
                        )
                    else:
                        level_segments = np.empty((0, 2560, 1))

                    if valid_baseline_segments:
                        baseline_features = np.array(
                            [seg["features"] for seg in valid_baseline_segments]
                        )
                        baseline_labels = np.full(
                            (baseline_features.shape[0], 1, 1), np.nan
                        )
                        baseline_labels_expanded = np.tile(
                            baseline_labels, (1, baseline_features.shape[1], 1)
                        )
                        baseline_segments_array = np.concatenate(
                            [baseline_features, baseline_labels_expanded], axis=-1
                        )

                        level_segments = np.concatenate(
                            [level_segments, baseline_segments_array], axis=0
                        )

                    participant_segments.append(level_segments)

                except ValueError as e:
                    print(f"{str(e)}")
                    continue

            if participant_segments:
                participant_data = np.concatenate(participant_segments, axis=0)
                tensor_data.append(participant_data)

        except ValueError as e:
            print(f"{str(e)}")
            continue

    return tensor_data


def load_tensor(dataset, modality, fs=256):
    print("NEW777")
    participant_id = [
        "1030",
        "1105",
        "1106",
        "1241",
        "1271",
        "1314",
        "1323",
        "1337",
        "1372",
        "1417",
        "1434",
        "1544",
        "1547",
        "1595",
        "1629",
        "1716",
        "1717",
        "1744",
        "1868",
        "1892",
        "1953",
    ]
    error_margin = 5
    complexity_levels = range(1, 10)
    LEVEL_DATA_LENGTH = fs * 10 - error_margin
    BASELINE_DATA_LENGTH = fs * 100 - error_margin

    level_data_by_patient = []
    base_data_by_patient = []

    for participant in participant_id:
        try:
            data = read_data_with_structure(
                dataset,
                drop_na=True,
                load_baseline=False,
                participant=participant,
                modality=modality,
            )
            baseline = read_data_with_structure(
                dataset,
                drop_na=True,
                load_baseline=True,
                participant=participant,
                modality=modality,
            )

            patient_level_data = []
            patient_base_data = []

            for level in complexity_levels:
                try:
                    level_data = data[data["Complexity_Level"] == level]
                    level_baseline = baseline[baseline["Complexity_Level"] == level]

                    segments = prepare_segments(
                        level_data, preprocess=True, modality=modality
                    )
                    baseline_segments = (
                        prepare_segments(
                            level_baseline,
                            preprocess=True,
                            modality=modality,
                            time_step=120,
                        )
                        if not level_baseline.empty
                        else []
                    )

                    valid_level_segments = []
                    for idx, seg in enumerate(segments):
                        if len(seg["features"]) < LEVEL_DATA_LENGTH:
                            print(
                                f"Dropping segment: Participant {participant}, Level {level}, "
                                f"Segment {idx}, Entries: {len(seg['features'])} (expected {LEVEL_DATA_LENGTH})"
                            )
                            continue
                        valid_level_segments.append(seg)

                    valid_baseline_segments = []
                    for idx, seg in enumerate(baseline_segments):
                        if len(seg["features"]) < BASELINE_DATA_LENGTH:
                            print(
                                f"Dropping baseline: Participant {participant}, Level {level}, "
                                f"Entries: {len(seg['features'])} (expected {BASELINE_DATA_LENGTH})"
                            )
                            continue
                        valid_baseline_segments.append(seg)

                    if not valid_level_segments:
                        print(
                            f"Missing level data for Participant {participant}, Level {level}. Skipping."
                        )
                        continue

                    if not valid_baseline_segments:
                        print(
                            f"Missing baseline data for Participant {participant}, Level {level}. Skipping."
                        )
                        continue

                    level_segments_data = []
                    for seg in valid_level_segments:
                        level_segments_data.append(
                            {"features": seg["features"], "label": seg["label"]}
                        )

                    baseline_data = []
                    for seg in valid_baseline_segments:
                        baseline_data.append(seg["features"])

                    patient_level_data.append((level, level_segments_data))
                    patient_base_data.append((level, baseline_data))

                except ValueError as e:
                    print(
                        f"Error processing level {level} for participant {participant}: {str(e)}"
                    )
                    continue

            if patient_level_data and patient_base_data:
                patient_level_data.sort(key=lambda x: x[0])
                patient_base_data.sort(key=lambda x: x[0])
                level_data_arrays = [d for _, d in patient_level_data]
                base_data_arrays = [d for _, d in patient_base_data]
                level_data_by_patient.append(level_data_arrays)
                base_data_by_patient.append(base_data_arrays)

        except ValueError as e:
            print(f"Error processing participant {participant}: {str(e)}")
            continue

    return level_data_by_patient, base_data_by_patient
