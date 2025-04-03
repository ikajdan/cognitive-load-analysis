import os
import scipy.io
import pandas as pd
import numpy as np

def read_participant_names(perclos_dir):
    participants = []

    for file_name in os.listdir(perclos_dir):
        if file_name.endswith(".mat"):
            participant_name = file_name.split(".")[0]
            participants.append(participant_name)

    df = pd.DataFrame(participants, columns=["Participant Name"])
    return df

def load_and_flatten(filepath, keys, num_channels=None, num_features=None):
    if os.path.exists(filepath):
        mat = scipy.io.loadmat(filepath)
        flattened = []
        for key in keys:
            if key in mat:
                data = mat[key]
                if data.ndim == 3:
                    data = np.transpose(data, (1, 0, 2)).reshape(885, -1)
                elif data.ndim == 2:
                    data = data
                flattened.append(data)
        return np.concatenate(flattened, axis=1)
    return np.zeros((885, 0))


def generate_column_names(prefix, num_channels, num_features):
    return [
        f"{prefix}_ch{ch+1}_feat{feat+1}"
        for ch in range(num_channels)
        for feat in range(num_features)
    ]


def load_single_user_data(participant_name, base_dir="SEED-VIG"):
    perclos_path = os.path.join(base_dir, "perclos_labels", f"{participant_name}.mat")
    eeg_2hz_path = os.path.join(base_dir, "EEG_Feature_2Hz", f"{participant_name}.mat")
    eeg_5bands_path = os.path.join(
        base_dir, "EEG_Feature_5Bands", f"{participant_name}.mat"
    )
    eog_path = os.path.join(base_dir, "EOG_Feature", f"{participant_name}.mat")
    forehead_2hz_path = os.path.join(
        base_dir, "Forehead_EEG", "EEG_Feature_2Hz", f"{participant_name}.mat"
    )
    forehead_5bands_path = os.path.join(
        base_dir, "Forehead_EEG", "EEG_Feature_5Bands", f"{participant_name}.mat"
    )

    perclos_data = load_and_flatten(
        perclos_path, ["perclos"], num_channels=1, num_features=1
    )
    eeg_2hz_data = load_and_flatten(
        eeg_2hz_path, ["psd_movingAve", "psd_LDS", "de_movingAve", "de_LDS"], 17, 25
    )
    eeg_5bands_data = load_and_flatten(
        eeg_5bands_path, ["psd_movingAve", "psd_LDS", "de_movingAve", "de_LDS"], 17, 5
    )
    eog_data = load_and_flatten(
        eog_path,
        ["features_table_ica", "features_table_minus", "features_table_icav_minh"],
        1,
        36,
    )
    forehead_2hz_data = load_and_flatten(
        forehead_2hz_path, ["psd_movingAve", "psd_LDS", "de_movingAve", "de_LDS"], 4, 25
    )
    forehead_5bands_data = load_and_flatten(
        forehead_5bands_path,
        ["psd_movingAve", "psd_LDS", "de_movingAve", "de_LDS"],
        4,
        5,
    )

    combined_data = np.hstack(
        [
            perclos_data,
            eeg_2hz_data,
            eeg_5bands_data,
            eog_data,
            forehead_2hz_data,
            forehead_5bands_data,
        ]
    )

    column_names = (
        ["perclos"]
        + generate_column_names("EEG_2Hz", 17, 25 * 4)
        + generate_column_names("EEG_5Bands", 17, 5 * 4)
        + generate_column_names("EOG", 1, 36 * 3)
        + generate_column_names("Forehead_EEG_2Hz", 4, 25 * 4)
        + generate_column_names("Forehead_EEG_5Bands", 4, 5 * 4)
    )

    df = pd.DataFrame(combined_data, columns=column_names)

    df["Participant Name"] = participant_name

    df['quantized_perclos'] = pd.cut(
    df['perclos'],
    bins=[0, 0.35, 0.70, 1.0],
    labels=[0, 1, 2],
    include_lowest=True
    )
    return df

def load_all_participants(base_dir="SEED-VIG"):
    perclos_dir = os.path.join(base_dir, "perclos_labels")
    participant_names = read_participant_names(perclos_dir)["Participant Name"]

    all_data = pd.DataFrame()

    for participant_name in participant_names:
        print(f"Loading data for participant: {participant_name}")
        df = load_single_user_data(participant_name, base_dir)
        all_data = pd.concat([all_data, df], ignore_index=True)

    return all_data


