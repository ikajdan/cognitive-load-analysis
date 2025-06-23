import os
import os
import pandas as pd

def load_subject_data(base_dir: str, subject_id: str):

    data = {}
    modalities = ['EEG', 'ECG', 'EDA', 'Gaze']

    # Load each modality
    for mod in modalities:
        mod_dir = os.path.join(base_dir, mod, subject_id)
        if not os.path.isdir(mod_dir):
            data[mod] = None
            continue
        modality_data = {}
        for level in range(1, 10):
            data_file = os.path.join(mod_dir, f'{mod.lower()}_data_level_{level}.csv')
            baseline_file = os.path.join(mod_dir, f'{mod.lower()}_baseline_level_{level}.csv')
            if os.path.isfile(data_file) and os.path.isfile(baseline_file):
                modality_data[level] = {
                    'data': pd.read_csv(data_file),
                    'baseline': pd.read_csv(baseline_file)
                }
        data[mod] = modality_data if modality_data else None

    # Load Labels (single CSV file named <subject_id>.csv)
    labels_file = os.path.join(base_dir, 'Labels', f"{subject_id}.csv")
    if os.path.isfile(labels_file):
        data['Labels'] = pd.read_csv(labels_file)
    else:
        data['Labels'] = None

    return data

def load_all_data(base_dir: str):
    all_data = {}
    eeg_dir = os.path.join(base_dir, 'EEG')
    if not os.path.isdir(eeg_dir):
        return all_data

    for subject_id in os.listdir(eeg_dir):
        subject_path = os.path.join(eeg_dir, subject_id)
        if os.path.isdir(subject_path):
            all_data[subject_id] = load_subject_data(base_dir, subject_id)
    return all_data