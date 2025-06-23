import numpy as np
from scipy.signal import welch
from tqdm import tqdm



def extract_rbp_features(seg, fs=256):
    BANDS = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 75),
    }
    num_channels = seg.shape[1]
    rbp = np.zeros((num_channels, len(BANDS)), dtype=np.float32)
    nperseg = min(len(seg), 512)
    for ch in range(num_channels):
        freqs, psd = welch(seg[:, ch], fs=fs, nperseg=nperseg)
        total_power = np.trapz(psd, freqs)
        if total_power < 1e-12:
            total_power = 1e-12
        for idx, (lo, hi) in enumerate(BANDS.values()):
            band_mask = (freqs >= lo) & (freqs < hi)
            band_power = np.trapz(psd[band_mask], freqs[band_mask])
            rbp[ch, idx] = band_power / total_power
    return rbp



def extract_psd_features(eeg_segment, fs=256):
    num_channels = eeg_segment.shape[1]
    features = np.zeros((num_channels, 5), dtype=np.float32)
    for ch in range(num_channels):
        freqs, psd = welch(eeg_segment[:, ch], fs=fs, nperseg=min(len(eeg_segment), 512))
        total_power = np.trapz(psd, freqs)
        features[ch, 0] = total_power
        features[ch, 1] = np.mean(psd)
        features[ch, 2] = np.max(psd)
        features[ch, 3] = np.min(psd)
        features[ch, 4] = np.median(psd)
    #print("PSD features shape:", features.shape)
    return features

def extract_spectral_entropy(eeg_segment, fs=256):
    num_channels = eeg_segment.shape[1]
    features = np.zeros((num_channels, 1), dtype=np.float32)
    for ch in range(num_channels):
        freqs, psd = welch(eeg_segment[:, ch], fs=fs, nperseg=min(len(eeg_segment), 512))
        psd_norm = psd / (np.sum(psd) + 1e-12)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        features[ch, 0] = entropy
    #print("Spectral Entropy shape:", features.shape)
    return features

def extract_hjorth_features(eeg_segment):
    num_channels = eeg_segment.shape[1]
    features = np.zeros((num_channels, 2), dtype=np.float32)
    for ch in range(num_channels):
        x = eeg_segment[:, ch]
        var_x = np.var(x)
        dx = np.diff(x)
        var_dx = np.var(dx)
        mobility = np.sqrt(var_dx / (var_x + 1e-12))
        ddx = np.diff(dx)
        var_ddx = np.var(ddx)
        mobility_dx = np.sqrt(var_ddx / (var_dx + 1e-12))
        complexity = mobility_dx / (mobility + 1e-12)
        features[ch, 0] = mobility
        features[ch, 1] = complexity
    #print("Hjorth features shape:", features.shape)
    return features

def lempel_ziv_complexity(binary_str):
    i, c, l = 0, 1, 1
    n = len(binary_str)
    while i + l <= n:
        substring = binary_str[i:i+l]
        if binary_str.find(substring, i+1) == -1:
            c += 1
            i += l
            l = 1
            if i >= n:
                break
        else:
            l += 1
            if i + l > n:
                c += 1
                break
    return c

def extract_lz_complexity(eeg_segment):
    num_channels = eeg_segment.shape[1]
    features = np.zeros((num_channels, 1), dtype=np.float32)
    for ch in range(num_channels):
        x = eeg_segment[:, ch]
        med = np.median(x)
        binary_str = ''.join(['1' if val > med else '0' for val in x])
        features[ch, 0] = lempel_ziv_complexity(binary_str)
    #print("Lempel-Ziv Complexity shape:", features.shape)
    return features

def higuchi_fd(x, kmax=10):
    N = len(x)
    L = []
    for k in range(1, kmax+1):
        Lk = []
        for m in range(k):
            Lmk = 0.0
            count = int(np.floor((N - m) / k))
            if count > 1:
                for i in range(1, count):
                    Lmk += abs(x[m + i * k] - x[m + (i - 1) * k])
                Lmk = (Lmk * (N - 1) / (count * k))
                Lk.append(Lmk)
        if len(Lk) > 0:
            L.append(np.mean(Lk))
    L = np.array(L)
    lnL = np.log(L + 1e-12)
    lnK = np.log(1.0 / np.arange(1, kmax+1))
    slope, _ = np.polyfit(lnK, lnL, 1)
    return slope

def extract_higuchi_fd(eeg_segment, kmax=10):
    num_channels = eeg_segment.shape[1]
    features = np.zeros((num_channels, 1), dtype=np.float32)
    for ch in range(num_channels):
        x = eeg_segment[:, ch]
        features[ch, 0] = higuchi_fd(x, kmax=kmax)
    #print("Higuchi Fractal Dimension shape:", features.shape)
    return features

def extract_raw_signal_features(eeg_segment):
    num_channels = eeg_segment.shape[1]
    features = np.zeros((num_channels, 6), dtype=np.float32)
    for ch in range(num_channels):
        x = eeg_segment[:, ch]
        features[ch, 0] = np.mean(x)
        features[ch, 1] = np.min(x)
        features[ch, 2] = np.max(x)
        features[ch, 3] = np.median(x)
        features[ch, 4] = np.var(x)
        features[ch, 5] = np.std(x)
    #print("Raw signal features shape:", features.shape)
    return features


def compute_features(eeg_segment, fs=256):
    rbp_feats = extract_rbp_features(eeg_segment, fs=fs)
    psd_feats = extract_psd_features(eeg_segment, fs=fs)
    spec_entropy_feats = extract_spectral_entropy(eeg_segment, fs=fs)
    hjorth_feats = extract_hjorth_features(eeg_segment)
    lz_feats = extract_lz_complexity(eeg_segment)
    higuchi_feats = extract_higuchi_fd(eeg_segment)
    raw_feats = extract_raw_signal_features(eeg_segment)
  
    return np.concatenate(
        [rbp_feats, psd_feats, spec_entropy_feats, hjorth_feats, 
         lz_feats, higuchi_feats, raw_feats],
        axis=1
    )


def extract_all_features(level_data_by_patient, base_data_by_patient):
    print('GG')
    fs = 256
    epsilon = 1e-6
    
    # Get the shape of the RBP features
    rbp_feats_shape = extract_rbp_features(np.zeros((4, 256)), fs=fs).shape[1]

    all_patient_features = []
    all_patient_labels = []

    for p_idx, (participant_levels, participant_baselines) in enumerate(
        zip(level_data_by_patient, base_data_by_patient)
    ):
        participant_features_raw = []  # will eventually be shape (N, 4, 21)
        participant_labels_raw = []

        # Go level by level
        for level_idx in range(len(participant_levels)):
            level_segments = participant_levels[level_idx]
            baseline_segments = participant_baselines[level_idx]

            # Expecting exactly 1 baseline segment per level
            if len(baseline_segments) != 1:
                print(
                    f"Participant {p_idx+1}, Level {level_idx+1}: "
                    f"Expected 1 baseline segment but found {len(baseline_segments)}. Skipping."
                )
                continue

            # Baseline feature matrix => shape (4, 21)
            f_base = compute_features(baseline_segments[0], fs=fs)

            # Avoid dividing by near-zero baseline features
            f_base_safe = np.where(np.abs(f_base) < epsilon, epsilon, f_base)

            # For each segment in this level
            for seg_info in level_segments:
                seg_data = seg_info["features"]
                label = seg_info["label"]

                # Segment feature matrix => shape (4,21)
                f_seg = compute_features(seg_data, fs=fs)
                
                # Create a copy of the segment features for modification
                f_ratio = f_seg.copy()
                
                # Only divide non-RBP features by baseline
                # Assuming RBP features are the first rbp_feats_shape columns
                f_ratio[:, rbp_feats_shape:] = f_seg[:, rbp_feats_shape:] / f_base_safe[:, rbp_feats_shape:]

                # Collect
                participant_features_raw.append(f_ratio)
                participant_labels_raw.append(label)

  

        # Stack features => shape: (N, 4, 21)
        participant_features_raw = np.stack(participant_features_raw, axis=0)
        participant_labels_raw = np.array(participant_labels_raw, dtype=np.float32)

        # Z-score normalization across the N dimension for each channel-feature pair
        # mean & std => shape (4, 21), computed along axis=0
        feat_mean = participant_features_raw.mean(axis=0)  # (4,21)
        feat_std = participant_features_raw.std(axis=0)    # (4,21)

        # Prevent division by zero
        feat_std_safe = np.where(feat_std < epsilon, epsilon, feat_std)

        # Apply z-score 
        participant_features_z = (participant_features_raw - feat_mean) / feat_std_safe

        all_patient_features.append(participant_features_z)
        all_patient_labels.append(participant_labels_raw)

    return all_patient_features, all_patient_labels