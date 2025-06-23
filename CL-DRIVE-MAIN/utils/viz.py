import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import signal


def plot_segments(segment_data):
    for i, segment in enumerate(segment_data):
        features = segment["features"]
        timestamps = segment["timestamps"]
        label = segment["label"]

        print(f"Segment: {i+1}")
        print(f"Features: {features.shape[1]}")
        print(f"Entries: {features.shape[0]}")
        print(f"Label: {label}\n")

        # Downsample the data
        features_downsampled = features[::10]
        timestamps_downsampled = timestamps[::10]

        feature_df = pd.DataFrame(
            features_downsampled,
            columns=[f"Feature {j+1}" for j in range(features_downsampled.shape[1])],
        )
        feature_df["Timestamp"] = timestamps_downsampled

        feature_df_melted = feature_df.melt(
            id_vars=["Timestamp"], var_name="Feature", value_name="Value"
        )

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=feature_df_melted,
            x="Timestamp",
            y="Value",
            hue="Feature",
            palette="tab10",
        )
        plt.title(f"Features of Segment {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Feature Values")
        plt.legend(title="Features", loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_psd(
    segment_data,
    fs=1.0,
    nperseg=None,
    electrode_positions=["TP9", "AF7", "AF8", "TP10"],
):
    if nperseg is None:
        nperseg = 2 * fs
    for i, segment in enumerate(segment_data):
        features = segment["features"]
        label = segment["label"]

        plt.figure(figsize=(12, 6))
        for j, electrode in enumerate(electrode_positions):
            frequencies, psd = signal.welch(
                features[:, j], fs=fs, nperseg=nperseg, scaling="density"
            )
            plt.semilogy(frequencies, psd, label=electrode)

        plt.title(f"Power Spectral Density - Segment {i+1} (Label: {label})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
