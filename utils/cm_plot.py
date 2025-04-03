
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def plot_binary_confusion_matrix(y_true, y_pred):
    cm_binary = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_binary,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"],
    )
    plt.title("Binary Classification Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def plot_ternary_confusion_matrix(y_true, y_pred):
    cm_ternary = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_ternary,
        annot=True,
        fmt="d",
        cmap="Oranges",
        cbar=False,
        xticklabels=["Class 0", "Class 1", "Class 2"],
        yticklabels=["Class 0", "Class 1", "Class 2"],
    )
    plt.title("Ternary Classification Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def plot_continous_perclos(y_continuous, y_pred):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_continuous, y_pred)
    plt.xlabel("Actual PERCLOS")
    plt.ylabel("Predicted PERCLOS")
    plt.title("Predicted vs. Actual PERCLOS")
    plt.show()  