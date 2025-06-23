
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import accuracy_score, mean_squared_error


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


def plot_confusion_matrix(all_targets, all_predictions, task_type, model_name):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_targets, all_predictions)
    if task_type == 'binary':
        labels = ["Class 0", "Class 1"]
        cmap = "Blues"
        title = f"{model_name.upper()} Binary Classification Confusion Matrix"
    else: # ternary
  
        unique_labels = sorted(list(np.unique(np.concatenate((all_targets, all_predictions)))))
   
        if len(unique_labels) == 3:
            labels = ["Class 0", "Class 1", "Class 2"]
        elif len(unique_labels) == 2: 
             labels = [f"Class {i}" for i in unique_labels]
         
        else: 
             labels = [f"Class {i}" for i in unique_labels]

        cmap = "Oranges"
        title = f"{model_name.upper()} Ternary Classification Confusion Matrix"

    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def plot_regression_results(all_targets, all_predictions, model_name):
    overall_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    print(f"\nOverall Continuous Regression RMSE: {overall_rmse:.4f}")
    plt.figure(figsize=(8, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.5)
    min_val = min(np.min(all_targets), np.min(all_predictions)) if len(all_targets) > 0 else 0
    max_val = max(np.max(all_targets), np.max(all_predictions)) if len(all_targets) > 0 else 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f"{model_name.upper()} Regression: Predicted vs Actual PERCLOS")
    plt.xlabel("Actual PERCLOS")
    plt.ylabel("Predicted PERCLOS")
    plt.tight_layout()
    plt.show()
    return overall_rmse

