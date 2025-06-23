import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

def get_default_params(estimator_name):
    estimator_name = estimator_name.lower()
    if estimator_name in ['rf', 'randomforest']:
        return {'random_state': None, 'n_estimators': 1000, 'max_depth': 50}
    elif estimator_name in ['adaboost', 'ab']:
        return {'random_state': None, 'n_estimators': 100}
    elif estimator_name in ['decisiontree', 'dt']:
        return {'random_state': None, 'max_depth': None}
    elif estimator_name in ['nb', 'naivebayes']:
        return {}
    elif estimator_name in ['knn']:
        return {'n_neighbors': 5}
    elif estimator_name in ['lda']:
        return {}
    elif estimator_name in ['svm']:
        return {'random_state': None, 'C': 1.0, 'kernel': 'rbf'}
    elif estimator_name in ['xgb', 'xgboost']:
        return {
            
            'max_depth': 20,
            'n_estimators': 500
        }
    elif estimator_name in ['mlp']:
        return {'random_state': None, 'hidden_layer_sizes': (20,20)}
    else:
        return {}

def get_model(estimator_name, estimator_params):
    estimator_name = estimator_name.lower()
    if estimator_name in ['adaboost', 'ab']:
        return AdaBoostClassifier(**estimator_params)
    elif estimator_name in ['decisiontree', 'dt']:
        return DecisionTreeClassifier(**estimator_params)
    elif estimator_name in ['naivebayes', 'nb']:
        return GaussianNB(**estimator_params)
    elif estimator_name in ['knn']:
        return KNeighborsClassifier(**estimator_params)
    elif estimator_name in ['lda']:
        return LinearDiscriminantAnalysis(**estimator_params)
    elif estimator_name in ['randomforest', 'rf']:
        return RandomForestClassifier(**estimator_params)
    elif estimator_name in ['svm']:
        return SVC(**estimator_params)
    elif estimator_name in ['xgb', 'xgboost']:
        if XGBClassifier is None:
            raise ImportError("XGBoost is not installed.")
        return XGBClassifier(**estimator_params)
    elif estimator_name in ['mlp']:
        return MLPClassifier(**estimator_params)
    else:
        raise ValueError(f"Unknown estimator: {estimator_name}")

def loso_validation(all_features, all_labels, tensor_shape=(4, 16), 
                    estimator_name='rf', estimator_params=None):
    channels, feat_dim = tensor_shape
    flatten_dim = channels * feat_dim
    if estimator_params is None:
        estimator_params = get_default_params(estimator_name)
    all_y_test_bin = []
    all_y_pred_bin = []
    for test_idx in range(len(all_features)):
        X_train, y_train = [], []
        for i in range(len(all_features)):
            if i != test_idx:
                X_train.append(all_features[i].reshape(-1, flatten_dim))
                y_train.append(all_labels[i])
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        y_train_bin = np.where(y_train <= 4, 0, 1)
        y_train_bin = y_train
        X_test = all_features[test_idx].reshape(-1, flatten_dim)
        y_test = all_labels[test_idx]
        #y_test_bin = np.where(y_test <= 4, 0, 1)
        y_test_bin = y_test
        clf = get_model(estimator_name, estimator_params)
        clf.fit(X_train, y_train_bin)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test_bin, y_pred)
        cm = confusion_matrix(y_test_bin, y_pred)
        print(f"Participant {test_idx} Test Accuracy: {acc:.3f}")
        print("Confusion Matrix:\n", cm)
        print("------")
        all_y_test_bin.append(y_test_bin)
        all_y_pred_bin.append(y_pred)
    all_y_test_bin = np.concatenate(all_y_test_bin)
    all_y_pred_bin = np.concatenate(all_y_pred_bin)
    overall_accuracy = accuracy_score(all_y_test_bin, all_y_pred_bin)
    print(f"\nOverall Accuracy Across All Folds: {overall_accuracy:.3f}")
    cm_agg = confusion_matrix(all_y_test_bin, all_y_pred_bin)
    print("Aggregated Confusion Matrix:\n", cm_agg)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_agg, annot=True, fmt='d', cmap='Blues')
    plt.title("Aggregated Confusion Matrix (All Folds)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def kfold_validation(all_features, all_labels, tensor_shape=(4, 16), 
                     estimator_name='rf', estimator_params=None):
    print('ff')
    channels, feat_dim = tensor_shape
    flatten_dim = channels * feat_dim
    num_patients = len(all_features)
    folds = [
        [0, 20], [1, 19], [2, 18], [3, 17], [4, 16],
        [5, 15], [6, 14], [7, 13], [8, 12],
        [9, 10, 11]
    ]
    if estimator_params is None:
        estimator_params = get_default_params(estimator_name)
    all_y_test_bin = []
    all_y_pred_bin = []
    for fold_idx, test_indices in enumerate(folds):
        X_train, y_train = [], []
        for i in range(num_patients):
            if i not in test_indices:
                X_train.append(all_features[i].reshape(-1, flatten_dim))
                y_train.append(all_labels[i])
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        #y_train_bin = np.where(y_train <= 4, 0, 1)
        y_train_bin = y_train
        X_test, y_test = [], []
        for i in test_indices:
            X_test.append(all_features[i].reshape(-1, flatten_dim))
            y_test.append(all_labels[i])
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        #y_test_bin = np.where(y_test <= 4, 0, 1)
        y_test_bin = y_test
        clf = get_model(estimator_name, estimator_params)
        clf.fit(X_train, y_train_bin)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test_bin, y_pred)
        cm = confusion_matrix(y_test_bin, y_pred)
        print(f"Fold {fold_idx} Test Accuracy: {acc:.3f}")
        print("Confusion Matrix:\n", cm)
        print("------")
        all_y_test_bin.append(y_test_bin)
        all_y_pred_bin.append(y_pred)
    all_y_test_bin = np.concatenate(all_y_test_bin)
    all_y_pred_bin = np.concatenate(all_y_pred_bin)
    overall_accuracy = accuracy_score(all_y_test_bin, all_y_pred_bin)
    print(f"\nOverall Accuracy Across All Folds: {overall_accuracy:.3f}")
    cm_agg = confusion_matrix(all_y_test_bin, all_y_pred_bin)
    print("Aggregated Confusion Matrix:\n", cm_agg)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_agg, annot=True, fmt='d', cmap='Blues')
    plt.title("Aggregated Confusion Matrix (All Folds)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
