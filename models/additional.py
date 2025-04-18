import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import torch

def create_token_groups(all_participants_data):
    feature_groups = {}
    columns_to_drop = ["perclos", "quantized_perclos", "Participant Name"]
 
    columns_to_process = [col for col in all_participants_data.columns if col not in columns_to_drop]

    col_to_X_index = {col: i for i, col in enumerate(columns_to_process)}

    eeg_types = ["EEG_2Hz", "EEG_5Bands"]
    forehead_types = ["Forehead_EEG_2Hz", "Forehead_EEG_5Bands"]

    for feature_type in eeg_types:
    
        channel_features = [col for col in columns_to_process if col.startswith(feature_type)]
      
        channels = sorted(list(set(
            col.split('_')[2].replace('ch', '')
            for col in channel_features if '_ch' in col and len(col.split('_')) > 2 and col.split('_')[2].startswith('ch')
        )))
        for ch in channels:
         
            group_cols = [col for col in channel_features if f"_ch{ch}_" in col]
            if not group_cols:
                continue
       
            indices = [col_to_X_index[col] for col in group_cols]
            if not indices: continue
 
            feature_groups[f"{feature_type}_Channel_{ch}"] = (min(indices), len(indices))

    for feature_type in forehead_types:
        channel_features = [col for col in columns_to_process if col.startswith(feature_type)]
        channels = sorted(list(set(
            col.split('_')[3].replace('ch', '')
            for col in channel_features if '_ch' in col and len(col.split('_')) > 3 and col.split('_')[3].startswith('ch')
        )))
        for ch in channels:
            group_cols = [col for col in channel_features if f"_ch{ch}_" in col]
            if not group_cols:
                continue
            indices = [col_to_X_index[col] for col in group_cols]
            if not indices: continue
            feature_groups[f"{feature_type}_Channel_{ch}"] = (min(indices), len(indices))

    eog_features = [col for col in columns_to_process if col.startswith("EOG")]
    if eog_features:
        eog_indices = [col_to_X_index[col] for col in eog_features]
        if eog_indices:
             feature_groups["EOG"] = (min(eog_indices), len(eog_indices))

    return feature_groups

def evaluate_model(model, test_loader, criterion, device, task_type):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets_device = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets_device)
            running_loss += loss.item()

            if task_type == 'binary':
                preds = (outputs > 0.5).float().cpu().numpy()
            elif task_type == 'ternary':
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                preds = outputs.cpu().numpy()

            all_predictions.extend(preds.flatten())
            # Targets moved to CPU only when needed for extending list
            all_targets.extend(targets.cpu().numpy().flatten())

    all_predictions_np = np.array(all_predictions)
    all_targets_np = np.array(all_targets)

    if task_type == 'binary' or task_type == 'ternary':
      
        accuracy = accuracy_score(all_targets_np, all_predictions_np)
        return running_loss / len(test_loader), accuracy
    else:
        rmse = np.sqrt(mean_squared_error(all_targets_np, all_predictions_np))
        return running_loss / len(test_loader), rmse




def prepare_data_for_pytorch(all_participants_data):
    columns_to_drop = ["perclos", "quantized_perclos", "Participant Name"]
    feature_columns = [col for col in all_participants_data.columns if col not in columns_to_drop]

    #columns_to_drop = ["perclos", "quantized_perclos", "Participant Name"]
    ##feature_columns = [
    #col for col in all_participants_data.columns
    #if col not in columns_to_drop
    #and not col.startswith("EOG")]
    
    X = all_participants_data[feature_columns].values

    y_binary = (all_participants_data["perclos"] > 0.5).astype(int).values
    y_ternary = all_participants_data["quantized_perclos"].values
    y_continuous = all_participants_data["perclos"].values

    groups = all_participants_data["Participant Name"].values

    return X, y_binary, y_ternary, y_continuous, groups, feature_columns


def create_torch_dataset(X, y, task_type):
    X_tensor = torch.FloatTensor(X)
    if task_type == 'ternary':
        y_tensor = torch.LongTensor(y)
    elif task_type == 'binary':
        y_tensor = torch.FloatTensor(y).view(-1, 1)
    else:
        y_tensor = torch.FloatTensor(y).view(-1, 1)
    return torch.utils.data.TensorDataset(X_tensor, y_tensor)












