import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.data import read_data_with_structure,prepare_segments

def load_tensor1(dataset_path="./cl_drive/", modality="EEG"):
    participants = [
        "1030", "1105", "1106", "1241", "1271", "1314", "1323", "1337",
        "1372", "1417", "1434", "1544", "1547", "1595", "1629", "1716",
        "1717", "1744", "1868", "1892", "1953"
    ]
    
    complexity_levels = range(1, 10)  # Levels 1-9
    all_participant_data = []
    valid_segments = []
    
    for participant in tqdm(participants, desc="Processing participants"):
        participant_segments = []
        
        try:
            # Read data using existing function
            data = read_data_with_structure(
                dataset_path,
                drop_na=True,
                participant=participant,
                modality=modality
            )
            for level in complexity_levels:
                try:
                    # Filter by complexity level
                    level_data = data[data["Complexity_Level"] == level]

                    if len(level_data) == 0:
                        print(f"Warning: No data for participant {participant}, level {level}")
                        continue

                    # Use existing prepare_segments function
                    segments = prepare_segments(level_data,preprocess=True, modality=modality)

                    # Validate and process each segment
                    valid_level_segments = []
                    for idx, seg in enumerate(segments):
                        n_entries = len(seg['features'])
                        label_val = seg['label']

                        # Skip segments that don't match 2560 data points
                        if n_entries < 2558:
                            print(f"Dropping segment: Participant {participant}, Level {level}, "
                                  f"Segment {idx}, Entries: {n_entries}")
                            continue

                        # Skip segments with label=0
                        if label_val == 0:
                            print(f"Skipping segment with label=0: Participant {participant}, "
                                  f"Level {level}, Segment {idx}")
                            continue

                        valid_level_segments.append(seg)

                    if not valid_level_segments:
                        print(f"Warning: No valid segments for participant {participant}, level {level}")
                        continue

                    # Convert valid segments to arrays
                    features = np.array([seg["features"] for seg in valid_level_segments])
                    labels = np.array([seg["label"] for seg in valid_level_segments])

                    # Tile the label across the time dimension
                    labels_expanded = np.tile(labels[:, np.newaxis, np.newaxis],
                                              (1, features.shape[1], 1))

                    # Concatenate along the last dimension
                    level_segments = np.concatenate([features, labels_expanded], axis=-1)

                    participant_segments.append(level_segments)

                except Exception as e:
                    print(f"Error processing level {level} for participant {participant}: {str(e)}")
                    continue

            
            if participant_segments:
                participant_data = np.concatenate(participant_segments, axis=0)
                all_participant_data.append(participant_data)
                
        except Exception as e:
            print(f"Error processing participant {participant}: {str(e)}")
            continue
    
    if not all_participant_data:
        raise ValueError("No data was successfully processed")
    
    
    return all_participant_data



