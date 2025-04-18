import torch
import torch.nn as nn

class MultiHeadConv1DModel(nn.Module):
    def __init__(self, feature_groups, output_size, task_type='binary'):

        super().__init__()
        self.task_type = task_type

        # 1) Define the channel groups
        groups = {
            'eeg2':   sorted(k for k in feature_groups if 'EEG_2Hz_Channel' in k),
            'eeg5':   sorted(k for k in feature_groups if 'EEG_5Bands_Channel' in k),
            'fore2':  sorted(k for k in feature_groups if 'Forehead_EEG_2Hz_Channel' in k),
            'fore5':  sorted(k for k in feature_groups if 'Forehead_EEG_5Bands_Channel' in k),
        }

        eog_key = 'EOG'

        # 2) Compute slice bounds and dims for each group
        self.slices = {}
        for name, keys in groups.items():
            start = min(feature_groups[k][0] for k in keys)
            size  = sum(feature_groups[k][1] for k in keys)
            n_ch  = len(keys)
            feat  = feature_groups[keys[0]][1]
            self.slices[name] = (start, size, n_ch, feat)

        # EOG dims
        self.eog_start, self.eog_size = feature_groups[eog_key]

        # 3) Build heads for each group
        self.heads = nn.ModuleDict({
            # EEG 2Hz head
            'eeg2': nn.Sequential(
                nn.Conv1d(self.slices['eeg2'][2], 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            ),
            # EEG 5Bands head
            'eeg5': nn.Sequential(
                nn.Conv1d(self.slices['eeg5'][2], 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            ),
            # Forehead 2Hz head
            'fore2': nn.Sequential(
                nn.Conv1d(self.slices['fore2'][2], 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            ),
            # Forehead 5Bands head
            'fore5': nn.Sequential(
                nn.Conv1d(self.slices['fore5'][2], 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            ),
            # EOG head
            'eog': nn.Sequential(
                nn.Linear(self.eog_size, 64),
                nn.ReLU()
            )
        })

        # 4) Classifier on concatenated head outputs
        total_feats = 32 + 32 + 16 + 16 + 64
        self.classifier = nn.Sequential(
            nn.Linear(total_feats, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

        # 5) Output activation
        if task_type == 'binary':
            self.output_activation = nn.Sigmoid()
        elif task_type == 'ternary':
            self.output_activation = None
        else:
            self.output_activation = nn.Sigmoid()

    def forward(self, x):
        b = x.size(0)
        feats = []

        for name, head in self.heads.items():
            if name == 'eog':
                # EOG: flat slice
                start, size = self.eog_start, self.eog_size
                xi = x[:, start:start+size]
            else:
                # Conv1d heads: reshape to (batch, channels, feat_dim)
                start, size, n_ch, feat_dim = self.slices[name]
                flat = x[:, start:start+size]
                xi = flat.view(b, n_ch, feat_dim)
            feats.append(head(xi))

        # Concatenate all head outputs
        combined = torch.cat(feats, dim=1)
        out = self.classifier(combined)

        if self.output_activation is not None:
            out = self.output_activation(out)
        return out





