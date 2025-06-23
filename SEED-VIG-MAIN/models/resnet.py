from imports import *
class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1   = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)                
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + residual)


class ResNet1DModel(nn.Module):
    def __init__(self, input_size, output_size, task_type='binary',
                 channels=128, num_blocks=2, dropout=0.4):
        super().__init__()
        self.task_type = task_type
        self.initial = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout)               
        )
        self.layers = nn.Sequential(*[
            ResBlock1D(channels, dropout=dropout) for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)    
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels, output_size)

        if task_type == 'binary':
            self.output_activation = nn.Sigmoid()
        elif task_type == 'ternary':
            self.output_activation = None
        else:
            self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.initial(x)
        x = self.layers(x)
        x = self.dropout(x)                  
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x