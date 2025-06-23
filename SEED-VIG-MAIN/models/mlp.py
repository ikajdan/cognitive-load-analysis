
import torch
import torch.nn as nn
class MLPModel(nn.Module):
    def __init__(self, input_size, output_size, task_type='binary'):
        super(MLPModel, self).__init__()
        
        self.task_type = task_type
        
     
        hidden1 = 256
        hidden2 = 128
        hidden3 = 64

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(0.4),
            
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden3),
            nn.Dropout(0.2),
            
            nn.Linear(hidden3, output_size)
        )
        

        if task_type == 'binary':
            self.output_activation = nn.Sigmoid()
        elif task_type == 'ternary':
            self.output_activation = None  
        else:  # continuous regression
            self.output_activation = nn.Sigmoid()  
    
    def forward(self, x):
        x = self.layers(x)
        
        if self.output_activation is not None:
            x = self.output_activation(x)
            
        return x

