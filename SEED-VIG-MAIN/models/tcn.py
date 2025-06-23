from imports import *
class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, task_type='binary',
                 num_channels=(256, 128, 64), kernel_size=4, dropout=0.4, causal=False):
        super().__init__()
        self.task_type = task_type
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=list(num_channels),
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
            input_shape='NCL'
        )
        self.fc = nn.Linear(num_channels[-1], output_size)
        if task_type == 'binary':
            self.act = nn.Sigmoid()
        elif task_type == 'ternary':
            self.act = None
        else:
            self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.tcn(x.unsqueeze(2))
        out = out[:, :, -1]
        out = self.fc(out)
        if self.act:
            out = self.act(out)
        return out