import torch
import torch.nn as nn

class LSTMExtractor(nn.Module):
    def __init__(self, input_size=60, hidden_size=7, num_layers=2, batch_first=True, dropout=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.LSTM = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=dropout)

    def forward(self, x):
        x = self.LSTM(x)
        return x[0][:, -1]