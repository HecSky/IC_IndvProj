import torch
import torch.nn as nn

class CNNLSTMExtractor(nn.Module):
    def __init__(self, input_size=192, hidden_size=7, num_layers=2, batch_first=True, dropout=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cnn_1_1 = nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2))
        self.cnn_1_2_1 = nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(1, 1), padding=(1, 0))
        self.cnn_1_2_2 = nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(1, 1), padding=(2, 0))
        self.cnn_2_1 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2))
        self.cnn_2_2_1 = nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(1, 1), padding=(1, 0))
        self.cnn_2_2_2 = nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(1, 1), padding=(2, 0))
        self.cnn_3 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 15))
        self.cnn_4_1_1 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        self.cnn_4_1_2 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.cnn_4_2_1 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        self.cnn_4_2_2 = nn.Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        self.cnn_4_3_1 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        self.cnn_4_3_2 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))

        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=0.05)

        self.LSTM = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=dropout)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.activation(self.cnn_1_1(x))
        x = self.activation(self.cnn_1_2_1(x))
        x = self.activation(self.dropout(self.cnn_1_2_2(x)))
        x = self.activation(self.cnn_2_1(x))
        x = self.activation(self.cnn_2_2_1(x))
        x = self.activation(self.cnn_2_2_2(x))
        x = self.activation(self.dropout(self.cnn_3(x)))
        tensor1 = self.activation(self.cnn_4_1_1(x))
        tensor1 = self.activation(self.cnn_4_1_2(tensor1))
        tensor2 = self.activation(self.cnn_4_2_1(x))
        tensor2 = self.activation(self.cnn_4_2_2(tensor2))
        tensor3 = self.activation(self.cnn_4_3_1(x))
        tensor3 = self.activation(self.cnn_4_3_2(tensor3))
        tensor = torch.cat((tensor1, tensor2, tensor3), dim=1)
        tensor = torch.squeeze(tensor)
        x = torch.permute(tensor, dims=(0, 2, 1))
        x = self.dropout(x)
        x = self.LSTM(x)
        return x[0][:, -1]