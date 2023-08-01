import torch
import torch.nn as nn


class TransformerEncoderExtractor(nn.Module):
    def __init__(self, d_model=60, nhead=6, dim_feedforward=256, num_layers=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=nhead,
                                                        dim_feedforward=dim_feedforward,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                         num_layers=num_layers)

        self.activation = nn.PReLU()
        self.output_layer = nn.Linear(60, 7)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.activation(x)
        out = self.output_layer(x)
        return out
