# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Summarizer_transformer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        heads=8
        enc_layers=6
        dropout=0.1
        self.d_model = hidden_size
        self.pos_enc = PositionalEncoding(input_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=enc_layers
        )
        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, image_features):

        # [seq_len, D]
        feat1 = image_features.squeeze(1)
        video_emb  = feat1.unsqueeze(0)
        input_emb = self.pos_enc(video_emb)
        video_emb = self.transformer_encoder(input_emb)
        video_emb = video_emb.contiguous().view(-1, self.d_model)
        logits = self.fc(video_emb)

        return logits


if __name__ == '__main__':

    pass
