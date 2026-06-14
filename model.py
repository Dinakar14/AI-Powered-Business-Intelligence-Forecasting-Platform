import torch
import torch.nn as nn


class TicketLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        output_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.dropout = nn.Dropout(
            0.3
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            output_dim
        )

    def forward(self, x):

        embedded = self.embedding(x)

        _, (hidden, _) = self.lstm(
            embedded
        )

        forward_hidden = hidden[-2]

        backward_hidden = hidden[-1]

        hidden_cat = torch.cat(
            (
                forward_hidden,
                backward_hidden
            ),
            dim=1
        )

        hidden_cat = self.dropout(
            hidden_cat
        )

        output = self.fc(
            hidden_cat
        )

        return output