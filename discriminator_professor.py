import torch
import torch.nn as nn

import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, decoder_hidden_size, discriminator_hidden_size, discriminator_linear_size, discriminator_lin_dropout, use_cuda):
        super(Discriminator, self).__init__()

        self.discriminator_hidden_size = discriminator_hidden_size
        self.discriminator_linear_size = discriminator_linear_size
        self.device = use_cuda

        self.rnn = nn.GRU(
            decoder_hidden_size, discriminator_hidden_size,
            batch_first=True, bidirectional=True
        )

        self.linears = nn.Sequential(
            nn.Linear(discriminator_hidden_size * 2, discriminator_linear_size),
            nn.ReLU(),
            nn.Dropout(discriminator_lin_dropout),
            nn.Linear(discriminator_linear_size, discriminator_linear_size),
            nn.ReLU(),
            nn.Dropout(discriminator_lin_dropout),
            nn.Linear(discriminator_linear_size, 1)
        )

    def forward(self, hidden_states):
        # hidden_states                                           # [batch_size * seq_len * hid_size]
        batch_size = hidden_states.size(0)
        initial_hidden = self.init_hidden(hidden_states.size(0))
        _, rnn_final_hidden = self.rnn(
            hidden_states, initial_hidden)                        # [2 * batch_size * hid_size]
        rnn_final_hidden = rnn_final_hidden.view(batch_size, -1)  # [batch_size * (2 * discriminator_hidden_size)]
        scores = self.linears(rnn_final_hidden)                   # [batch_size * 1]
        scores = torch.sigmoid(scores)          # [batch_size * 1]
        return scores

    def init_hidden(self, batch_size):
        hidden = torch.zeros(2, batch_size, self.discriminator_hidden_size).cuda()
        return hidden