import torch
import torch.nn as nn

class sigmod(nn.Module):
    def __init__(self, max_words=20000, emb_size=100, num_of_classes=2, hidden_size=64, dropout=0):
        super(sigmod, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.max_length = 100

        self.Embedding = nn.Embedding(self.max_words, self.emb_size)

        self.cell_linear_in = nn.Linear(self.emb_size, hidden_size * 3)
        self.cell_linear_hidden = nn.Linear(hidden_size, hidden_size * 3)

        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, num_of_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.Embedding(x)  # [batch_size, seq_len, emb_size]

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(x.shape) != 3:
            raise ValueError('Input shape should be [batch, seq_len, emb_dim]')

        hid_i = torch.zeros(x.size(0), self.hidden_size).to(x.device)

        for l in range(x.shape[1]):
            if l >= self.max_length:
                break
            tmp_i = self.cell_linear_in(x[:, l, :])
            tmp_h = self.cell_linear_hidden(hid_i)

            r_i, z_i, n_i = torch.chunk(tmp_i, 3, dim=1)
            r_h, z_h, n_h = torch.chunk(tmp_h, 3, dim=1)

            r_t = torch.sigmoid(r_i + r_h)
            z_t = torch.sigmoid(z_i + z_h)
            n_t = torch.tanh(n_i + r_t * n_h)
            hid_i = (1 - z_t) * n_t + z_t * hid_i

        x = self.fc1(hid_i)
        x = self.tanh(x)
        x = self.fc2(x)
        return x


class Tanh(nn.Module):
    def __init__(self, max_words=20000, emb_size=100, num_of_classes=2, hidden_size=64, dropout=0):
        super(sigmod, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.max_length = 100

        self.Embedding = nn.Embedding(self.max_words, self.emb_size)

        self.cell_linear_in = nn.Linear(self.emb_size, hidden_size * 3)
        self.cell_linear_hidden = nn.Linear(hidden_size, hidden_size * 3)

        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, num_of_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.Embedding(x)  # [batch_size, seq_len, emb_size]

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(x.shape) != 3:
            raise ValueError('Input shape should be [batch, seq_len, emb_dim]')

        hid_i = torch.zeros(x.size(0), self.hidden_size).to(x.device)

        for l in range(x.shape[1]):
            if l >= self.max_length:
                break
            tmp_i = self.cell_linear_in(x[:, l, :])
            tmp_h = self.cell_linear_hidden(hid_i)

            r_i, z_i, n_i = torch.chunk(tmp_i, 3, dim=1)
            r_h, z_h, n_h = torch.chunk(tmp_h, 3, dim=1)

            r_t = torch.tanh(r_i + r_h)
            z_t = torch.tanh(z_i + z_h)
            n_t = torch.tanh(n_i + r_t * n_h)
            hid_i = (1 - z_t) * n_t + z_t * hid_i

        x = self.fc1(hid_i)
        x = self.tanh(x)
        x = self.fc2(x)
        return x