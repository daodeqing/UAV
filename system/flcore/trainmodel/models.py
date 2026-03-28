import torch
import torch.nn.functional as F
from torch import nn


class BaseHeadSplit(nn.Module):
    def __init__(self, base, head, adapter=None):
        super(BaseHeadSplit, self).__init__()
        self.base = base
        self.adapter = adapter if adapter is not None else nn.Identity()
        self.head = head

    def forward(self, x):
        rep = self.base(x)
        rep = self.adapter(rep)
        out = self.head(rep)
        return out


class ResidualAdapter(nn.Module):
    """A lightweight bottleneck adapter for cluster-level personalization."""

    def __init__(self, dim: int, bottleneck_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        bottleneck_dim = max(1, int(bottleneck_dim))
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = F.gelu(self.down(out))
        out = self.dropout(out)
        out = self.up(out)
        return residual + self.scale * out


class CNN1D(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, dropout=0.2, num_classes=10):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = self.global_maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        if self.conv1.padding[0] > 0:
            out = out[:, :, :-self.conv1.padding[0]]
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        if self.conv2.padding[0] > 0:
            out = out[:, :, :-self.conv2.padding[0]]
        out = F.gelu(out)
        out = self.dropout(out)
        return out + self.downsample(x)


class TCN1D(nn.Module):
    def __init__(self, hidden_dim, num_layers=3, dropout=0.2, num_classes=10, kernel_size=3):
        super().__init__()
        channels = [hidden_dim * (2 ** idx) for idx in range(max(1, num_layers))]
        blocks = []
        in_ch = 1
        for layer_idx, out_ch in enumerate(channels):
            blocks.append(
                _TemporalBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    dilation=2 ** layer_idx,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
