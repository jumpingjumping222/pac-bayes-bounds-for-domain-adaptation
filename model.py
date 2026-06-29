import torch
import torch.nn as nn

class ResLayer(nn.Module):
    def __init__(self, h, act=None, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(h, h)
        self.act = act if act is not None else nn.LeakyReLU(0.01)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        z = self.drop(self.act(self.fc(x)))
        return x + z

class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=22, num_hidden=16, dropout=0.0):
        super().__init__()
        self.act = nn.LeakyReLU(0.01)
        self.inp = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            ResLayer(hidden_dim, act=self.act, dropout=dropout)
            for _ in range(num_hidden - 1)
        ])
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, X):
        h = self.act(self.inp(X))
        for layer in self.layers:
            h = layer(h)
        return self.out(h)



def init_like_pacbayes_pdf(model: ResMLP, std: float = 0.04) -> None:
    """
    Initialization used as DL prior in the experiment:
      - weights ~ N(0, std^2)
      - first layer bias = 0.1
      - remaining biases = 0
    """
    with torch.no_grad():
        nn.init.normal_(model.inp.weight, mean=0.0, std=std)
        for layer in model.layers:
            nn.init.normal_(layer.fc.weight, mean=0.0, std=std)
        nn.init.normal_(model.out.weight, mean=0.0, std=std)

        nn.init.constant_(model.inp.bias, 0.1)
        for layer in model.layers:
            nn.init.constant_(layer.fc.bias, 0.0)
        nn.init.constant_(model.out.bias, 0.0)


def flat_dim(input_dim: int, hidden_dim: int = 64) -> int:
    """Number of flattened parameters for the MLP architecture."""
    input_dim = int(input_dim)
    hidden_dim = int(hidden_dim)
    return (
        hidden_dim * input_dim
        + hidden_dim
        + hidden_dim * hidden_dim
        + hidden_dim
        + hidden_dim
        + 1
    )


FLAT_DIM = flat_dim(input_dim=2)


def unpack_params(theta: torch.Tensor, input_dim: int = 2, hidden_dim: int = 64):
    """Unpack a flat parameter vector into ToyMLP weights and biases."""
    input_dim = int(input_dim)
    hidden_dim = int(hidden_dim)
    expected = flat_dim(input_dim=input_dim, hidden_dim=hidden_dim)
    if theta.numel() != expected:
        raise ValueError(
            f"theta has {theta.numel()} parameters, expected {expected} "
            f"for input_dim={input_dim}, hidden_dim={hidden_dim}."
        )

    idx = 0

    w1 = theta[idx: idx + hidden_dim * input_dim].view(hidden_dim, input_dim)
    idx += hidden_dim * input_dim

    b1 = theta[idx: idx + hidden_dim]
    idx += hidden_dim

    w2 = theta[idx: idx + hidden_dim * hidden_dim].view(hidden_dim, hidden_dim)
    idx += hidden_dim * hidden_dim

    b2 = theta[idx: idx + hidden_dim]
    idx += hidden_dim

    w3 = theta[idx: idx + hidden_dim].view(1, hidden_dim)
    idx += hidden_dim

    b3 = theta[idx: idx + 1]
    idx += 1

    return w1, b1, w2, b2, w3, b3


def functional_forward(
    X: torch.Tensor,
    theta: torch.Tensor,
    input_dim: int = 2,
    hidden_dim: int = 64,
):
    """Forward pass using a flat parameter vector rather than an nn.Module."""
    w1, b1, w2, b2, w3, b3 = unpack_params(
        theta,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )
    x = F.linear(X, w1, b1)
    x = F.relu(x)
    x = F.linear(x, w2, b2)
    x = F.relu(x)
    x = F.linear(x, w3, b3)
    return x
