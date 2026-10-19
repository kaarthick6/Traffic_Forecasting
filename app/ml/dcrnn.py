"""PyTorch bidirectional diffusion recurrent encoder-decoder."""
from __future__ import annotations
import torch
from torch import nn


class DiffusionConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, order: int):
        super().__init__(); self.order = order
        self.linear = nn.Linear(in_channels * (1 + 2 * order), out_channels)

    def forward(self, x: torch.Tensor, supports: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # x: [batch, nodes, channels]; supports: [order+1, nodes, nodes]
        f, b = supports
        if x.ndim != 3 or f.shape[1:] != (x.shape[1], x.shape[1]) or b.shape != f.shape:
            raise ValueError("DiffusionConv input/support dimensions do not match.")
        terms = [x]
        for k in range(1, self.order + 1):
            terms.extend([torch.einsum("ij,bjc->bic", f[k], x), torch.einsum("ij,bjc->bic", b[k], x)])
        return self.linear(torch.cat(terms, dim=-1))


class DCGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, order: int):
        super().__init__(); self.hidden_size = hidden_size
        self.gates = DiffusionConv(input_size + hidden_size, hidden_size * 2, order)
        self.candidate = DiffusionConv(input_size + hidden_size, hidden_size, order)

    def forward(self, x: torch.Tensor, h: torch.Tensor, supports: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], -1), supports)).chunk(2, -1)
        candidate = torch.tanh(self.candidate(torch.cat([x, r * h], -1), supports))
        return z * h + (1 - z) * candidate


class DCRNN(nn.Module):
    """Input [B,T,N,F], output [B,H,N,F]. Autoregressive inference."""
    def __init__(self, num_nodes: int, input_dim: int = 1, output_dim: int = 1, hidden_channels: int = 64, diffusion_order: int = 2, encoder_layers: int = 2, decoder_layers: int = 2, horizon: int = 12):
        super().__init__()
        if min(num_nodes, input_dim, output_dim, hidden_channels, encoder_layers, decoder_layers, horizon) < 1:
            raise ValueError("Model dimensions must be positive.")
        self.num_nodes=num_nodes; self.input_dim=input_dim; self.output_dim=output_dim; self.hidden=hidden_channels; self.horizon=horizon
        self.encoder = nn.ModuleList([DCGRUCell(input_dim if i == 0 else hidden_channels, hidden_channels, diffusion_order) for i in range(encoder_layers)])
        self.decoder = nn.ModuleList([DCGRUCell(output_dim if i == 0 else hidden_channels, hidden_channels, diffusion_order) for i in range(decoder_layers)])
        self.output = nn.Linear(hidden_channels, output_dim)

    def forward(self, x: torch.Tensor, supports: tuple[torch.Tensor, torch.Tensor], targets: torch.Tensor | None = None, teacher_forcing: float = 0.0) -> torch.Tensor:
        if x.ndim != 4 or x.shape[2] != self.num_nodes or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [batch,time,{self.num_nodes},{self.input_dim}], got {tuple(x.shape)}")
        b, t, n, _ = x.shape
        states = [x.new_zeros((b,n,self.hidden)) for _ in self.encoder]
        for step in range(t):
            value=x[:,step]
            for layer, cell in enumerate(self.encoder):
                states[layer]=cell(value,states[layer],supports); value=states[layer]
        dec_states=[x.new_zeros((b,n,self.hidden)) for _ in self.decoder]
        for i in range(min(len(dec_states),len(states))): dec_states[i]=states[i]
        decoder_input=x.new_zeros((b,n,self.output_dim)); outputs=[]
        horizon = targets.shape[1] if targets is not None else self.horizon
        for step in range(horizon):
            value=decoder_input
            for layer, cell in enumerate(self.decoder):
                dec_states[layer]=cell(value,dec_states[layer],supports); value=dec_states[layer]
            pred=self.output(value); outputs.append(pred)
            if targets is not None and teacher_forcing > 0 and torch.rand(()) < teacher_forcing:
                decoder_input=targets[:,step]
            else: decoder_input=pred
        return torch.stack(outputs,dim=1)


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    valid = torch.isfinite(target) if mask is None else (mask.bool() & torch.isfinite(target))
    if not valid.any(): return prediction.sum() * 0
    return (prediction[valid] - target[valid]).abs().mean()

