import numpy as np
import torch
from torch import Tensor, nn
from typing import Literal, assert_never

from gmst.contracts import FloatArray, Panel, Prediction
from gmst.hmm_forecast import HMMState, day_tensors, filter_last, issue_centre


@torch.no_grad()
def issue_flow(state: HMMState, panel: Panel, d: int) -> FloatArray:
    model = state['model']
    if model.ar or model.K != 3:
        raise ValueError('Transition-flow experiment requires AR-free B3 with K=3')
    m = issue_centre(state, panel, np.array([d], dtype=np.int64))
    q, _ = filter_last(model, panel, d, m, state['protocol'])
    z = day_tensors(model, panel, np.array([d], dtype=np.int64), m, state['protocol'])[0]
    transition = model.trans(z)[0]
    flow = torch.empty_like(transition)
    for h in range(96):
        joint = q[:, None] * transition[h]
        flow[h] = joint
        q = joint.sum(0)
    return np.asarray(flow.cpu().numpy(), dtype=np.float64)


type FlowMode = Literal['mlp', 'q', 'j', 'gated_j']


class FlowFiLM(nn.Module):
    def __init__(self, mode: FlowMode, scale: float) -> None:
        super().__init__()
        self.mode = mode
        self.scale = scale
        self.hidden = nn.Linear(15, 16)
        self.condition = nn.Linear(6, 32)
        self.output = nn.Linear(16, 1)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: Tensor, flow: Tensor) -> Tensor:
        directed = flow[:, ~torch.eye(3, dtype=torch.bool, device=flow.device)]
        match self.mode:
            case 'mlp':
                condition = torch.zeros_like(directed)
            case 'q':
                q = flow.sum(dim=1)
                condition = torch.cat((q, q), dim=1)
            case 'j' | 'gated_j':
                condition = directed
            case unreachable:
                assert_never(unreachable)
        gamma, beta = self.condition(condition).chunk(2, dim=1)
        hidden = torch.relu(self.hidden(x))
        delta = self.scale * torch.tanh(self.output((1 + gamma) * hidden + beta).squeeze(1))
        if self.mode == 'gated_j':
            return delta * directed.sum(dim=1)
        return delta


def shift_prediction(base: Prediction, delta: FloatArray, thresholds: FloatArray) -> Prediction:
    paths = base['paths']
    if paths is None or paths.ndim != 2 or paths.shape[1] != 96 or delta.shape != (96,) or not np.isfinite(delta).all():
        raise ValueError('Path correction requires finite 96-slot joint paths and delta')
    samples = paths + delta[None, :]
    maximum = samples.max(axis=1)
    risk = np.where(np.isfinite(thresholds), (maximum[:, None] > thresholds).mean(axis=0), np.nan)
    return {'y_mean': samples.mean(axis=0), 'y_median': np.median(samples, axis=0),
            'q': np.quantile(samples, np.arange(1, 20) / 20, axis=0), 'paths': samples,
            'M_hat_median': float(np.median(maximum)), 'M_hat_mean': float(maximum.mean()),
            'peak_time_mode': int(np.bincount(samples.argmax(axis=1), minlength=96).argmax()),
            'risk_raw': risk}
