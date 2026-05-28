"""
pinn.py
-------
Physics-Informed Neural Network for caking strength regression.
All code extracted verbatim from caking-prediction.ipynb (Section 8).

Architecture : Linear(input,128)→SiLU → Linear(128,64)→SiLU →
               Linear(64,32)→SiLU → Linear(32,1)
Physics loss : MSE(dCI/dt, Arrhenius rate) + ReLU(−dCI/dt) monotonicity penalty
Target       : StandardScaler-normalised caking_strength_Pa
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ── Architecture ─────────────────────────────────────────────────────────────

class CakingPINN(nn.Module):
    """
    MLP with SiLU activations.
    SiLU is required (not ReLU) because the physics loss uses autograd to
    compute dCI/dt — ReLU's zero gradient at negative values would produce
    zero physics residuals for half the inputs.
    Extracted verbatim from notebook Section 8.
    """
    def __init__(self, input_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.SiLU()]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ── Physics Loss ─────────────────────────────────────────────────────────────

def sintering_physics_loss(
    model: CakingPINN,
    X_batch: torch.Tensor,
    T_idx: int,
    t_idx: int,
    Ea: float = 80000,
    R: float = 8.314,
) -> torch.Tensor:
    """
    Combined physics residual loss.
    Extracted verbatim from notebook Section 8.

    Components
    ----------
    1. MSE(dCI/dt, Arrhenius rate)   — sintering rate law residual
    2. mean(ReLU(−dCI/dt))           — monotonicity penalty (caking irreversible)

    T_K unnormalisation: X_batch[:, T_idx] * 15 + 310
    (linear approximation mapping Yeo-Johnson-scaled Temp_C back to ~[283,333] K)
    """
    X_t  = X_batch.clone().detach().requires_grad_(True)
    pred = model(X_t)

    # dCI/dt  (gradient w.r.t. time feature in scaled input space)
    grad_t = torch.autograd.grad(
        pred.sum(), X_t, create_graph=True
    )[0][:, t_idx]

    # Approximate Arrhenius rate in original Kelvin space
    T_K  = X_batch[:, T_idx] * 15 + 310
    rate = torch.exp(-torch.tensor(Ea / R) / T_K)

    mse_residual       = nn.MSELoss()(grad_t, rate)
    monotonicity_penalty = torch.mean(torch.relu(-grad_t))

    return mse_residual + monotonicity_penalty


# ── Training ──────────────────────────────────────────────────────────────────

def train_pinn(
    X_train_s: np.ndarray,
    y_reg_train: np.ndarray,
    T_IDX: int,
    t_IDX: int,
    epochs: int = 200,
    batch_size: int = 64,
    lambda_phys: float = 0.5,
    lr: float = 1e-3,
    random_state: int = 42,
) -> tuple:
    """
    Full PINN training loop extracted from notebook Section 8.

    Returns
    -------
    pinn          : trained CakingPINN
    target_scaler : fitted StandardScaler (inverse-transform predictions to Pa)
    train_losses  : list of per-epoch data loss
    phys_losses   : list of per-epoch physics loss
    """
    torch.manual_seed(random_state)

    # --- 1. Target scaling ---
    target_scaler = StandardScaler()
    yr_scaled = target_scaler.fit_transform(
        y_reg_train.reshape(-1, 1)
    ).flatten()

    # --- 2. Tensors ---
    X_tr_t  = torch.FloatTensor(X_train_s)
    yr_tr_t = torch.FloatTensor(yr_scaled)

    # --- 3. Model, optimiser ---
    pinn      = CakingPINN(input_dim=X_tr_t.shape[1])
    optimizer = optim.Adam(pinn.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_tr_t, yr_tr_t)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    train_losses, phys_losses = [], []

    pinn.train()
    for _ in range(epochs):
        epoch_data, epoch_phys = 0.0, 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred      = pinn(xb)
            data_loss = nn.MSELoss()(pred, yb)
            phys_loss = sintering_physics_loss(pinn, xb, T_IDX, t_IDX)
            total     = data_loss + lambda_phys * phys_loss
            total.backward()
            optimizer.step()
            epoch_data += data_loss.item()
            epoch_phys += phys_loss.item()
        train_losses.append(epoch_data / len(loader))
        phys_losses.append(epoch_phys / len(loader))

    return pinn, target_scaler, train_losses, phys_losses


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_pinn(
    pinn: CakingPINN,
    target_scaler: StandardScaler,
    X_scaled: np.ndarray,
) -> np.ndarray:
    """
    Run PINN inference and inverse-transform predictions back to Pa.
    Extracted from notebook Section 8, inference block.
    """
    pinn.eval()
    X_t = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        pred_scaled = pinn(X_t).numpy()
    return target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()
