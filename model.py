import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from tqdm import trange
from datetime import datetime
import os
import sys

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def build_mlp(layer_sizes, activation=Sin):
    layers = []
    for i in range(len(layer_sizes)-1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes)-2:  # no activation after last layer
            layers.append(activation())
    return nn.Sequential(*layers)

class PINN(nn.Module):
    def __init__(self, layers, activation=Sin):
        super().__init__()
        self.net = build_mlp(layers, activation=activation)

        # Weight init (Xavier)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        # x: (N, 3) -> returns (N, 3)
        return self.net(x)

    def predict_fields(self, xyt):
        out = self.forward(xyt)
        D = out[:, 0:1]
        u = out[:, 1:2]
        v = out[:, 2:3]
        return D, u, v


class PINNLoss:
    def __init__(
        self,
        K,
        lambda_D,
        lambda_u,
        lambda_v,
        lambda_f,
        device="cpu"
    ):
        self.K = K  # Diffusion coefficient
        self.lambda_D = lambda_D
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_f = lambda_f
        self.device = device


    def pde_residual(self, model, xyt, S=None, scales=None):
        """
        Compute PDE residual f(x,y,t) for:
            D_t = K( D_xx + D_yy ) - (u D_x + v D_y) - D(u_x + v_y) + S

        If S is None assumes source-free PDE (for collocation points).
        """
        xyt = xyt.clone().detach().to(self.device).requires_grad_(True)
        D, u, v = model.predict_fields(xyt)

        # Compute gradients
        def grad(f):
            return autograd.grad(
                f, xyt, torch.ones_like(f),
                retain_graph=True, create_graph=True
            )[0]

        D_grads = grad(D)
        D_t  = D_grads[:, 2:3]
        D_x  = D_grads[:, 0:1]
        D_y  = D_grads[:, 1:2]
        D_xx = grad(D_x)[:, 0:1]
        D_yy = grad(D_y)[:, 1:2]

        u_x = grad(u)[:, 0:1]
        v_y = grad(v)[:, 1:2]

        # --- Use actual source term if provided ---
        if S is None:
            S = torch.zeros_like(D).to(self.device)

        # --- Apply scaling corrections ---
        if scales is not None:
            x_scale = scales["x"]
            y_scale = scales["y"]
            t_scale = scales["t"]

            D_t  = D_t / t_scale
            D_x  = D_x / x_scale
            D_y  = D_y / y_scale
            D_xx = D_xx / (x_scale ** 2)
            D_yy = D_yy / (y_scale ** 2)
            u_x  = u_x / x_scale
            v_y  = v_y / y_scale

        # PDE residual (scaled to real units)
        f = (
            self.K * (D_xx + D_yy)
            - D * (u_x + v_y)
            - (u * D_x)
            - (v * D_y)
            + S
            - D_t


        )
        return f


    def total_loss(self, model, xyt_data, D_data, u_data, v_data, S_data, xyt_f, scales=None):
        """Compute total PINN loss = data loss + PDE residual loss."""
        # Predictions for data points
        D_pred, u_pred, v_pred = model.predict_fields(xyt_data)

        # PDE residual for data points (with source term)
        f_data = self.pde_residual(model, xyt_data, S=S_data, scales=scales)

        # PDE residual for collocation points (no source term)
        f_colloc = self.pde_residual(model, xyt_f, S=None, scales=scales)

        # Data losses
        loss_D = torch.mean((D_pred - D_data) ** 2)
        loss_u = torch.mean((u_pred - u_data) ** 2)
        loss_v = torch.mean((v_pred - v_data) ** 2)

        # PDE loss (combine data + collocation)
        # loss_f = 0.5 * (torch.mean(f_data ** 2) + torch.mean(f_colloc ** 2))
        loss_f = torch.mean(f_colloc ** 2)

        # Total weighted loss
        total = (
            self.lambda_D * loss_D
            + self.lambda_u * loss_u
            + self.lambda_v * loss_v
            + self.lambda_f * loss_f
        )

        return total, (loss_D, loss_u, loss_v, loss_f)


# def train(
#     model,
#     data_loader,
#     loss_fn,
#     n_epochs=5000,
#     lr=1e-3,
#     print_every=500,
#     scales=None,
#     device="cpu",
#     save_dir=None,  # <-- now passed from run.py
# ):
#     """
#     Train the PINN model using Adam optimizer with data and PDE residual losses.
#     Returns full training history.
#     """

#     # === Load preprocessed tensors ===
#     T_D_train, T_D_test, T_f = data_loader.get_train_test_data()

#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     history = {
#         "total": [], "D": [], "u": [], "v": [], "f": [],
#         "val_total": [], "val_D": [], "val_u": [], "val_v": [], "val_f": []
#     }

#     model.to(device)
#     model.train()

#     if save_dir is None:
#         save_dir = "run/temp"
#     os.makedirs(save_dir, exist_ok=True)

#     best_val_loss = float("inf")
#     best_epoch = 0

#     print("=" * 60)
#     print("Training PINN Model")
#     print("=" * 60)
#     print(f"Device:            {device}")
#     print(f"Epochs:            {n_epochs}")
#     print(f"Learning rate:     {lr}")
#     print(f"Print interval:    {print_every}")
#     print(f"Diffusion coeff K: {loss_fn.K}")
#     print(f"λ_D={loss_fn.lambda_D}, λ_u={loss_fn.lambda_u}, λ_v={loss_fn.lambda_v}, λ_f={loss_fn.lambda_f}")
#     print("-" * 60)
#     print(f"Train samples:     {T_D_train['xyt'].shape[0]}")
#     print(f"Test samples:      {T_D_test['xyt'].shape[0]}")
#     print(f"Collocation pts:   {T_f.shape[0]}")
#     print(f"Run directory:     {save_dir}")
#     print("=" * 60, "\n")

#     pbar = trange(n_epochs, desc="Training PINN", leave=True)
#     for epoch in pbar:
#         optimizer.zero_grad()

#         total_loss, (loss_D, loss_u, loss_v, loss_f) = loss_fn.total_loss(
#             model,
#             T_D_train["xyt"],
#             T_D_train["D"],
#             T_D_train["u"],
#             T_D_train["v"],
#             T_D_train["S"],
#             T_f,
#             scales=scales,
#         )

#         total_loss.backward()
#         optimizer.step()

#         history["total"].append(total_loss.item())
#         history["D"].append(loss_D.item())
#         history["u"].append(loss_u.item())
#         history["v"].append(loss_v.item())
#         history["f"].append(loss_f.item())

#         # --- Validation ---
#         if (epoch + 1) % print_every == 0 or epoch == 0:
#             model.eval()
#             with torch.no_grad():
#                 D_pred, u_pred, v_pred = model.predict_fields(T_D_test["xyt"])
#                 val_D = torch.mean((D_pred - T_D_test["D"]) ** 2)
#                 val_u = torch.mean((u_pred - T_D_test["u"]) ** 2)
#                 val_v = torch.mean((v_pred - T_D_test["v"]) ** 2)

#             T_f_grad = T_f.clone().detach().requires_grad_(True)
#             f_val = loss_fn.pde_residual(model, T_f_grad, S=None, scales=scales)
#             val_f = torch.mean(f_val ** 2)

#             val_total = (
#                 loss_fn.lambda_D * val_D
#                 + loss_fn.lambda_u * val_u
#                 + loss_fn.lambda_v * val_v
#                 + loss_fn.lambda_f * val_f
#             )

#             history["val_total"].append(val_total.item())
#             history["val_D"].append(val_D.item())
#             history["val_u"].append(val_u.item())
#             history["val_v"].append(val_v.item())
#             history["val_f"].append(val_f.item())

#             # Save best model
#             if val_total.item() < best_val_loss:
#                 best_val_loss = val_total.item()
#                 best_epoch = epoch + 1
#                 torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

#             pbar.set_postfix({
#                 "train_total": f"{total_loss.item():.3e}",
#                 "L_D": f"{loss_D.item():.3e}",
#                 "L_u": f"{loss_u.item():.3e}",
#                 "L_v": f"{loss_v.item():.3e}",
#                 "L_f": f"{loss_f.item():.3e}",
#                 "val_total": f"{val_total.item():.3e}",
#                 "val_f": f"{val_f.item():.3e}"
#             })

#             print(f"[Epoch {epoch+1}/{n_epochs}] "
#                   f"train_total={total_loss.item():.3e}, "
#                   f"L_D={loss_D.item():.3e}, L_u={loss_u.item():.3e}, "
#                   f"L_v={loss_v.item():.3e}, L_f={loss_f.item():.3e}, "
#                   f"val_total={val_total.item():.3e}, val_f={val_f.item():.3e}")

#             model.train()

#     # === Save final model ===
#     torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))

#     # === Reload best model for test prediction ===
#     model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
#     model.eval()
#     print(f"\nReloaded best model (epoch {best_epoch}) for test prediction.")

#     with torch.no_grad():
#         D_pred, u_pred, v_pred = model.predict_fields(T_D_test["xyt"])

#     preds = {
#         "D_true": T_D_test["D"].cpu(),
#         "u_true": T_D_test["u"].cpu(),
#         "v_true": T_D_test["v"].cpu(),
#         "D_pred": D_pred.cpu(),
#         "u_pred": u_pred.cpu(),
#         "v_pred": v_pred.cpu(),
#     }
#     torch.save(preds, os.path.join(save_dir, "test_predictions.pt"))

#     print("\nTraining completed.")
#     print(f"Best validation total loss: {best_val_loss:.4e} at epoch {best_epoch}")

#     history["best_epoch"] = best_epoch
#     torch.save(history, os.path.join(save_dir, "training_history.pt"))

#     return history

