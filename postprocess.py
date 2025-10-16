from run import load_loader
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def load_trained_model(run_dir, model_class, model_layers, device="cpu"):
    """
    Load only the trained model weights from a run directory.
    """
    model_path = os.path.join(run_dir, "best_model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = model_class(model_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loaded model from: {model_path}")
    return model


def load_training_history(run_dir, device="cpu"):
    """
    Load the saved training and validation loss history dictionary.
    """
    history_path = os.path.join(run_dir, "training_history.pt")

    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found at {history_path}")

    history = torch.load(history_path, map_location=device)
    print(f"Loaded training history from: {history_path}")
    return history


def load_test_predictions(run_dir, device="cpu"):
    """
    Load the saved test prediction dictionary from a given run directory.

    Expected file: test_prediction.pt
    Returns the loaded dictionary with model predictions and test inputs.
    """
    pred_path = os.path.join(run_dir, "test_predictions.pt")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found at {pred_path}")

    predictions = torch.load(pred_path, map_location=device)
    print(f"Loaded test predictions from: {pred_path}")

    # Optional: summarize contents
    if isinstance(predictions, dict):
        print("Contents:")
        for k, v in predictions.items():
            if torch.is_tensor(v):
                print(f"  {k:<15s}: tensor {tuple(v.shape)}")
            else:
                print(f"  {k:<15s}: {type(v)}")

    return predictions


def plot_training_results(history, n_epochs, print_every, save_dir=None, show=True):
    """
    Plot training and validation losses from history in a 2x2 grid:
      (1,1) D loss
      (1,2) u, v losses
      (2,1) f (PDE residual) loss
      (2,2) total loss
    """
    epochs = np.arange(1, n_epochs + 1)
    val_epochs = np.arange(0, len(history["val_total"])) * print_every + 1
    best_epoch = history.get("best_epoch", None)

    def _mark_best(ax):
        if best_epoch is not None:
            ax.axvline(best_epoch, color="k", linestyle="--", linewidth=1.2, label=f"Best epoch {best_epoch}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.flatten()  # easier indexing

    # --- (1,1) D loss ---
    ax = axes[0]
    ax.plot(epochs, history["D"], label="Train D")
    ax.plot(val_epochs, history["val_D"], "o-", label="Val D", markersize=3)
    _mark_best(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("D Loss")
    ax.legend()
    ax.grid(True)

    # --- (1,2) u, v losses ---
    ax = axes[1]
    ax.plot(epochs, history["u"], label="Train u")
    ax.plot(epochs, history["v"], label="Train v")
    ax.plot(val_epochs, history["val_u"], "o-", label="Val u", markersize=3)
    ax.plot(val_epochs, history["val_v"], "o-", label="Val v", markersize=3)
    _mark_best(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("u, v Losses")
    ax.legend()
    ax.grid(True)

    # --- (2,1) f loss ---
    ax = axes[2]
    ax.plot(epochs, history["f"], label="Train f")
    ax.plot(val_epochs, history["val_f"], "o-", label="Val f", markersize=3)
    _mark_best(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("f (PDE residual) Loss")
    ax.legend()
    ax.grid(True)

    # --- (2,2) total loss ---
    ax = axes[3]
    ax.plot(epochs, history["total"], label="Train total")
    ax.plot(val_epochs, history["val_total"], "o-", label="Val total", markersize=3)
    _mark_best(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "training_losses_grid.png"), dpi=250, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def get_Duv_from_predictions(predictions):
    """
    Given a predictions dictionary (from load_test_predictions),
    return a torch tensor with D, u, v concatenated along the last dimension.
    Shape: (N, 3)
    """
    required_keys = ["D_pred", "u_pred", "v_pred"]
    missing = [k for k in required_keys if k not in predictions]
    if missing:
        raise KeyError(f"Missing keys in predictions: {missing}")

    D = predictions["D_pred"]
    u = predictions["u_pred"]
    v = predictions["v_pred"]

    if not (torch.is_tensor(D) and torch.is_tensor(u) and torch.is_tensor(v)):
        raise TypeError("D_pred, u_pred, v_pred must all be torch tensors")

    Duv_pred = torch.hstack((D, u, v))

    D_true = predictions["D_true"]
    u_true = predictions["u_true"]
    v_true = predictions["v_true"]

    if not (torch.is_tensor(D) and torch.is_tensor(u) and torch.is_tensor(v)):
        raise TypeError("D_true, u_true, v_pred must all be torch tensors")

    Duv_true = torch.hstack((D_true, u_true, v_true))

    return Duv_pred, Duv_true


def plot_true_vs_pred(Duv_true, Duv_pred, save_path=None, show=True):
    """
    Plot correlation (true vs predicted) for D, u, and v.

    Args:
        Duv_true (np.ndarray): Array of shape (N, 3) containing true values [D, u, v].
        Duv_pred (np.ndarray): Array of shape (N, 3) containing predicted values [D, u, v].
        save_path (str, optional): Path to save the figure.
        show (bool): Whether to display the plot.
    """
    if Duv_true.shape != Duv_pred.shape or Duv_true.shape[1] != 3:
        raise ValueError("Inputs must have shape (N, 3) for [D, u, v].")

    labels = ["D (NO₂)", "u (wind x)", "v (wind y)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, ax in enumerate(axes):
        true = Duv_true[:, i]   # <-- fixed (removed comma)
        pred = Duv_pred[:, i]   # <-- fixed
        r = np.corrcoef(true, pred)[0, 1]

        ax.scatter(true, pred, s=12, alpha=0.6, edgecolor="none")
        min_val, max_val = np.min([true.min(), pred.min()]), np.max([true.max(), pred.max()])
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)

        ax.set_xlabel(f"True {labels[i]}")
        ax.set_ylabel(f"Predicted {labels[i]}")
        ax.set_title(f"{labels[i]} (r = {r:.3f})")
        ax.grid(True, linestyle=":", linewidth=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_concentration_wind_predictions(run_dir, n_colloc, save_path=None):
    predictions = load_test_predictions(run_dir=run_dir)

    Duv_pred, Duv_true = get_Duv_from_predictions(predictions)

    loader = load_loader(n_colloc)
    Duv_pred_rescaled = loader.inverse_scale_predictions(Duv_pred)
    Duv_true_rescaled = loader.inverse_scale_predictions(Duv_true)

    plot_true_vs_pred(Duv_true_rescaled, Duv_pred_rescaled, save_path=save_path)