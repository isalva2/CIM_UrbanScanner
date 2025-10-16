import numpy as np
from data import read_r3, read_fishnet, read_source_data
from preprocess import windspeed_decomp_preprocess, prepare_training_dataset_metric, CDiffDataLoader
from model import PINN, PINNLoss, train
from datetime import datetime
import os
import sys
import logging

RANDOM_STATE=42
np.random.seed(RANDOM_STATE)
TEST_DAY = "2021-05-22"

TEST_SIZE = 0.50
N_COLLOC = 20_000
LAYERS = [3, 30, 30, 30, 30, 3]

# K = 2.3e-9 # https://www.engineeringtoolbox.com/diffusion-coefficients-d_1404.html
K = 0.5 # literature
LAMBDAd = 1e1
LAMBDAu = 1e1
LAMBDAv = 1e1
LAMBDAf = 1e-1

EPOCHS = 2500
LR = 1e-3
PRINTEV = 500

DEVICE = "mps"

def load_loader(n_colloc:int):
    # load data
    wind_speeds = read_r3()
    fishnet_gdf = read_fishnet(verbose=True).get("FishCenterJoin.shp")
    S_df, key_df = read_source_data()

    # preprocess data
    wind_speeds_decomposed = windspeed_decomp_preprocess(wind_speeds)
    dataset_model_metric = prepare_training_dataset_metric(wind_speeds_decomposed, fishnet_gdf, S_df, key_df, analysis_date=TEST_DAY)
    loader = CDiffDataLoader(
        dataset_model_metric,
        fishnet_gdf,
        n_collocation=n_colloc,
        device=DEVICE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    ).prepare_data().preprocess()
    loader.summary(verbose=True)

    return loader


def setup_logging(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")

    # Configure logging: both file + console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Redirect all print() calls into logging
    class PrintLogger:
        def __init__(self, logger, level=logging.INFO):
            self.logger = logger
            self.level = level
        def write(self, message):
            # tqdm writes carriage returns "\r" — ignore them
            if message and not message.startswith("\r") and message.strip():
                self.logger.log(self.level, message.strip())
        def flush(self):
            pass

    sys.stdout = PrintLogger(logging.getLogger(), logging.INFO)
    sys.stderr = PrintLogger(logging.getLogger(), logging.ERROR)

    print(f"Logging to: {log_path}\n")
    return log_path


def full_train():
    # === Create run dir and set up logging ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_dir = os.path.join("run", f"model-{timestamp}")
    log_path = setup_logging(save_dir)

    # === Load and preprocess data ===
    loader = load_loader(n_colloc=N_COLLOC)
    x_min, y_min, t_min = loader.scaler_xyt.data_min_
    x_max, y_max, t_max = loader.scaler_xyt.data_max_
    scales = {"x": x_max - x_min, "y": y_max - y_min, "t": t_max - t_min}

    # === Build model & loss ===
    print(f"NN Layers:{LAYERS}")
    model = PINN(layers=LAYERS).to(DEVICE)
    loss_fn = PINNLoss(
        K,
        lambda_D=LAMBDAd,
        lambda_u=LAMBDAu,
        lambda_v=LAMBDAv,
        lambda_f=LAMBDAf,
        device=DEVICE
    )

    # === Train ===
    history = train(
        model=model,
        data_loader=loader,
        loss_fn=loss_fn,
        n_epochs=EPOCHS,
        lr=LR,
        print_every=PRINTEV,
        scales=scales,
        device=DEVICE,
        save_dir=save_dir,
    )

    print("\nTraining finished successfully.")
    print(f"Artifacts saved in: {save_dir}")
    print(f"Full log written to: {log_path}")

    return history

if __name__ == "__main__":
    full_train()