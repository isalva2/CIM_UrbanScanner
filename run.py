import numpy as np
from data import read_r3, read_fishnet, read_source_data
from preprocess import windspeed_decomp_preprocess, prepare_training_dataset_metric, CDiffDataLoader
from model import Sin, PINN, PINNLoss, train

RANDOM_STATE=1
np.random.seed(RANDOM_STATE)

TEST_SIZE = 0.50
N_COLLOC = 15_000
LAYERS = [3, 32, 64, 128, 256, 128, 64, 32, 3]

# K = 1.23e-5 # https://www.engineeringtoolbox.com/diffusion-coefficients-d_1404.html
K = 0.5 # literature
LAMBDAd = 0.1
LAMBDAu = 0.01
LAMBDAv = 0.01
LAMBDAf = 1.0

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
    dataset_model_metric = prepare_training_dataset_metric(wind_speeds_decomposed, fishnet_gdf, S_df, key_df)
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

def full_train():

    loader = load_loader(n_colloc=N_COLLOC)

    # get scaling for PINN physical loss
    x_min, y_min, t_min = loader.scaler_xyt.data_min_
    x_max, y_max, t_max = loader.scaler_xyt.data_max_

    scales = {
        "x": (x_max - x_min),
        "y": (y_max - y_min),
        "t": (t_max - t_min),
    }

    model = PINN(layers=LAYERS).to(DEVICE)
    loss_fn = PINNLoss(K, lambda_D=LAMBDAd, lambda_u=LAMBDAu, lambda_v=LAMBDAv, lambda_f=LAMBDAf, device=DEVICE)

    history = train(
        model=model,
        data_loader=loader,
        loss_fn=loss_fn,
        n_epochs=EPOCHS,
        lr=LR,
        print_every=PRINTEV,
        scales=scales,
        device=DEVICE
    )

    return history


if __name__ == "__main__":
    full_train()