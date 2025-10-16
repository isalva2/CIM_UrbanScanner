# University of Toronto Cities in Motion: Student Hackathon

from pandas import DataFrame
from geopandas import GeoDataFrame
import pandas as pd
import geopandas as gpd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def windspeed_decomp_preprocess(df: pd.DataFrame, temporal_resolution: str = "hour") -> pd.DataFrame:
    """
    Preprocess wind data for projected CRS (x = east, y = north).

    Converts meteorological 'from' direction (0°=N, 90°=E) into vector components (u,v)
    compatible with Cartesian coordinates.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'WS', 'WD', 'Timestamp', 'Lat1', 'Long1', 'Temp', 'RH'.
        temporal_resolution (str): Temporal granularity ('hour' supported).

    Returns:
        pd.DataFrame: Cleaned DataFrame with added 'wind_x', 'wind_y', and 'hour'.
    """
    df = df.copy()

    # --- Validate input ---
    required_cols = {"WS", "WD", "Timestamp", "Lat1", "Long1", "Temp", "RH"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Filter invalid data ---
    df = df[
        (df["Lat1"] > 0.0) &
        (df["Long1"] < 0.0) &
        (df["WD"].between(0.0, 360.0)) &
        (df["Temp"] > -35.0) &
        (df["RH"].between(0.0, 100.0))
    ].copy()

    # --- Parse timestamps ---
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)

    # --- Convert wind direction to radians ---
    # Convert "from" direction (meteorological) to "toward" direction (Cartesian)
    direction_math = np.radians(270.0 - df["WD"])
    df["wind_x"] = df["WS"] * np.cos(direction_math)
    df["wind_y"] = df["WS"] * np.sin(direction_math)

    # add temporal resolution
    if temporal_resolution=="hour":
        df["hour"] = df["Timestamp"].dt.hour

    return df

def prepare_training_dataset_metric(
    wind_speed_df:DataFrame,
    fishnet_gdf:GeoDataFrame,
    S_df: DataFrame,
    key_df: DataFrame,
    analysis_date:str="2021-06-09",
    to_model:bool=True,
    ) -> DataFrame:

    # copy and filter to analysis date
    training_dataset = wind_speed_df.copy()

    # # attache source df to training dataset we will need to fix this eventually
    # keyed_source = pd.merge(left=S_df, right=key_df[["Id", "id_1"]], on="id_1")
    # training_dataset = pd.merge(left=training_dataset, right=keyed_source, on=["Id", "hour"], how="left")
    # training_dataset.drop_duplicates(inplace=True)
    # training_dataset.dropna(inplace=True)
    # training_dataset["S"] = training_dataset["S"]

    training_dataset["S"] = 0.0

    if analysis_date is not None:
        target_date = pd.to_datetime(analysis_date).date()
        training_dataset = training_dataset[training_dataset.Timestamp.dt.date==target_date]

    # spatial join on fishnet gdf `FishCenterJoin.shp`
    right_on = "Id"
    left_on = "Id"
    training_dataset = pd.merge(
        left=fishnet_gdf,
        right=training_dataset,
        left_on=left_on,
        right_on=right_on,
        how="right"
        )

    # get centroid
    training_dataset["x_meter"] = training_dataset.geometry.centroid.x
    training_dataset["y_meter"] = training_dataset.geometry.centroid.y

    # add minute resolution to hour
    training_dataset["hour"] = training_dataset["hour"] + training_dataset["Timestamp"].dt.minute / 60.0

    # filter relevant columns
    columns_keep = [left_on, "Timestamp", "Lat1", "Long1", "WS", "r3_key", "x_meter", "y_meter", "hour", "NO2", "wind_x", "wind_y", "S"]
    training_dataset = training_dataset[columns_keep]

    if to_model:
        return training_dataset.iloc[:,-8:]
    else:
        return training_dataset


class CDiffDataLoader:
    def __init__(
        self,
        df: DataFrame,
        gdf: GeoDataFrame,
        n_collocation=1000,
        device="cpu",
        scale_inputs=True,
        test_size=0.5,
        random_state=1
    ):
        self.n_collocation = n_collocation
        self.device = device
        self.scale_inputs = scale_inputs
        self.test_size = test_size
        self.random_state = random_state

        # rescaling
        self.scaler_xyt = MinMaxScaler()
        self.scaler_outputs = MinMaxScaler()
        self.scaler_S = MinMaxScaler()

        # numpy rng
        np.random.seed(random_state)

        # data structures
        self.df: DataFrame = df
        self.gdf: GeoDataFrame = gdf
        self.T_D_train = None
        self.T_D_test = None
        self.T_f = None


    def prepare_data(self):
        expected_cols = {"r3_key", "x_meter", "y_meter", "hour", "NO2", "wind_x", "wind_y"}
        missing = expected_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Add placeholder S if missing
        if "S" not in self.df.columns:
            self.df["S"] = 0.0

        return self


    def preprocess(self):
        """Normalize and split data into training/test sets."""
        if self.df is None:
            raise ValueError("DataFrame not loaded.")

        # generate collocation points and fit input scaler
        self._generate_collocation_points_and_xyt_scaler()

        # Extract data columns
        x = self.df["x_meter"].values.reshape(-1, 1)
        y = self.df["y_meter"].values.reshape(-1, 1)
        t = self.df["hour"].values.reshape(-1, 1)
        D = self.df["NO2"].values.reshape(-1, 1)
        u = self.df["wind_x"].values.reshape(-1, 1)
        v = self.df["wind_y"].values.reshape(-1, 1)
        S = self.df["S"].values.reshape(-1, 1)
        keys = self.df["r3_key"].values.reshape(-1, 1)

        xyt = np.hstack((x, y, t))
        outputs = np.hstack((D, u, v))

        # Scale inputs and outputs
        if self.scale_inputs:
            xyt_scaled = self.scaler_xyt.transform(xyt)
        else:
            xyt_scaled = xyt

        self.scaler_outputs.fit(outputs)
        outputs_scaled = self.scaler_outputs.transform(outputs)

        self.scaler_S.fit(S)
        S_scaled = self.scaler_S.fit_transform(S)

        # Train/test split
        n = xyt_scaled.shape[0]
        indices = np.arange(n)
        train_idx, test_idx = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state
        )

        # Split scaled data
        xyt_train = xyt_scaled[train_idx]
        xyt_test = xyt_scaled[test_idx]

        D_train, u_train, v_train = outputs_scaled[train_idx].T
        D_test, u_test, v_test = outputs_scaled[test_idx].T

        S_train = S_scaled[train_idx]
        S_test = S_scaled[test_idx]

        keys_train = keys[train_idx].squeeze()
        keys_test = keys[test_idx].squeeze()

        # Convert to torch tensors
        self.T_D_train = {
            "xyt": torch.tensor(xyt_train, dtype=torch.float32, device=self.device),
            "D": torch.tensor(D_train.reshape(-1, 1), dtype=torch.float32, device=self.device),
            "u": torch.tensor(u_train.reshape(-1, 1), dtype=torch.float32, device=self.device),
            "v": torch.tensor(v_train.reshape(-1, 1), dtype=torch.float32, device=self.device),
            "S": torch.tensor(S_train, dtype=torch.float32, device=self.device),
            "r3_key": torch.tensor(keys_train, dtype=torch.int64, device=self.device),
        }

        self.T_D_test = {
            "xyt": torch.tensor(xyt_test, dtype=torch.float32, device=self.device),
            "D": torch.tensor(D_test.reshape(-1, 1), dtype=torch.float32, device=self.device),
            "u": torch.tensor(u_test.reshape(-1, 1), dtype=torch.float32, device=self.device),
            "v": torch.tensor(v_test.reshape(-1, 1), dtype=torch.float32, device=self.device),
            "S": torch.tensor(S_test, dtype=torch.float32, device=self.device),
            "r3_key": torch.tensor(keys_test, dtype=torch.int64, device=self.device),
        }

        return self


    def _generate_collocation_points_and_xyt_scaler(self):
        """Generate collocation points and fit xyt MinMax scaler."""
        self.gdf["x"] = self.gdf.geometry.centroid.x
        self.gdf["y"] = self.gdf.geometry.centroid.y

        sample_gdf = self.gdf.sample(n=self.n_collocation, random_state=self.random_state)
        x_f = sample_gdf["x"].values.reshape(-1, 1)
        y_f = sample_gdf["y"].values.reshape(-1, 1)

        hour_dist = self.df.hour.value_counts(normalize=True)
        hours = np.random.choice(hour_dist.index, size=self.n_collocation, p=hour_dist.values)
        t_f = hours + np.random.rand(self.n_collocation)
        t_f = t_f.reshape(-1, 1)

        xyt_f = np.hstack((x_f, y_f, t_f))

        # Fit scaler on full grid area
        x_proj = self.gdf["x"].values.reshape(-1, 1)
        y_proj = self.gdf["y"].values.reshape(-1, 1)
        t_proj = np.random.choice(hour_dist.index, size=len(self.gdf)).reshape(-1, 1)
        xyt_proj = np.hstack((x_proj, y_proj, t_proj))

        if self.scale_inputs:
            self.scaler_xyt.fit(xyt_proj)
            xyt_f = self.scaler_xyt.transform(xyt_f)

        self.T_f = torch.tensor(xyt_f, dtype=torch.float32, device=self.device)


    def get_train_test_data(self):
        if self.T_D_train is None or self.T_D_test is None:
            raise ValueError("Data not preprocessed. Run preprocess() first.")
        return self.T_D_train, self.T_D_test, self.T_f


    def inverse_scale_predictions(self, Duv_pred):
        """
        Inverse scale model predictions given a stacked tensor [D, u, v].

        Args:
            Duv_pred (torch.Tensor): Tensor of shape (N, 3) containing
                                    predicted [D, u, v] values (scaled).

        Returns:
            np.ndarray: Array of shape (N, 3) with values in the original scale.
        """
        if not torch.is_tensor(Duv_pred):
            raise TypeError("Duv_pred must be a torch tensor")

        Duv_np = Duv_pred.detach().cpu().numpy()
        return self.scaler_outputs.inverse_transform(Duv_np)


    def summary(self, verbose=True):
            """Print a concise summary of the dataset and preprocessing status."""
            print("=== Pollution Data Summary ===")

            if self.df is None:
                print("No data loaded yet.")
                return

            print(f"Total rows: {len(self.df):,}")
            print(f"Columns: {list(self.df.columns)}\n")

            print("Feature ranges (original scale):")
            for col in ["x_meter", "y_meter", "hour", "NO2", "wind_x", "wind_y", "S"]:
                vals = self.df[col].values
                print(f"  {col:<10s} min={vals.min():>10.3f}  max={vals.max():>10.3f}  mean={vals.mean():>10.3f}")

            if self.T_D_train is not None:
                n_train = self.T_D_train["xyt"].shape[0]
                n_test = self.T_D_test["xyt"].shape[0]
                print(f"\nPreprocessed:")
                print(f"  Train samples: {n_train:,}")
                print(f"  Test samples:  {n_test:,}")
                print(f"  Collocation pts: {self.n_collocation:,}")
                print(f"  Scaling: {'on' if self.scale_inputs else 'off'}")
                print(f"  Device: {self.device}")

                if verbose:
                    print("\n--- Verbose Info ---")
                    print("Tensor shapes:")
                    for k, v in self.T_D_train.items():
                        if k != "r3_key":
                            print(f"  {k:<6s}: {tuple(v.shape)}")

                    # Input ranges (scaled)
                    xyt_scaled = self.T_D_train["xyt"].cpu().numpy()
                    print("\nScaled input ranges:")
                    for i, col in enumerate(["x", "y", "t"]):
                        print(f"  {col}-scaled  min={xyt_scaled[:, i].min():>8.4f}  max={xyt_scaled[:, i].max():>8.4f}")

                    # Output min/max
                    out_minmax = (self.scaler_outputs.data_min_, self.scaler_outputs.data_max_)
                    print("\nOriginal output feature min/max:")
                    for name, mn, mx in zip(["D", "u", "v"], out_minmax[0], out_minmax[1]):
                        print(f"  {name:<3s}: min={mn:>8.4f}, max={mx:>8.4f}")

                    # --- NEW: S scaled range ---
                    S_scaled = self.T_D_train["S"].cpu().numpy()
                    print("\nScaled S range:")
                    print(f"  S-scaled  min={S_scaled.min():>8.4f}  max={S_scaled.max():>8.4f}")

            else:
                print("\nData not yet preprocessed (no train/test tensors).")

            print("==============================\n")


