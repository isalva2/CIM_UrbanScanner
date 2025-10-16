# University of Toronto Cities in Motion: Student Hackathon

from pandas import DataFrame
from geopandas import GeoDataFrame
import pandas as pd
import geopandas as gpd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def windspeed_decomp_prerocess(df:DataFrame, bias_deg:float = 180.00, temporal_resolution:str="hour"):

    # columns match `r3File_Merge1.csv`
    magnitude_col = "WS"
    direction_col = "WD"

    # copy df, filter, and convert timestamp
    df = df.copy()
    df = df[
        (df.Lat1 > 0.0) & # northen hemisphere
        (df.Long1 < 0.0) & # western hemisphere
        (df[direction_col] > 0.0) &
        (df.Temp > -35.00) &
        (df.RH > 0.0) &
        (df.RH < 100.0)
    ]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.dropna(subset=['Timestamp'], inplace=True)

    # decompose wind speed
    bias_rad = np.radians(bias_deg)
    df["direction_rad"] = np.radians(df[direction_col]) + bias_rad

    df["wind_x"] = df[magnitude_col] * np.sin(df["direction_rad"])
    df["wind_y"] = df[magnitude_col] * np.cos(df["direction_rad"])

    # add temporal resolution
    if temporal_resolution=="hour":
        df["hour"] = df["Timestamp"].dt.hour

    return df

def prepare_training_dataset_metric(
    wind_speed_df:DataFrame,
    fishnet_gdf:GeoDataFrame,
    analysis_date:str,
    to_model:bool=True,
    ) -> DataFrame:

    # copy and filter to analysis date
    training_dataset = wind_speed_df.copy()

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

    # filter relevant columns
    columns_keep = [left_on, "Timestamp", "Lat1", "Long1", "WS", "r3_key", "x_meter", "y_meter", "hour", "NO2", "wind_x", "wind_y"]
    training_dataset = training_dataset[columns_keep]

    if to_model:
        return training_dataset.iloc[:,-7:]
    else:
        return training_dataset

class DataLoader:
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
        pass

        self.n_collocation = n_collocation
        self.device = device
        self.scale_inputs = scale_inputs
        self.test_size = test_size
        self.random_state = random_state

        # rescaling
        self.scaler_xyt = MinMaxScaler()
        self.scaler_outputs = MinMaxScaler()

        # data structures
        self.df:DataFrame = df
        self.gdf:GeoDataFrame = gdf
        self.T_D_train = None
        self.T_D_test = None
        self.T_f = None

        def prepare_data(self):
            expected_cols = {"r3_key", "x_meter", "y_meter", "hour", "NO2", "wind_x", "wind_y"}
            missing = expected_cols - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Add placeholder S if missing
            if "S" not in df.columns:
                df["S"] = 0.0

            self.df = df
            return self

        def preprocess(self):
            """Normalize and split data into training/test sets."""
            if self.df is None:
                raise ValueError("Call load_data() first.")

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

            # Fit scalers
            if self.scale_inputs:
                xyt_scaled = self.scaler_xyt.fit_transform(xyt)
            else:
                xyt_scaled = xyt
            self.scaler_outputs.fit(outputs)

            # --- Robust train/test splitting using indices (avoids long unpacking) ---
            n = xyt_scaled.shape[0]
            indices = np.arange(n)
            train_idx, test_idx = train_test_split(
                indices, test_size=self.test_size, random_state=self.random_state
            )

            xyt_train = xyt_scaled[train_idx]
            xyt_test = xyt_scaled[test_idx]

            D_train = D[train_idx]
            D_test = D[test_idx]

            u_train = u[train_idx]
            u_test = u[test_idx]

            v_train = v[train_idx]
            v_test = v[test_idx]

            S_train = S[train_idx]
            S_test = S[test_idx]

            keys_train = keys[train_idx].squeeze()
            keys_test = keys[test_idx].squeeze()

            # Convert to torch tensors
            self.T_D_train = {
                "xyt": torch.tensor(xyt_train, dtype=torch.float32, device=self.device),
                "D": torch.tensor(D_train, dtype=torch.float32, device=self.device),
                "u": torch.tensor(u_train, dtype=torch.float32, device=self.device),
                "v": torch.tensor(v_train, dtype=torch.float32, device=self.device),
                "S": torch.tensor(S_train, dtype=torch.float32, device=self.device),
                "r3_key": torch.tensor(keys_train.squeeze(), dtype=torch.int64, device=self.device),
            }

            self.T_D_test = {
                "xyt": torch.tensor(xyt_test, dtype=torch.float32, device=self.device),
                "D": torch.tensor(D_test, dtype=torch.float32, device=self.device),
                "u": torch.tensor(u_test, dtype=torch.float32, device=self.device),
                "v": torch.tensor(v_test, dtype=torch.float32, device=self.device),
                "S": torch.tensor(S_test, dtype=torch.float32, device=self.device),
                "r3_key": torch.tensor(keys_test.squeeze(), dtype=torch.int64, device=self.device),
            }

        def _generate_collocation_points(self):
            """
            scale x-y to fishnet
            """

            # get centroid and sample from fishnet
            self.gdf["x"] = self.gdf.geometry.centroid.x
            self.gdf["y"] = self.gdf.geometry.centroid.y
            sample_gdf = self.gdf.sample(n=n_collocation, random_state=self.random_state)

            x_f = sample_gdf["x"]
            y_f = sample_gdf["y"]

            # sample t_f from T_D hour distribution
            hour_dist = self.df.hour.value_counts(normalize=True)
            t_f = np.random.choice(
                hour_dist.index,
                size=self.n_collocation,
            )

            xyt_f = np.hstack((x_f,y_f,t_f))

            # fit scaler to entire fishnet area and from T_D hour distribution
            x_proj = self.gdf["x"]
            y_proj = self.gdf["y"]

            hour_dist_proj = self.df
            t_proj = np.random.choice(
                hour_dist.index,
                size=len(self.gdf)
            )

            xyt_proj = np.hstack((x_proj,y_proj,t_proj))

            if self.scale_inputs:
                self.scaler_xyt.fit(xyt_proj)



