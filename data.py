# University of Toronto Cities in Motion: Student Hackathon

from pathlib import Path
from typing import List, Dict
from pandas import DataFrame
from geopandas import GeoDataFrame
import pandas as pd
import geopandas as gpd

def read_gis_predictors() -> List[DataFrame]:
    """Returns a list of pandas DataFrames for the GIS Predictors dataset.

    Returns:
        List[DataFrame]: The Fishnet_Predictor.csv and GISRoadWith4Samples.csv files as DataFrames
    """
    data_path = Path("data/GIS_predictors")
    file_paths = sorted(data_path.glob("*.csv"))
    return [pd.read_csv(path) for path in file_paths]


def read_fishnet(verbose=False) -> List[GeoDataFrame]|Dict[str,GeoDataFrame]:
    """Returns GeoDataFrames of the fishnet dataset

    Args:
        verbose (bool, optional): Returns GeoDataFrames with or without filenames. Defaults to False.

    Returns:
        List[GeoDataFrame]|Dict[str,GeoDataFrame]: List of GeoDataFrames or dict of GeoDataFrames with file name.
    """
    data_path = Path("data/fishnet")
    file_paths = list(sorted(data_path.glob("*.shp")))
    if verbose:
        return {file.name:gpd.read_file(file) for file in file_paths}
    else:
        return [gpd.read_file(file) for file in file_paths]

def read_drove_on() -> GeoDataFrame:
    data_path = Path("data/DroveOn100mRoad/DroveOn100mRoad.shp")
    return gpd.read_file(data_path)

def read_r3() -> DataFrame:
    data_path = Path("data/r3File_Merge1.csv")
    return pd.read_csv(data_path)