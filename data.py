# University of Toronto Cities in Motion: Student Hackathon

from pathlib import Path
from typing import List
from pandas import DataFrame
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