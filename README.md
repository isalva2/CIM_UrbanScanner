# CIM_UrbanScanner
University of Toronto Cities in Motion: Student Hackathon

## Project Structure

The project is currently organized as so:

```
CIM_UrbanScanner
├── data
│   ├── datareadme.md
│   ├── DroveOn100mRoad (folder)
│   ├── fishnet (folder)
│   ├── GIS_predictors (folder)
│   └── zipped (zipped, original data downloads)
├── data.py (data input/output and preprocessing)
├── LICENSE
├── notebooks
│   └── data_analysis.ipynb
├── README.md
└── requirements.txt
```

The data folder contains the project data in the exact folder structure provided. Python files at the top level will be the main working documents, and the notebooks folder contains jupyter notebooks for personal use and data exploration.

## Source Emission data

To obtain the spatially and temporally resolved source emission term S(x,y,t) for nitrogen dioxide (NO₂), the raw mobile sensing data were first processed and mapped onto the spatial analysis grid. The original reading dataset contained instantaneous NO₂ readings associated with latitude–longitude coordinates of sensours which reads emission concentarion  and timestamps. These readings were projected into the same coordinate reference system as the 100×100 m fishnet grid and spatially joined so that each measurement was assigned to the grid cell in which it occurred. The timestamp of each observation was decomposed into a date and an hourly component to enable temporal aggregation. Within each cell and day, multiple measurements recorded during the same hour were combined into a single hourly value, representing the total (or mean) NO₂ concentration for that spatial–temporal unit. This procedure produced a dataset of observed hourly NO₂ concentrations for every grid cell and day covered by the mobile campaign.

Because the sensor platform did not collect readings continuously across all hours, many grid cells had gaps in their hourly coverage. To reconstruct a continuous diurnal pattern while avoiding unrealistic extrapolation, we performed a cell-wise temporal interpolation constrained to the observed hour window of each cell. Specifically, for each grid cell the earliest and latest observed hours were identified, and linear interpolation was applied only within this interval to fill missing intermediate hours. Hours lying outside the observed range were left undefined. This approach preserved the integrity of the measured data while yielding a smooth and physically plausible hourly emission profile consistent with the temporal extent of actual observations. The interpolated hourly averages were then treated as the zone-level emission field S(x,y,t). The resulting tables serve as the spatiotemporal source input for the Physics-Informed Neural Network model that estimates and constrains pollutant dynamics through the convection–diffusion equation residual.
