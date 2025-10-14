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
│   ├── r3File_Merge1.csv (wind speed data)
│   └── zipped (zipped, original data downloads)
├── data.py (data input/output and preprocessing)
├── LICENSE
├── notebooks
│   └── data_analysis.ipynb
├── README.md
└── requirements.txt
```

The data folder contains the project data in the exact folder structure provided. Python files at the top level will be the main working documents, and the notebooks folder contains jupyter notebooks for personal use and data exploration.

## TODO
- Set up a template for plots and maps
- Spatial join `GISRoadWith4Samples.csv` with fishnet shapefiles
- Read through GIS variables data dictionary and identify target, input, and other pertient variables for prediction and static input to model
- Exploratory data analysis on GIS variable data (Chloropleth maps and box plots)
- Select and finalize model architecture

## Model Description