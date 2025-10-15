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
- Analysis on AADT and correlation between traffic volumes and pollutants
-
- Select and finalize model architecture

## Model Description

### Convection-Diffusion Equation of Air Pollution Propogation

We can model the diffusion of $\text{NO}_2$ using air propogation dynamics using the convection-diffusion equation, a more advanced form of the diffusion equation considering the influence of bulk velocity:

$$
\frac{\partial D(\overrightarrow{r}, t)}{\partial t} = \nabla\cdot\big[K(D,\overrightarrow{r})\nabla~D(\overrightarrow{r},t)\big]-\nabla\cdot\big[\overrightarrow{v}(\overrightarrow{r},t)D(\overrightarrow{r},t)\big]+S(\overrightarrow{r},t)
$$

Where:
- $D$ is the concentration of $\text{NO}_2$ in parts per million (PPM)
- $K$ is the coefficient of diffusion
- $\overrightarrow{r}$ is a location within the urban area in meters
- $t$ is the time in hours
- $\overrightarrow{v}$ is windspeed in meters per second
- $S$ is the source of $\text{NO}_2$ in PPM per second

We can train a neural network (NN) which outputs three observable air and environmental quantities: $\hat{D}(\cdot;\theta)$, $\hat{u}(\cdot;\theta)$, and $\hat{v}(\cdot;\theta)$; predicted pollution concentration, x-direction wind speed, and y-direction wind-speed respectively. Accordingly, there are three loss components $L_D(\theta)$, $L_u(\theta)$, and $L_v(\theta)$ that measures data discrepancy loss.

Decomposing $\overrightarrow{r}$ into its x and y components and making the assumption that the coefficient of diffusion $K$ is constant, we can obtain a partial derivative equation (PDE) residual function to obtain the physical loss discrepancy:

$$
\hat{f}(x,y,t) := K((\hat{D}''_{xx})(x,y,t;\theta)+K(\hat{D}''_{yy})(x,y,t;\theta))
$$
$$
-\hat{D}(x,y,t;\theta)(\hat{u}'_x(x,y,t;\theta)+\hat{v}'_y(x,y,t;\theta))
$$
$$
-\hat{u}(x,y,t;\theta)\hat{D}'_x(x,y,t\theta)-\hat{v}(x,y,t;\theta)\hat{D}'_y(x,y,t;\theta)
$$
$$
+\hat{S}(x,y,t;\theta)-\hat{D}'_t(x,y,t;\theta)
$$

Which is trained on the collocation (virtual) training set $T_f = \{(x^i_f, y^i_f, t^i_f, )|i=1,2,3,\ldots,N_f\}$ to form the physical discrepancy loss $\mathcal{L_f}(\theta)$. Convsersely, the data discrepancy loss is trained on $T_D = \{(x^i_f, y^i_f, t^i_f,D^i_D)|i=1,2,3,\ldots,N_D\}$. Combining the above data and physical discrepancy, the complete loss function, utilizing mean square error (MSE), is:

$$
\mathcal{L}(\theta) = \lambda_D\mathcal{D}(\theta)+\lambda_u\mathcal{u}(\theta)+\lambda_v\mathcal{v}(\theta)+\lambda_f\mathcal{f}(\theta)=
$$


$$
= \frac{\lambda_D}{N_D}\sum_{i=1}^{N_D} \left| \hat{D}(x_D^i, y_D^i, t_D^i; \theta) - D_D^i \right|^2
$$
$$
+ \frac{\lambda_u}{N_D}\sum_{i=1}^{N_D} \left| u(x_D^i, y_D^i, t_D^i; \theta) - u_D^i \right|^2
$$
$$
+ \frac{\lambda_v}{N_D}\sum_{i=1}^{N_D} \left| \hat{v}(x_D^i, y_D^i, t_D^i; \theta) - v_D^i \right|^2
$$
$$
+ \frac{\lambda_f}{N_f}\sum_{i=1}^{N_f}\left|\hat{f}(x_f^i, y_f^i, t_f^i; \theta) \right|^2
$$