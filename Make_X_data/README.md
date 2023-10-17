# Instructions for Making X data

### 1. Create yrly files of X variables
#### Make_yrly_data.py, Make_yrly_data.script

* Run in parallel over year 1980 - 2019
* Variables include:
|| Variable | Level | Lats Included | Lead Times |
|:---:| :---:     |  :---:|  :---:        | :---: |
|1| U wind | 950hPa |    -75, -45 ||
|2| V wind | 950hPa |    -75, -45 ||
|3| Temp | 950hPa |   -80, -50 ||
|4| SLP|  |    -75, -45 ||
|5 |Upwards Latent Energy Flux|  |    -50, -20 ||
|6 |Tropical Convection| |  -30, 0 | 4 days|
|7 |Stream Function| |   -90, 0 |2 days |
|8|IWV|  |    -70, -40 ||
* Select the year, level, and variable using cdo commands. These are saved to scratch directories because memory did not allow loading these to work with
* Call function Resample
    * Resample to 6hourly mean (this may smooth some anomalous values but we will calculate the standardized anomalies based on these values so we should capture anomalies)
    * If you want a leadtime x, shift data forward by that amount, leaving nan values for the first x hours 
    * Interpolate data using the Lats Included and -180 to 180 longitude for each variable. 
    (lon: 576, lat: 181) ->  (lon: 256, lat: 32)
    * For Tropical Convection variable, make this binary data with values of 3 if the outgoing longwave radiation is below 220 W/m2, 0 everywhere else. 
* Save as yearly data

### 2. Make Standardized Anomalies
#### Make_std_anomalies.py, Make_std_anomalies.script

* Run in parallel over list of 8 variables
* Add a uniform lat_index from 0 to 32 to make combining X data easy,
    leave the true lats for each variable as a coordinate
* Tropical convection variable does not need to be normalized because it is binary,
    so pass this data along to a new scratch directory
* Calculate monthly mean and standard deviation of all other variables
    * IWV is not normaly distributed because it starts at zero and has a long right tail.
        Calculate standard deviation by assuming a normal distribution of the right half 
        and a mirror image of the right half. 
* Based on the monthly mean and standard deviation, calculate the monthly standardized anomalies
* Save to a new directory 

### 2. Plot X variable data
#### Plot_X_data.ipynb
* plot individual timesteps of X data to check it out
![example of Y data at one timestep](X_data_example.png)