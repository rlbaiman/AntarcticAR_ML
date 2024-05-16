# AntarcticAR_ML
Exploring the predictability of Antarctic ARs using UNETs.

* Make_X_data: create training, testing, and validating data from 19 MERRA2 reanalysis variables
* Make_Y_data: create the dependent variable labeling landfalling ARs based on Wille et al (2019) AR catalog applied to MERRA2
* Use Combine_X_Y.py and Combine_X_Y.script to put all the data together
* create a full dataset (n = 14610 days), put 15% of data in testing (n = 2,191 days) and 15% in validating (n = 2,191 days) dataset.
* take the training dataset of 70% of the data (n = 10,258 days) and create trimmed dataset where the each of the 20 categories has about the same number of days

|Category | Num Training Days |
|:---:| :---:     | 
|1 | 448 |
|2 | 424 |
|3 | 418 |
|4 | 468 |
|5 | 388 |
|6 | 335 |
|7 | 396 |
|8 | 374 |
|9 | 366 |
|10 | 335 |
|11 | 489 |
|12| 471|
|13| 460 |
|14| 502 |
|15| 482 |
|16 | 507 |
|17| 510|
|18| 472 |
|19 | 415 |
|20| 382|
|Not a cagegory: no AR|488   | 