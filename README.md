# time-series-regression
 

## Implementation of regression techniques on time-series data to  generate future predictions

## Problem Statement
A computational system records transactions. These transactions are stored  in a time series log. Now consider following scenarios,
Scenario 1: In a grocery store, a sales purchase is made for some  goods. This transaction with relevant information is recorded in a log  by a computer system. This log then contains historical information of  all the goods purchased in that store. 
∙ Scenario 2: A Cloud platform continuously records the resource  utilization of instances per minute and stores this information in the  resource utilization log. For example, Instance A, has its CPU, Memory,  Disk and Network Bandwidth utilization with a timestamp stored in a  log file. This log file then contains the historical data of resource  utilization from the time that Instance A was created and till the  time it was destroyed.
Given scenario 1 and 2, How can we use Machine Learning constructs, to  implement an intelligent system, that allow the user to see a predicted  forecast of transaction. Input to the system is transactional log files. The  system is expected to perform required data cleaning and transformation.  Then the system is expected to perform required analysis on the processed  data and generate results in terms of future predictions. The output of the  system should be a predicted transactional log.
## Brief Description
This project attempts at predicting future CPU/Memory utilization based on previous resource utilization which was mainly used for transactional logs.
We use Machine Learning algorithms to figure out the problem at hand.
## Proposed Solution
The project is divided into two main parts.
Operations on Group
Operations on Individual Instance
After getting the dataset through the zip file, we gather the data groupwise and instancewise(Due to certain hardware constraints, I’m able to perform operations on the data of a single group and instance only), then we concatenate the dataset in dataframes, by using glob.
We then perform Data Cleaning techniques like Dropping unnecessary columns, Changing the index, checking for null and replacing null with relevant values,etc. on the dataset, to make sure the data is processed properly and can run without errors. Also, calculations for CPU Usage and Memory Usage have been performed for reducing the number of features.
The extensive Exploratory Data Analysis(EDA) is performed on the dataset on both parts mentioned above to get detailed insights on the dataset. We have used a module named pandas_profiling to get the insights easily. All the graphs and relevant data can be found in the notebook directly.
Based on these insights, we perform feature engineering to reduce the dimensionality in order to get a better performing model.
We use ML Algorithms like LinearRegression, MLPRegressor, KNeighborsRegressor, RandomForestRegressor and SVR for training our model.
We get the best model out of these by applying grid search hyperparameter optimization methods.
We then save the model and use it for testing and evaluation.
## Block Diagram
![Alt text](https://github.com/chiranthancv95/time-series-regression/blob/main/block_diagram_for_time-series-regression.png?raw=true)
## Sample Test Results
LR: 1.000000 (0.000000)
NN: 0.994855 (0.000688) 
KNN: 0.926255 (0.072600)
RF: 0.916538 (0.081973)
SVR: nan (nan)

Best Score - 0.018934020681710493
Best Model - RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',max_depth=14, max_features='auto', max_leaf_nodes=None,max_samples=None, min_impurity_decrease=0.0,min_impurity_split=None, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,n_estimators=100, n_jobs=None, oob_score=False,random_state=None, verbose=0, warm_start=False)
explained_variance:  0.9975
mean_squared_log_error:  0.0
r2:  0.997
MAE:  0.0016
MSE:  0.0
RMSE:  0.0044

## Future Scope
We can use HPC to work on the entire dataset, since the number of datapoints are very high.
We can perform seasonality detection, for improving the model accuracy.
We can use Auto-Regression Networks for better accuracy.
We can perform Principal Component Analysis and reduce the dimensionality further. 
