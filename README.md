# Project_2_Federal_Fund_Predective_Classifier

## Project Goal
* Make an algorithm using economic factors to predict Federal Fund Effective Rate change
* Finance, how it relates: 
  * Short-term Interest Rates: Changes in the federal funds rate directly affect short-term borrowing costs for banks, which in turn influences rates on consumer loans and credit cards.
  * Stock Market Performance: Investors closely monitor the federal funds rate, as changes can prompt strong reactions in the stock market. A rate decrease often leads to market gains due to lower borrowing costs for companies
## Data Collection, Exploration, and Cleanup Process:
* Data Collection:
  * The data was collected from the Federal Reserve Economic Data (FRED) website, a reliable and authoritative source maintained by the Federal Reserve Bank of St. Louis.
  * Various economic indicators, such as interest rates, GDP, and inflation related metrics, were downloaded in CSV format for analysis.
* Data Exploration:
  * Key statistics and data summaries were generated to understand the time structure, trends,  missing values or outliers in the data.
  * Visualizations like Time series and Plots used
* Data Cleanup:
  * Missing values were identified and handled using drop techniques, or were filled in based on their value for a pd of time. 
  *Outliers were existed, and were determined huge economic events during ‘08 and ’20.
  *Data was sorted chronologically, and unnecessary columns were dropped to streamline the dataset for analysis.
## Data Collected
* CSV Files
  * The Target: FF=Federal Fund Effective Rate
  * FOREX Rates: AUD=Australian Dollar, EUR=Euro, GBP= Great British Pound, JPY = Japanese Yen
  * Imported Commodities: Oil csv
  * Treasuries are benchmarks for performance of financial instruments: T10YR = 10YR treasury Bill, T3M= 3 Month Treasury BIll
  * Economic Features: GDP= Gross Domestic Product, NAT = Natural Unemployment Rate, PCE = Personal Consumption Expenditure
## Data Cleaning 
* This process was done to all csv files to ensure all datasets were the same in order to combine & compare. 
  * Renaming columns to better appeal tables
  * Generate weekly date range from start to the end of the DataFrame
  * Create new DataFrame with weekly dates
  * Use merge_asof to align the monthly values to the weekly dates
  * Shift 4 days back to align every dataframe on the same date 
## Federal Funds Change Line Plot
* Description: This plot displays the changes in the federal funds rate (FF_CHG) over time.
* Features:The x-axis represents time, with sequential data points.
* The y-axis shows the magnitude of changes, highlighting fluctuations and spikes.
* It indicates periods of volatility and stability in the federal funds rate.
## Prepping for Time Series Forecasting 
* Preparation for Analysis: Renaming columns to 'ds' and 'y' is a common practice when preparing data for time series forecasting, especially when using libraries like Prophet, which expect these specific column names.
* Preparing a time series forecasting model using Prophet, ensuring the data is complete by checking for missing values in the DataFrame, creating an instance of the Prophet forecasting model, & Fitting the Prophet model to the data in FF_df, preparing it for forecasting so that the model is ready for training.
Creating a Future DataFrame:
   - **Code**: `future_trends = model.make_future_dataframe(periods=52, freq="W")`
   - **Function**: Generates a DataFrame to extend predictions 52 weeks (1 year) into the future on a weekly basis.
   - **Output**: Displays the last five rows of this future DataFrame, showing dates from early August to the end of August 2025.

* **Making Predictions**:
   - **Code**: `forecast_trends = model.predict(future_trends)`
   - **Function**: Uses the Prophet model to predict future trends based on the `future_trends` DataFrame.
   - **Output**: Displays the last five rows of the forecast DataFrame, including columns like `trend`, `yhat_lower`, `yhat_upper`, and `additive_terms`.
* Prophet model, depicting a time series forecast:
  * Black Dots: Represent actual historical data points of the variable being analyzed.
  * Blue Line: Shows the predicted trend over time, illustrating the central forecast.
  * Shaded Blue Area: Indicates the uncertainty intervals around the predictions, showing the range within which future values are expected to fall
* Key Features:
  * Time Frame: The x-axis spans from 2004 to 2024, providing a long-term view of the data.
  * Y-axis: Represents the magnitude of the variable being forecasted.
  * Volatility and Stability: The plot highlights periods of volatility and stability, with some fluctuations and outliers in the data.
* Trend Plot (Top):X-axis:
  * X-axis: Represents time from 2004 to 2024.
  * Y-axis: Shows the trend component of the forecast.
  * Line: Indicates the overall direction of the trend, with a dip around 2008 and a gradual increase thereafter.
* Yearly Seasonality Plot (Bottom):
  * X-axis: Represents the days of the year.
  * Y-axis: Displays the yearly seasonal effect on the forecast.
  * Line: Shows recurring patterns within each year, with peaks and drops indicating times of higher and lower values.
## Preparing to Train Model
* Checking for Na's dropping Na's
* Assure similar Data Types for Classification Regression
  * Checking data types of each column and converting to numeric values
* Instantiate Target Variable and Features
  * We also set the X and the Y to start our training.
  * Converting Federal funds change into numeric value of either 1 for a positive change or 0 if negative or no change this brought about some concerns in our data as there was an imbalance that could affect or models score. May cause Model bias, Reduced Sensitivity, Misleading Accuracy, or Overfitting. 
## Building the Models  
* Balanced Random Forest: 
  * It uses a balanced bootstrap sampling technique, drawing an equal number of samples from both minority and majority classes.
  * This approach ensures that each tree in the forest is trained on a balanced subset of the data, preventing bias towards the majority class.
  * It helps to improve the model's ability to predict the minority class without losing information from the majority class
* XGBOOST:
  * The algorithm can be configured to focus more on misclassification of the minority class during training.
  * XGBoost's tree-based approach allows it to capture complex non-linear relationships in the data, which can be particularly useful for imbalanced datasets
* LightGBM
  * It provides a class_weight parameter that allows assigning different weights to classes, helping to balance the importance of minority and majority classes.
  * LightGBM uses a leaf-wise growth strategy, which can lead to better handling of imbalanced data compared to level-wise tree growth.
  * The algorithm is highly efficient and can handle large datasets, making it suitable for real-world imbalanced problems
## Model Metrics Analysis
* Overall Performance:
  * All three models show similar performance, with F1 scores ranging from 0.6199 to 0.6398, indicating moderate predictive power. The MCC scores are relatively low (between 0.0621 and 0.1129), suggesting that the models' predictions are only slightly better than random chance. The PR AUC scores are also relatively low (around 0.3), indicating limited ability to distinguish between classes.
Model Comparison:
1. Random Forest performs slightly better in terms of F1 score (0.6398) compared to XGBoost (0.6199) and LightGBM (0.6280).
2. XGBoost has the highest MCC score (0.1129), indicating it might be slightly better at balancing true positives and true negatives.
3. LightGBM has the highest PR AUC (0.3116), suggesting it might have a slight edge in precision-recall trade-off.
Class Imbalance:
All models show signs of class imbalance, with much better performance on class 0 (likely the majority class) compared to class 1. This is evident from the classification reports, where precision, recall, and F1-score for class 0 are significantly higher than for class 1.
The top 5 important features across the models are:
1. GBP_EX_rate (percent change)
2. JPY_EX_rate (percent change)
3. AUD_EX_rate (percent change)
4. 10YR_Treasury_Yield (Percent Change)
5. EUR_EX_rate (percent change) / PCE (Percent Change)







