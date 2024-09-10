# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import datetime as dt
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# %%
# load all csv files into respective dataframes

#The Target, and Actual Rate that Federal Banks charge each other for over night Loans
FR_target_pt1_df = pd.read_csv('..\\Project_2_BTC_Price_Modeling\csv_files\DFEDTARU_pt1.csv')
FR_target_pt2_df = pd.read_csv('..\\Project_2_BTC_Price_Modeling\csv_files\DFEDTAR_pt2.csv')

FF_df = pd.read_csv('..\\Project_2_BTC_Price_Modeling\csv_files\FF.csv')


#FOREX rates 
AUD_df = pd.read_csv('..\\Project_2_BTC_Price_Modeling\csv_files\DEXUSAL.csv')
EUR_df = pd.read_csv('..\\Project_2_BTC_Price_Modeling\csv_files\EXUSEU.csv')
GBP_df = pd.read_csv('..\\Project_2_BTC_Price_Modeling\csv_files\DEXUSUK.csv')
JPY_df = pd.read_csv('..\\Project_2_BTC_Price_Modeling\csv_files\DEXJPUS.csv')


#Important commodities that help determine interest rates
OIL_df = pd.read_csv('..\\Project_2_BTC_Price_Modeling\csv_files\DCOILBRENTEU (1).csv')


#Treasuries are benchmarks for performance of financial instruments
T10YR_df = pd.read_csv('..\\Project_2_BTC_Price_Modeling\csv_files\WGS10YR.csv')
T3M_df = pd.read_csv('..\\Project_2_BTC_Price_Modeling\csv_files\TB3MS.csv')


#Economic features like GDP, Natural Unemployment Rate, Personal Consumption Expenditures
GDP_df = pd.read_csv(r'..\\Project_2_BTC_Price_Modeling\csv_files\GDPC1.csv')
NAT_UNEMPLOYMENT_df = pd.read_csv(r'..\\Project_2_BTC_Price_Modeling\csv_files\NROU.csv')
PCE_df = pd.read_csv('..\\Project_2_BTC_Price_Modeling\csv_files\PCE.csv')





# %%
#Euro exchange is in monthly units. Calendarize to weekly, and repeat the percent change for each week in the month

EUR_df.head()

EUR_df.rename(columns={'DATE':'Date'},inplace=True)
EUR_df["Date"] = pd.to_datetime(EUR_df['Date'])

#Generate weekly date range from the start to the end of the DataFrame
weekly_dates = pd.date_range(start=EUR_df['Date'].min(), end=EUR_df['Date'].max(), freq='W')

# Step 2: Create a new DataFrame with these weekly dates
df_weekly = pd.DataFrame({'Date': weekly_dates})

#Use merge_asof to align the monthly values to the weekly dates
EUR_df = pd.merge_asof(df_weekly, EUR_df, on='Date', direction='backward')

#Shift 4 days back to align every dataframe on the same date
EUR_df['Date'] = EUR_df['Date'] - pd.Timedelta(days=4)

EUR_df.head(20)

# %%
#shift date column for 10YR Treasury percent change 5 days ahead, so that it will match other dataframes

T10YR_df.rename(columns={'DATE':'Date'}, inplace=True)

T10YR_df['Date'] = pd.to_datetime(T10YR_df['Date'])

T10YR_df['Date'] = T10YR_df['Date'] + pd.Timedelta(days=5)

T10YR_df.head()


# %%
#Rename Dataframes
OIL_df.rename(columns={"DATE":"Date","DCOILBRENTEU_PCH":"Price_of_Barrel (Percent Change)"},inplace=True)
OIL_df.head()

# %%
T3M_df.rename(columns={'DATE':'Date'}, inplace=True)
T3M_df["Date"] = pd.to_datetime(T3M_df['Date'])

#Generate weekly date range from the start to the end of the DataFrame
weekly_dates = pd.date_range(start=T3M_df['Date'].min(), end=T3M_df['Date'].max(), freq='W')

# Create a new DataFrame with these weekly dates
df_weekly = pd.DataFrame({'Date': weekly_dates})

#Use merge_asof to align the monthly values to the weekly dates
T3M_df = pd.merge_asof(df_weekly, T3M_df, on='Date', direction='backward')

T3M_df['Date'] = T3M_df['Date'] - pd.Timedelta(days=4)

T3M_df.head()

# %%
#Add the missing date, 2008-12-14 and interst value change of 0 (filling missing data) to dataframe
FR_target_pt2_df.loc[225] = ['2008-12-24', '0.0']

FR_target_pt2_df.tail()

# %%
#rename target variable pt1 to match pt2 and combine

FR_target_pt1_df.rename(columns={'DATE':'Date','DFEDTARU_CHG' : 'DFEDTAR_CHG'}, inplace=True)
FR_target_pt2_df.rename(columns={'DATE':'Date'}, inplace=True)
FR_target_df = pd.concat([FR_target_pt2_df,FR_target_pt1_df], ignore_index=True)
FR_target_df['Date'] = pd.to_datetime(FR_target_df['Date'])
display(FR_target_df.head())
display(FR_target_df.tail())



# %%
FF_df.info()

# %%
FF_df['DATE'] = pd.to_datetime(FF_df['DATE'])
FF_df = FF_df.rename(columns={"DATE":"Date"})


# %%
FF_df.head()

# %%
FF_df['FF_CHG'].plot(x='Date')

# %%
FF_df.columns = ['ds','y']


# %%
display(FF_df.shape)
display(FF_df.head(10))
display(FF_df.tail(10))

# %%
#check na values

FF_df.isna().value_counts()

# %%
#Instantiate a Prophet model
model = Prophet()
model

# %%
#Fit the prophet model to BTC_close_df
model.fit(FF_df)

# %%
# Create a future dataframe to hold predictions
# Make the prediction go out as far as  52 weeeks (1 year)
future_trends = model.make_future_dataframe(periods=52, freq="W")

# View the last five rows of the predictions
future_trends.tail()

# %%
# Make the predictions for the trend data using the future_trends DataFrame
forecast_trends = model.predict(future_trends)

# Display the first five rows of the forecast DataFrame
forecast_trends.tail()

# %%
model.plot(forecast_trends)

# %%
figures = model.plot_components(forecast_trends)

# %%
# At this point, it's useful to set the `datetime` index of the forecast data.
forecast_trends = forecast_trends.set_index(["ds"])
forecast_trends.head()

# %%
# From the `forecast_trends` DataFrame, plot to visualize
#  the yhat, yhat_lower, and yhat_upper columns over the last 12 weeks  
forecast_trends[["yhat", "yhat_lower", "yhat_upper"]].iloc[-12:, :].plot()

# %%
forecast_Nov_2024 = forecast_trends.loc["2024-11-01":"2024-11-30"][["yhat_upper", "yhat_lower", "yhat"]]

# Replace the column names to something less technical sounding
forecast_Nov_2024 = forecast_Nov_2024.rename(
    columns={
        "yhat_upper": "Best Case",
        "yhat_lower": "Worst Case", 
        "yhat": "Most Likely Case"
    }
)

# Review the last five rows of the DataFrame
forecast_Nov_2024.tail()

# %%
forecast_Nov_2024.mean()

# %%

#Create return rate columns for FOREX, rename columns to specify currency, and set date column to date time.

AUD_rename_df = AUD_df.rename(columns={'DATE': 'Date','DEXUSAL_PCH':'AUD_EX_rate (percent change)'})
EUR_rename_df = EUR_df.rename(columns={'DATE': 'Date','EXUSEU_PCH':'EUR_EX_rate (percent change)'})
GBP_rename_df = GBP_df.rename(columns={'DATE': 'Date','DEXUSUK_PCH':'GBP_EX_rate (percent change)'})
JPY_rename_df = JPY_df.rename(columns={'DATE': 'Date','DEXJPUS_PCH':'JPY_EX_rate (percent change)'})

AUD_rename_df['Date'] = pd.to_datetime(AUD_rename_df['Date'])
EUR_rename_df['Date'] = pd.to_datetime(EUR_rename_df['Date'])
GBP_rename_df['Date'] = pd.to_datetime(GBP_rename_df['Date'])
JPY_rename_df['Date'] = pd.to_datetime(JPY_rename_df['Date'])




# %%
#Combine treasury bills dataframe, and divide by 100 to reduce to decimals instead of percentages
combined_treasuries_df = T10YR_df.merge(T3M_df, on='Date',how='outer')
combined_treasuries_df['Date'] = pd.to_datetime(combined_treasuries_df['Date'])
combined_treasuries_df['WGS10YR_PCH'] = combined_treasuries_df['WGS10YR_PCH']/100
combined_treasuries_df['TB3MS_PCH'] = combined_treasuries_df['TB3MS_PCH']/100




# %%
#Merge FOREX data frames

combined_currency_ex_df = AUD_rename_df.merge(EUR_rename_df, on='Date', how='outer').merge(GBP_rename_df, on='Date', how='outer').merge(JPY_rename_df, on='Date', how='outer')

combined_currency_ex_df.head()


# %%
#Clean GDP dataframe

GDP_df.rename(columns={'DATE':'Date','GDPC1_PCH':'GDP (Percent Change)'},inplace=True)
GDP_df['Date'] = pd.to_datetime(GDP_df['Date'])

# Generate Weekly Date Range for the whole year
weekly_dates = pd.date_range(start='2004-09-01', end='2024-04-30', freq='W-WED')  # Weekly on Wednesdays
weekly_df = pd.DataFrame({'Date': weekly_dates})

# Assign Quarterly Values to Weekly Dates
# Use merge_asof to assign each weekly date to the corresponding quarter's value
GDP_df = pd.merge_asof(weekly_df, GDP_df, on= 'Date', direction='backward')

GDP_df.head()


# %%
#Clean Natural Unemployment dataframe

NAT_UNEMPLOYMENT_df.rename(columns={'DATE':'Date','NROU_PCH':'Natural_UE_rate (Percent Change)'},inplace=True)
NAT_UNEMPLOYMENT_df['Date'] = pd.to_datetime(NAT_UNEMPLOYMENT_df['Date'])

# Generate Weekly Date Range for the whole year
weekly_dates = pd.date_range(start='2004-09-01', end='2024-04-30', freq='W-WED')  # Weekly on Wednesdays
weekly_df = pd.DataFrame({'Date': weekly_dates})

# Assign Quarterly Values to Weekly Dates
# Use merge_asof to assign each weekly date to the corresponding quarter's value
NAT_UNEMPLOYMENT_df = pd.merge_asof(weekly_df, NAT_UNEMPLOYMENT_df, on= 'Date', direction='backward')

NAT_UNEMPLOYMENT_df.head()

# %%
#Clean PCE dataframe
PCE_df.rename(columns={'DATE':'Date','PCE_PCH':'PCE (Percent Change)'}, inplace=True)
PCE_df["Date"] = pd.to_datetime(PCE_df['Date'])

#Generate weekly date range from the start to the end of the DataFrame
weekly_dates = pd.date_range(start=PCE_df['Date'].min(), end=PCE_df['Date'].max(), freq='W')

# Create a new DataFrame with these weekly dates
df_weekly = pd.DataFrame({'Date': weekly_dates})

#Use merge_asof to align the monthly values to the weekly dates
PCE_df = pd.merge_asof(df_weekly, PCE_df, on='Date', direction='backward')

PCE_df['Date'] = PCE_df['Date'] - pd.Timedelta(days=4)
PCE_df.head()

# %%
#Combine Economic Features Dataframes
economic_df = GDP_df.merge(PCE_df, on='Date',how='outer').merge(NAT_UNEMPLOYMENT_df, on='Date',how='outer')

# %%
#combine ALL dataframes
combo_df = FR_target_df.merge(economic_df, on='Date',how='outer').merge(combined_currency_ex_df, on='Date',how='outer').merge(combined_treasuries_df, on='Date',how='outer')
combo_df.rename(columns={'DFEDTAR_CHG':'Federal_Fund_Target_Rate Change', 'WGS10YR_PCH':'10YR_Treasury_Yield (Percent Change)','TB3MS_PCH':'3M_Treasury_Yield (Percent Change)'},inplace=True)

# %%
#Clean NA values

combo_clean_df = combo_df.dropna()
combo_clean_df.isna()



# %%
#check datatypes of each columns

combo_clean_df.dtypes

# %%
combo_clean_df[['Federal_Fund_Target_Rate Change','AUD_EX_rate (percent change)','JPY_EX_rate (percent change)']] = combo_clean_df[['Federal_Fund_Target_Rate Change',
                                                                                                                                    'AUD_EX_rate (percent change)',
                                                                                                                                    'JPY_EX_rate (percent change)']].apply(pd.to_numeric)

# %%
#Create the target variable based on the Federal Fund Target column, encoding positive changes as 1, negative changes as 0, and no change as 2
combo_clean_df['FT_Change_Encoded'] = combo_clean_df['Federal_Fund_Target_Rate Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else)


# %%
combo_clean_df.dtypes

# %%
# Get the target variable (the "FT_Change_Encoded" column)
y = combo_clean_df["FT_Change_Encoded"]



# %%
# Get the features (everything except the "FT_Change_Encoded" column)
X = combo_clean_df.copy()
X = X.drop(columns=["FT_Change_Encoded",'Date','Federal_Fund_Target_Rate Change'])



# %%
def apply_models(X, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the models
    models = {
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'ExtraTrees': ExtraTreesClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'GradientBoost': GradientBoostingClassifier(),
    }

    # Apply each model
    for model_name, model in models.items():
        print(f"\nClassification with {model_name}:\n{'-' * 30}")
        
        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Print the accuracy score
        print("**Accuracy**:\n", accuracy_score(y_test, y_pred))

        # Print the confusion matrix
        print("\n**Confusion Matrix**:\n", confusion_matrix(y_test, y_pred))

        # Print the classification report
        print("\n**Classification Report**:\n", classification_report(y_test, y_pred))

# %%
apply_models(X, y)


