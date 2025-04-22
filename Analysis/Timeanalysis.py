import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from scipy.stats import zscore
from prophet import Prophet 
from sklearn.ensemble import IsolationForest

# Function to perform time series analysis, decomposition, anomaly detection, and forecasting
def timeseries_analysis(df):
    df1 = df.copy()
    df1["Date"] = pd.to_datetime(df1["Date"])
    df1 = df1.sort_values(by="Date")

    # Aggregate daily sales
    sales_ts = df1.groupby("Date")["Units Sold"].sum().dropna()

    # Compute Simple Moving Average (SMA)
    sales_ts_sma = sales_ts.rolling(window=7, min_periods=1).mean()
    df1 = df1.merge(sales_ts_sma.rename("SMA_7"), on="Date", how="left")

    # plt.figure(figsize=(12, 5))
    # plt.plot(sales_ts, label="Actual Sales", color="black")
    # plt.plot(sales_ts_sma, label="7-Day SMA", color="blue", linestyle="dashed")
    # plt.title("Actual Sales vs. 7-Day SMA")
    # plt.legend()
    # plt.show()

    # Perform STL decomposition (Seasonal-Trend decomposition using LOESS)
    stl = STL(sales_ts, seasonal=7) 
    result = stl.fit()

    # fig, ax = plt.subplots(4, 1, figsize=(12, 8))
    # ax[0].plot(sales_ts, label="Original", color="black")
    # ax[0].set_title("Original Time Series")

    # ax[1].plot(result.trend, label="Trend", color="blue")
    # ax[1].set_title("Trend Component")

    # ax[2].plot(result.seasonal, label="Seasonality", color="green")
    # ax[2].set_title("Seasonal Component")

    # ax[3].plot(result.resid, label="Residual", color="red")
    # ax[3].set_title("Residual Component")

    # plt.tight_layout()
    # plt.show()
    # Calculate Z-scores on the residual component to detect anomalies
    df1["Z_Score"] = zscore(df1["Units Sold"])
    df1["Anomaly"] = df1["Z_Score"].abs() > 3  
    iso_df = df1[[
        "Units Sold", "SMA_7", "Ad Spend (USD)", "Website Traffic (Visits)",
        "Click-Through Rate (CTR%)", "Conversion Rate (%)", "Customer Satisfaction Score",
        "Return Rate (%)", "Competitor Influence Score", "Retail Store Footfall"
    ]].dropna()

    # Apply IsolationForest for anomaly detection
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_labels = iso_forest.fit_predict(iso_df)
    df1.loc[iso_df.index, "IF_Anomaly"] = iso_labels == -1

    # Combine anomalies
    df1["Combined_Anomaly"] = df1["Anomaly"] | df1["IF_Anomaly"]
    anomalies = df1[df1["Combined_Anomaly"] == True]

    additional_features = [
        "Ad Spend (USD)", "Website Traffic (Visits)", "Click-Through Rate (CTR%)",
        "Conversion Rate (%)", "Customer Satisfaction Score", "Return Rate (%)",
        "Competitor Influence Score", "Retail Store Footfall", "SMA_7"
    ]

    # Prepare data for Prophet forecasting
    # prophet_df1 = df1[["Date", "Units Sold"] + additional_features].copy()
    # prophet_df1.columns = ["ds", "y"] + additional_features # Prophet requires columns: ds (date), y (value)
    # model = Prophet()
    # for feature in additional_features:
    #     model.add_regressor(feature)  

    # model.fit(prophet_df1)
    # future = model.make_future_dataframe(periods=30)
    # for feature in additional_features:
    #     future[feature] = prophet_df1[feature].mean()  

    # forecast = model.predict(future)
    # model.plot(forecast)
    # plt.title("Sales Forecasting (Next 30 Days)")
    # plt.show()

    # Extract actual analysis and convert to basic types
    actual_analysis = {
        "Mean Sales": sales_ts.mean().item(),
        "Median Sales": sales_ts.median().item(),
        "Standard Deviation": sales_ts.std().item(),
        "Max Sales": sales_ts.max().item(),
        "Min Sales": sales_ts.min().item(),
        "SMA (Latest 7-Day Average)": sales_ts_sma.iloc[-1].item(),
        "Trend Insights": {
            "Overall Trend": "Increasing" if result.trend.diff().mean() > 0 else "Decreasing",
            "Recent Trend": "Increasing" if result.trend.diff().iloc[-7:].mean() > 0 else "Decreasing"
        }
    }
    print(actual_analysis,anomalies)
    # output
    # {'Mean Sales': 11193.406593406593, 'Median Sales': 10943.0, 'Standard Deviation': 2191.575120917079, 'Max Sales': 16269, 'Min Sales': 6662, 'SMA (Latest 7-Day Average)': 12639.857142857143, 'Trend Insights': {'Overall Trend': 'Increasing', 'Recent Trend': 'Increasing'}}
    # Date         Brand     Model      Units Sold  Total Revenue  ...         SMA_7   Z_Score  Anomaly  IF_Anomaly Combined_Anomaly
    #  2024-01-01  Realme  realme c55         474       198606.0  ...  12820.000000  1.559493    False        True             True
    #  2024-01-01  Realme  realme c55          33        27555.0  ...  12820.000000 -1.498573    False        True             True
    return {
        "anomalies": anomalies.copy().assign(Date=anomalies["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "actual_analysis": actual_analysis
    }


# If you have any doubts or suggestions please drop a message to me(rushi) or aman