import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from scipy.stats import zscore
from prophet import Prophet 


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

    plt.figure(figsize=(12, 5))
    plt.plot(sales_ts, label="Actual Sales", color="black")
    plt.plot(sales_ts_sma, label="7-Day SMA", color="blue", linestyle="dashed")
    plt.title("Actual Sales vs. 7-Day SMA")
    plt.legend()
    plt.show()

    # Perform STL decomposition (Seasonal-Trend decomposition using LOESS)
    stl = STL(sales_ts, seasonal=7) 
    result = stl.fit()

    fig, ax = plt.subplots(4, 1, figsize=(12, 8))
    ax[0].plot(sales_ts, label="Original", color="black")
    ax[0].set_title("Original Time Series")

    ax[1].plot(result.trend, label="Trend", color="blue")
    ax[1].set_title("Trend Component")

    ax[2].plot(result.seasonal, label="Seasonality", color="green")
    ax[2].set_title("Seasonal Component")

    ax[3].plot(result.resid, label="Residual", color="red")
    ax[3].set_title("Residual Component")

    plt.tight_layout()
    plt.show()

    # Calculate Z-scores on the residual component to detect anomalies
    df1["Z_Score"] = zscore(df1["Units Sold"])
    df1["Anomaly"] = df1["Z_Score"].abs() > 3  
    anomalies = df1[df1["Anomaly"]]

    additional_features = [
        "Ad Spend (USD)", "Website Traffic (Visits)", "Click-Through Rate (CTR%)",
        "Conversion Rate (%)", "Customer Satisfaction Score", "Return Rate (%)",
        "Competitor Influence Score", "Retail Store Footfall", "SMA_7"
    ]   

   # If forecasting not required comment the below code and run the app
   # Prepare data for Prophet forecasting
    prophet_df1 = df1[["Date", "Units Sold"] + additional_features].copy()
    prophet_df1.columns = ["ds", "y"] + additional_features # Prophet requires columns: ds (date), y (value)
    # Initialize and configure Prophet model with extra regressor
    model = Prophet()
    for feature in additional_features:
        model.add_regressor(feature)  

    model.fit(prophet_df1)
    # Generate forecast
    future = model.make_future_dataframe(periods=30)
    for feature in additional_features:
        future[feature] = prophet_df1[feature].mean()  

    forecast = model.predict(future)

    model.plot(forecast)
    plt.title("Sales Forecasting (Next 30 Days)")
    plt.show()

    actual_analysis = {
        "Mean Sales": sales_ts.mean(),
        "Median Sales": sales_ts.median(),
        "Standard Deviation": sales_ts.std(),
        "Max Sales": sales_ts.max(),
        "Min Sales": sales_ts.min(),
        "SMA (Latest 7-Day Average)": sales_ts_sma.iloc[-1],
        "Trend Insights": {
            "Overall Trend": "Increasing" if result.trend.diff().mean() > 0 else "Decreasing",
            "Recent Trend": "Increasing" if result.trend.diff().iloc[-7:].mean() > 0 else "Decreasing"
        }
    }

    return {
        "decomposition": result, 
        "forecast": forecast, 
        "anomalies": anomalies, 
        "actual_analysis": actual_analysis
    }
