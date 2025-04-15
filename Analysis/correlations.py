import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from logging_config import logger


#Function to perform the correlations between required features using pearson and spearman
def correlation_analysis(df):
    df2 = df.copy()
    try:
        logger.info("Correlation analysis started")

        relevant_columns = [
            'Units Sold', 'Total Revenue', 'Website Traffic (Visits)', 'Ad Spend (USD)',
            "Click-Through Rate (CTR%)", "Conversion Rate (%)", "Retail Store Footfall",
            "Customer Satisfaction Score", "Competitor Influence Score", "Return Rate (%)"
        ]
        df_corr = df2[relevant_columns].dropna()
        #pearson method
        pearson_corr = df_corr.corr(method='pearson')
        #Spearman method
        spearman_corr = df_corr.corr(method='spearman')
        logger.info("Correlation matrices calculated")

        plt.figure(figsize=(12, 8))
        sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Pearson Correlation Heatmap")
        plt.savefig("heatmap.png")
        plt.close()
        logger.info("Saved heatmap as heatmap.png")

        #Granger Casuality Test
        # Granger Causality Test helps determine whether changes in ad spend can predict changes in sales
        # It assumes time-series data and checks if the lagged values of 'Ad Spend' add predictive power to 'Units Sold'
        #If not required comment the code and remove from the return 
        if df_corr.shape[0] > 10: 
            granger_df = df_corr[["Units Sold", "Ad Spend (USD)"]].dropna()
            max_lag = 3  # Reduced lag if data is small
            logger.info("Performing Granger Causality Test")
            granger_results = grangercausalitytests(granger_df, max_lag, verbose=False)
            logger.info("Granger Causality Test completed")
        else:
            logger.warning("Not enough data points for Granger Causality Test")
            granger_results = None

        return pearson_corr, spearman_corr, granger_results

    except Exception as e:
        logger.error(f"Error in correlation_analysis: {str(e)}", exc_info=True)
        return {"error": str(e)}
