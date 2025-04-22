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


        return {

            "pearson_correlation": pearson_corr.to_dict(),
            "spearman_correlation": spearman_corr.to_dict()
        }

    except Exception as e:
        logger.error(f"Error in correlation_analysis: {str(e)}", exc_info=True)
        return {"error": str(e)}


# If you have any questions or suggestions, please feel free to reach out to me(rushi) or aman