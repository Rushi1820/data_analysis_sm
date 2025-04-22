import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from sklearn.preprocessing import LabelEncoder
from logging_config import logger

# This function performs SHAP analysis using all data for training
def shap_sales_insights(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Define the feature columns used for training
    features = [
        "Ad Spend (USD)", "Website Traffic (Visits)", "Click-Through Rate (CTR%)",
        "Conversion Rate (%)", "Customer Satisfaction Score", "Return Rate (%)",
        "Competitor Influence Score", "Retail Store Footfall"
    ]

    # Handle the optional categorical feature "Promotion Type"
    if "Promotion Type" in df.columns:
        df["Promotion Type"] = LabelEncoder().fit_transform(df["Promotion Type"].astype(str))
        features.append("Promotion Type")

    # Target variable
    target = "Units Sold"
    data = df[features + [target]].dropna()
    X = data[features]
    y = data[target]

    # Train LightGBM model on full dataset
    logger.info("LightGBM model training started")
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    print("Model parameters:", model.get_params())

    # SHAP analysis on the same full dataset
    logger.info("SHAP Insights started")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Compute mean absolute and mean signed SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    mean_signed_shap = shap_values.values.mean(axis=0)

    # Summarize impact and direction
    feature_summary = []
    for i, feature in enumerate(X.columns):
        impact = mean_abs_shap[i]
        direction = "positive" if mean_signed_shap[i] > 0 else "negative" if mean_signed_shap[i] < 0 else "neutral"
        feature_summary.append({
            "feature": feature,
            "impact": impact,
            "direction": direction
        })

    feature_summary.sort(key=lambda x: x["impact"], reverse=True)
    print(feature_summary)
    #Output:
    # {'feature': 'Competitor Influence Score', 'impact': np.float64(9.807922191266458), 'direction': 'positive'},
    # {'feature': 'Retail Store Footfall', 'impact': np.float64(9.43769400918851), 'direction': 'positive'}, 
    # {'feature': 'Click-Through Rate (CTR%)', 'impact': np.float64(8.924854260052959), 'direction': 'negative'}

    summary_text = " **SHAP-Based Feature Impact Summary on Units Sold**\n\n"
    for i, f in enumerate(feature_summary[:12], 1):
        summary_text += f"{i}. **{f['feature']}** – {f['direction'].capitalize()} impact (importance score: {f['impact']:.2f})\n"

    return summary_text

#  Currently all the models do run every time when we hit the api we can use the weights and decrease the latency.
# If you have any doubts or suggestions, please feel free to reach out to(rushi) or aman