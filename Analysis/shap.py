import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from logging_config import logger
from sklearn.metrics import accuracy_score

# This function performs SHAP (SHapley Additive exPlanations) analysis on an XGBoost regression model

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

    # Time-based train-test split (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train XGBoost regression model
    logger.info("XGB model training started")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    print("Model parameters:", model.get_params())
    #As of now model will be trained every time when we hit api......based on the requirements we can do it once in 20-30 days..But need to check with weekly updated data once
    
    # Perform SHAP analysis
    logger.info("SHAP Insights started")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    #We can also add shap.plot for summarizing the values through graphs

    # Compute mean absolute and mean signed SHAP values for feature impact and direction
    print(shap_values)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    mean_signed_shap = shap_values.values.mean(axis=0)
    
    # Summarize the impact and direction of each feature
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

    summary_text = " **SHAP-Based Feature Impact Summary on Units Sold**\n\n"
    for i, f in enumerate(feature_summary[:12], 1):
        summary_text += f"{i}. **{f['feature']}** – {f['direction'].capitalize()} impact (importance score: {f['impact']:.2f})\n"

    return summary_text
