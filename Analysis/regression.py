import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats

# This function performs a comprehensive regression analysis to understand the impact of various marketing and sales metrics
def regression_analysis(df):
    df3=df.copy()
    features = [
            "Total Revenue", "Website Traffic (Visits)", "Ad Spend (USD)",
            "Click-Through Rate (CTR%)", "Conversion Rate (%)", "Retail Store Footfall",
            "Customer Satisfaction Score", "Competitor Influence Score", "Return Rate (%)"
    ]
    target = "Units Sold"

    df3 = df3[features + [target]].dropna()

    X = df3[features]
    y = df3[target]
    #if we are looking for only training then change the below division to train params only
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # 1. Multiple Linear Regression (MLR) using statsmodels to get coefficients and p-values.
    model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    y_pred = model.predict(sm.add_constant(X_test))
    
    # 2. Ridge and Lasso regression using scikit-learn for regularized models and comparison of R² scores.
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_ridge_pred = ridge_model.predict(X_test)
    
    # 3. Performance evaluation using Mean Absolute Error and Root Mean Squared Error for MLR.
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)
    y_lasso_pred = lasso_model.predict(X_test)

    results = {
        "MLR R²": model.rsquared,
        "Ridge R²": r2_score(y_test, y_ridge_pred),
        "Lasso R²": r2_score(y_test, y_lasso_pred),
        "MLR P-Values": model.pvalues.to_dict(),
        "MLR Coefficients": model.params.to_dict(),
        "Mean Absolute Error (MLR)": mean_absolute_error(y_test, y_pred),
        "Root Mean Squared Error (MLR)": np.sqrt(mean_squared_error(y_test, y_pred))
    }
    # 4. ANOVA test to compare the mean units sold between low and high return rate groups to understand if return rate significantly affects sales.
    f_stat, p_value = stats.f_oneway(
        df3[df3["Return Rate (%)"] < 5]["Units Sold"],
        df3[df3["Return Rate (%)"] >= 5]["Units Sold"]
    )
    results["ANOVA F-Stat"] = f_stat
    results["ANOVA P-Value"] = p_value

    return results
