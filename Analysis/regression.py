import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
import scipy.stats as stats

# Performs regression analysis to understand relationships in the data (no prediction).
def regression_analysis(df):
    df3 = df.copy()
    features = [
        "Total Revenue", "Website Traffic (Visits)", "Ad Spend (USD)",
        "Click-Through Rate (CTR%)", "Conversion Rate (%)", "Retail Store Footfall",
        "Customer Satisfaction Score", "Competitor Influence Score", "Return Rate (%)"
    ]
    target = "Units Sold"

    df3 = df3[features + [target]].dropna()

    X = df3[features]
    y = df3[target]

    # 1. Multiple Linear Regression (MLR) with statsmodels for interpretability
    model = sm.OLS(y, sm.add_constant(X)).fit()

    # 2. Ridge and Lasso for assessing regularization effects (coefficients + R²)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X, y)

    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X, y)


    results = {
        "MLR R²": float(model.rsquared),
        "Ridge R²": float(ridge_model.score(X, y)),
        "Lasso R²": float(lasso_model.score(X, y)),
        "MLR P-Values": {k: float(v) for k, v in model.pvalues.to_dict().items()},
        "MLR Coefficients": {k: float(v) for k, v in model.params.to_dict().items()},
        "Ridge Coefficients": dict(zip(X.columns, map(float, ridge_model.coef_))),
        "Lasso Coefficients": dict(zip(X.columns, map(float, lasso_model.coef_))),
        
    }
    print(results)
    # {'MLR R²': 0.6385810650225491, 
    # 'Ridge R²': 0.6385810650206561, 
    # 'Lasso R²': 0.638580350460648, 
    # 'MLR P-Values': {'const': 7.503150209572195e-20, 'Total Revenue': 0.0, 'Website Traffic (Visits)': 0.9628370984821323, 'Ad Spend (USD)': 0.3761220311269524, 'Click-Through Rate (CTR%)': 0.32180437623080665, 'Conversion Rate (%)': 0.18417524262025697, 'Retail Store Footfall': 0.9784541023346578, 'Customer Satisfaction Score': 0.42318518001819416, 'Competitor Influence Score': 0.07667248417219445, 'Return Rate (%)': 0.6683035301056861}, 
    # 'MLR Coefficients': {'const': 91.90698075494869, 'Total Revenue': 0.0007101524092396685, 'Website Traffic (Visits)': 4.3915066798222025e-07, 'Ad Spend (USD)': -8.365395660478403e-05, 'Click-Through Rate (CTR%)': -0.49734803936758226, 'Conversion Rate (%)': 0.8354235911540331, 'Retail Store Footfall': 6.3251902978740625e-06, 'Customer Satisfaction Score': 0.7429308526254563, 'Competitor Influence Score': -0.9285785932718499, 'Return Rate (%)': -0.40891188439425197}, 
    # 'Ridge Coefficients': {'Total Revenue': 0.0007101524030583186, 'Website Traffic (Visits)': 4.391804352397215e-07, 'Ad Spend (USD)': -8.365409450623031e-05, 'Click-Through Rate (CTR%)': -0.4973302471209688, 'Conversion Rate (%)': 0.8353803000334742, 'Retail Store Footfall': 6.3263466550954575e-06, 'Customer Satisfaction Score': 0.7428461098549711, 'Competitor Influence Score': -0.9285442228166244, 'Return Rate (%)': -0.40886028360379256}, 
    # 'Lasso Coefficients': {'Total Revenue': 0.000710146080548887, 'Website Traffic (Visits)': 4.6253210328503554e-07, 'Ad Spend (USD)': -8.374766382787414e-05, 'Click-Through Rate (CTR%)': -0.48284649497037785, 'Conversion Rate (%)': 0.8140307882860265, 'Retail Store Footfall': 6.891890223467702e-06, 'Customer Satisfaction Score': 0.6964843223455338, 'Competitor Influence Score': -0.912984446938324, 'Return Rate (%)': -0.35824472013345554}}

    return results


# If you have any doubts or suggestions please drop a message to me(rushi) or aman