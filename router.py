from fastapi import APIRouter, HTTPException
import pandas as pd
from Analysis import preprocess, Timeanalysis, correlations, regression, genai,shap
from logging_config import logger

router = APIRouter()

@router.post("/ai_insights")
def ai_insights(user_query:str):
    
    try:
        logger.info("Preprocess started")
        model= preprocess.filtereddata(user_query)
        # Step 1: Preprocess Data
        df = preprocess.preprocess(model)
        logger.info("Time series analysis started")
        # Step 2: Time Series Analysis
        time_series_results = Timeanalysis.timeseries_analysis(df)
        logger.info("Correlations analysis started")
        # Step 3: Correlation Analysis
        correlation_results = correlations.correlation_analysis(df)
        logger.info("Regression analysis started")
        # Step 4: Regression Analysis
        regression_results = regression.regression_analysis(df)
        logger.info("genai summary started")
        shap_sales_insights= shap.shap_sales_insights(df)
        # Step 5: Generate AI Summary
        summary_text = genai.generate_summary(model,user_query,time_series_results, correlation_results, regression_results,shap_sales_insights)

        # Return all results
        return {
            "summary": summary_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
