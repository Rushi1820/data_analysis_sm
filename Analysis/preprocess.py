import numpy as np
import pandas as pd
import spacy
import pandas as pd
from Levenshtein import distance as levenshtein_distance


#Function for filtering the data to determine the model name from the user_query
def filtereddata(user_query):
    #import model from spacy
    nlp = spacy.load("en_core_web_sm")

    df = pd.read_csv("mobile_sales_data.csv")
    
    #COnvert all the columns to lowercase for better understanding of word
    df.columns = df.columns.str.lower()
    df["model"] = df["model"].str.lower().str.strip()

    doc = nlp(user_query)
    #Checking the model by dividing the user_query into token and matching it with all the letters, numbers to find exact match
    keywords = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN", "NUM"]]
    extracted_model = " ".join(keywords).strip()

    if not extracted_model:
        print("Error: No model extracted from the query.")
        exit()

    unique_models = df["model"].dropna().unique()
    #Using Levenshtein distance matching the keyword and returning it
    matched_model = min(unique_models, key=lambda x: levenshtein_distance(extracted_model, x))


    print(f"Extracted Model: {matched_model}")

    

    return matched_model

#Function to preprocess the data and aquire the relevant data from entire dataset
def preprocess(matched_model):
    #read the dataset
    df1 = pd.read_csv("mobile_sales_data.csv")
    # Normalize model column for case-insensitive filtering
    df1["Model"] = df1["Model"].str.lower().str.strip()

    #Only getting the model data for ensuring the sales is only for resepected model
    filtered_data = df1[df1["Model"] == matched_model]

    filtered_data["Date"] = pd.to_datetime(filtered_data["Date"])
    filtered_data = filtered_data.sort_values("Date")
    df = filtered_data
    print(df)
    df['Date'] = pd.to_datetime(df['Date'])

    df.fillna(method='ffill', inplace=True)  
    df.fillna(method='bfill', inplace=True)  

    # Define numeric columns for outlier detection and smoothing
    numerical_columns = ['Units Sold', 'Total Revenue', 'Website Traffic (Visits)', 'Ad Spend (USD)'] 

    #Why are using this:
    # It detects and smooths outliers in 4 key numerical columns:

        # 'Units Sold'

        # 'Total Revenue'

        # 'Website Traffic (Visits)'

        # 'Ad Spend (USD)'

    # Outliers can seriously skew analysis, especially in sales data. For example:

        # A random spike in website visits due to a bot attack.

        # Incorrect entries like $999999 ad spend.

        # Data entry mistakes or one-time promo spikes.

        # These don’t represent normal trends and can mislead models like XGBoost or visualizations.
        
        #  if not required we can skip it 
        
    for column in numerical_columns:
         # Calculate 7-day rolling mean
        rolling_mean = df[column].rolling(window=7, min_periods=1).mean()  

         # Define threshold as 3 times standard deviation
        threshold = 3 * df[column].std()  

         # Flag outliers
        df['outlier_' + column] = (abs(df[column] - rolling_mean) > threshold)

        # Replace outliers with the rolling mean
        df.loc[df['outlier_' + column], column] = rolling_mean[df['outlier_' + column]]
    # Drop intermediate outlier flag columns
    df.drop(columns=[col for col in df.columns if col.startswith('outlier_')], inplace=True)

    return df


