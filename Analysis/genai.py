import google.generativeai as genai

# Initialize Gemini API
genai.configure(api_key="add the api key")

# Function to generate the summary based on all the results generated from all the analysis functions.
def generate_summary(model,user_query,complete_analytical_results,shap_sales_insights):
   
   #Prompt used for understanding the results and generate the summary. Currently chain of thought prompting applied...If better prompting please add in comments 
    prompt = f"""
          You are a **Senior Data Analyst** tasked with uncovering the **underlying reasons behind sales trends**.  
          Based on the **completely analyzed results**, you are provided with the following insights:
          {complete_analytical_results}

          ---

          ### **SHAP Analysis Findings (Feature Importance):**  
          {shap_sales_insights}

          ---
          ## ** Big Note**: you are not predicting data. You need to understand the results and do the deep analysis.
          ## **🧩 Comprehensive Analysis to Understand Sales Behavior:**

          1. **Understanding the Sales Trends and Behavior**:  
          - Carefully evaluate the **trends, peaks, and drops** in the sales data.  
          - Investigate if there are noticeable **seasonal patterns**, long-term trends, or **abrupt changes**.  
          - Focus on identifying the root cause of these patterns, be it due to **external events**, **consumer behavior**, **promotions**, **ad campaigns**, or other variables.  
          - If there are fluctuations, consider the **contextual factors** like **holidays, festivals**, or **sales events** that might have contributed to any increase or decrease in sales.  

          2. **Understanding Feature Relationships**:  
          - Analyze the **impact of key features** on the sales behavior.  
          - Investigate how features like **ad spend**, **conversion rate**, **website traffic**, and **promotion type** affect the sales data.  
          - Understand the **nature of these relationships**—whether positive or negative—and explain why they **impact** the sales.  
          - Investigate if the observed relationships suggest **direct effects** (e.g., more ad spend equals more sales) or **indirect effects** (e.g., sales improve due to a combination of multiple factors like conversion rate and ad spend).  
          - Investigate if these findings align with **known patterns** or reveal unexpected insights that could inform strategic decisions.

          3. **Understanding the SHAP Values (Feature Importance)**:  
          - Examine the SHAP values to understand the **contribution** of each feature to the sales behavior.  
          - Focus on determining which features were most **impactful** in driving positive or negative sales outcomes.  
          - For each feature, explain **why** it had such an impact, and determine whether it **supports or contradicts the initial assumptions** from other analyses.  
          - Investigate if any **unexpected variables** (e.g., customer satisfaction, competitor influence) played a more significant role than anticipated.  
          - Pay attention to **interactions** between features and how they collectively influence sales.

          4. **Synthesizing Insights from All Analyses**:  
          - Combine insights from the **completely analyzed results** to form a **holistic understanding** of the sales data.  
          - Seek to explain **why** sales behaved the way they did.  
          - If sales were high in certain periods, evaluate **whether external factors** (like promotions, seasonal events) played a role, or if **internal factors** (like website traffic or ad spend) were responsible.  
          - Understand if **sales peaks or dips** can be explained purely by **business activities** or if they were affected by **consumer behavior** shifts, changes in the economy, or **market trends**.  
          - Integrate findings from **SHAP** (how each feature contributes) and all the insights to create a unified explanation for sales fluctuations.

          ---

          ## ** Now, Answer the following query based on the analysis above:**  
          {user_query}

          ---

          ### **Important Notes for Answering the Query:**
          - **Do NOT repeat the user query.** Start with a **deep analysis** of the data and trends to **explain the underlying reasons** for the sales behavior.
          - Synthesize the findings from the **completely analyzed results** and **SHAP values** to **uncover** the factors driving the sales trends.
          - Use a **comprehensive approach** to explain the sales fluctuations by considering **external factors, consumer behavior, and internal business decisions**.
          - Focus on **data-driven reasoning** and avoid oversimplified answers or generalized assumptions.
          - Provide **insightful and specific reasons** for why sales occurred as they did, **without relying on hypothetical examples**.
          - The answer should be **clear, well-reasoned**, and **based on actual data analysis**, drawing connections between different findings and uncovering the true drivers of sales behavior.
          """





    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    
    return response.text


# Pending work: 
# - Work on Better prompting for clear and desired output summary


# If you have any doubts or suggestions please drop a message to me(rushi) or aman