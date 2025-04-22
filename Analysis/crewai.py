from crewai import Agent, Task, Crew, Process, LLM
from langchain_google_genai import ChatGoogleGenerativeAI
import os


def run_analysis(timeseriesanalysis, correlationanalysis, regressionanalysis):

    GOOGLE_API_KEY = "add the api key "

    llm = LLM(
        model="gemini-1.5-pro",
        verbose=True,
        temperature=0.5,
        google_api_key=GOOGLE_API_KEY
    )

    # ---------------------------- AGENTS ---------------------------- #
    # Time Series Agent
    time_series_agent = Agent(
        role="Time Series Results Analyst",
        goal="You have to summarize the results of time series analysis, including trends, seasonality, anomalies, and outliers from the results shown in the data.",
        verbose=True,
        memory=True,
        backstory=(
            f"""
            You are a highly skilled time series analyst with deep expertise in interpreting analytical results from time series data. 
            Your task is to analyze the results of the time series analysis and provide a clear, concise, human-readable summary based on the data provided.

            Data: {timeseriesanalysis}

            Your job is to:
            - Summarize the overall sales trend, indicating whether it is generally increasing, decreasing, or remaining stable.
            - Interpret recent trends by analyzing recent time periods and identifying whether sales are on an upward or downward trajectory.
            - Analyze and describe seasonality, anomalies, and outliers observed in the data.
            - Summarize key statistical measures (e.g., mean, median) and highlight the most important insights from the data.
            - Dont repeat the same sentences or information repeatedly, make it clear, understandble, readbale.
            - Only use the input data no other random examples or assumptions.

            The output should be a human-readable summary, focusing solely on the interpretation of the results, with no additional suggestions or requests for further data.
            """
        ),
        llm=llm,
        allow_delegation=True
    )

    # Correlation Agent
    correlation_agent = Agent(
        role="Correlation Results Analyst",
        goal="Summarize and interpret the correlation analysis results, explaining the relationships between features and the target variable from the results shown in the data.",
        verbose=True,
        memory=True,
        backstory=(
            f"""
            You are a correlation results analysis expert, skilled at understanding the relationships between variables.
            Your task is to analyze the correlation results provided and generate a clear, concise summary of the relationships between features and the target variable.

            Data: {correlationanalysis}

            Your job is to:
            - Identify and explain the significant correlations between features and the target variable (Units Sold).
            - Interpret the strength and direction of these correlations (positive/negative).
            - Summarize which features have the most influence on the target variable.
            - Dont repeat the sentences or information repeatedly, make it clear, understandble, readbale.
            - Only use the input data no other random examples or assumptions.

            The output should be a human-readable summary, focusing solely on the interpretation of the correlations in the data, without additional suggestions or requests for further information.
            """
        ),
        llm=llm,
        allow_delegation=True
    )

    # Regression Agent
    regression_agent = Agent(
        role="Regression Results Analyst",
        goal="Summarize the results of regression analysis, explaining the impact of features on the target variable (Units Sold) from the results shown in the data.",
        verbose=True,
        memory=False, 
        backstory=(
                f"""
                You are a regression results analysis expert, skilled in interpreting regression outputs and model performance.
                Your task is to analyze the provided regression results and generate a clear, non-repetitive, concise summary of the relationship between features and the target variable.

                Data: {regressionanalysis}

                Guidelines:
                - Clearly summarize the regression coefficients, p-values, and R² scores to understand feature impacts on the target variable.
                - List statistically significant features (p < 0.05) and explain their impact (positive or negative).
                - Mention insignificant features briefly only once, if necessary.
                - DO NOT repeat any insights, phrases, or conclusions more than once.
                - DO NOT restate the same metric or explanation using different words repeatedly.
                - Use bullet points or clear paragraph separation if it helps clarity.
                - Keep the output structured and easy to read. Avoid redundancy.
                - Only use the input data, with no hypothetical scenarios or external assumptions.

                Output must be a concise human-readable summary focusing solely on interpreting the regression results.
                """
            ),
        llm=llm,
        allow_delegation=True
    )


    # ---------------------------- TASKS ---------------------------- #

    # Time Series Analysis Task
    research_task = Task(
        description=(
            "Analyze the provided time-series data to identify trends, seasonal components, anomalies, and patterns. "
            "Break down the data into its core components and provide actionable insights strictly based on the data."
        ),
        expected_output=(
            "A comprehensive report outlining time series decomposition, trends, seasonality, outliers, and business implications based on the provided data."
        ),
        agent=time_series_agent,
        async_execution=True,
    )

    # Correlation Analysis Task
    correlation_task = Task(
        description=(
            "Perform correlation analysis between all independent variables and the target variable using the provided dataset. "
            "Identify and explain strong correlations and their direction (positive/negative) based on the data."
        ),
        expected_output=(
            "A detailed explanation of the correlation matrix results with insights into which variables most influence the target variable, strictly based on the data."
        ),
        agent=correlation_agent,
        async_execution=True,
    )

    # Regression Analysis Task
    regression_task = Task(
        description=(
            "Perform a regression analysis using the input dataset. "
            "Interpret coefficients, p-values, and R² scores to explain feature influence on the target variable based on the data."
        ),
        expected_output=(
            "A detailed explanation of the regression model including variable significance, model fit, and predictive insights based on the data."
        ),
        agent=regression_agent,
        async_execution=True,
    )


    # Run the full crew process
    result1 = Crew(agents=[time_series_agent], tasks=[research_task]).kickoff()
    result2 = Crew(agents=[correlation_agent], tasks=[correlation_task]).kickoff()
    result3 = Crew(agents=[regression_agent], tasks=[regression_task]).kickoff()
    combined_results = [result1, result2, result3]

    return combined_results


# If you have any doubts or suggestions, please feel free to reach out to me(rushi) or aman