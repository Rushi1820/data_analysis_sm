**This README provides steps to get you started with setting up the Moy project in your local environment.**

Prerequisites:
Ensure that you have the following installed on your machine:

git
Python
pip

**Steps**
Follow these steps to clone the project and install dependencies.

Step 1: Git Clone
First, clone the project repository from your command line:

git clone https://github.com/Rushi1820/data_analysis_sm.git

Step 2: Create Virtual env

python -m venv env         

Step 3: activate the env

.\env\Scripts\activate

Step 4: Install all dependencies

pip install -r requirements.txt

Step 5: Download the spacy model

python -m spacy download en_core_web_sm  

Step 5: run the application

uvicorn main:app --reload 

you will find the swagger document in the following link:

http://localhost:8000/docs

**If you have any questions or suggestions, please feel free to reach out to me(rushi) or aman**

