# Home Credit loan status prediction project

## Introduction:
Home Credit is a financial institution that specializes in providing consumer loans to individuals with limited or no credit history. The company operates in multiple countries and focuses on offering responsible and accessible financial services, particularly to unbanked or underbanked populations. Home Credit leverages non-traditional data sources, such as mobile phone usage and other alternative data, to assess the creditworthiness of applicants.

## Dataset link: 
https://www.kaggle.com/c/home-credit-default-risk/data

## Deployed Flask app link: 
https://home-credit-loan-prediction.uc.r.appspot.com/

## Main Motivation:
We want to help Home Credit to fully automate their decisions and thus save time, resources, and avoid lending money to risky clients.

## Repository files description:
1. ``home_credit_EDA_statistical_inference.ipynb`` - Jupyter notebook with Exploratory data analysis (EDA), data visualization, data preprocessing, and statistical inference.
   **Main objectives of the notebook:**
   - Perform data exploration and data cleaning.
   - Handle outliers, missing values, and duplicates.
   - Find and handle correlated features.
   - Perform feature engineering and preprocessing for further ML modeling.
   - Visualize the data.
   - Answer research questions with the data.
   - Perform statistical inference.

2. ``home_credit_ML.ipynb`` - Jupyter notebook with ML modeling code.
   **Main objectives of the notebook:**
   - Create a preprocessing Pipeline
   - Create baseline models
   - Try out various algorithms for the classification of the Loan status
   - Perform model selection, tune hyperparameters for the best-performing models, and test those models.
   - Deploy the Flask app with the best-performing model to Google Cloud Platform.
   - Try out a deep NN model to predict the loan status.
     
3. **homecredit_app** (folder with the Flask app code):
   1) **templates** (folder with HTML files for the Flask app):
       - ``index.html`` (home page code)
       - ``predict.html`` (prediction page code)
   2) ``main.py`` - main Flask app file.
   3) ``requirements.txt`` - libraries used by the Flask app.
   4) ``app.yaml`` - a configuration file used by Google App Engine to define the runtime, environment settings, and other deployment-related configurations for the application.
   5) ``Dockerfile`` - a script used to create a Docker image for the application. It specifies the base image, installs dependencies, and sets up the environment needed for the application to run.
   6) ``ensemble_model_final.joblib`` - best-performing ML model, trained in home_credit_ML.ipynb.
   7) ``preprocessor_selected_full.joblib`` - preprocessor for input data, fitted in home_credit_ML.ipynb.
   8) ``X_test_selected_full_raw.csv`` - test dataframe for getting a prediction with random test input predictions.
