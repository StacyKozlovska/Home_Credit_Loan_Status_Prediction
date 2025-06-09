from flask import Flask, request, redirect, url_for, render_template, jsonify
import pandas as pd
import random
import joblib
import os


preprocessor_pipeline = joblib.load('preprocessor_selected_full.joblib')
model_classifier = joblib.load('ensemble_model_final.joblib')
test_df = pd.read_csv("X_test_selected_full_raw.csv")
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]
original_selected_full_col_names = test_df.columns.tolist()

app = Flask(__name__)


input_features = [
    {'name': 'LOAN_INCOME_RATIO',
     'explanation': 'Share of the loan to client\'s income (AMT_CREDIT/AMT_INCOME_TOTAL)',
     'type': 'float64',
     'options': None},

    {'name': 'NUM_TOTAL_CREDITS',
     'explanation': 'Total credits mentioned in the Credit Bureau',
     'type': 'float64',
     'options': None},

    {'name': 'NAME_TYPE_SUITE',
     'explanation': 'Who was accompanying client when he was applying for the loan',
     'type': 'object',
     'options': ['Unaccompanied', 'Spouse, partner', 'Family', 'Children', 'Other', 'Not specified', 'Group of people']},

    {'name': 'SUM_CREDIT_MAX_OVERDUE_BURR',
     'explanation': 'Sum of overdue credits mentioned in the Credit Bureau',
     'type': 'float64',
     'options': None},

    {'name': 'NUM_PAST_DUE_INSTALL',
     'explanation': 'Number of past due installments',
     'type': 'float64',
     'options': None},

    {'name': 'NAME_EDUCATION_TYPE',
     'explanation': 'Level of highest education the client achieved',
     'type': 'object',
     'options': ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']},

    {'name': 'OCCUPATION_TYPE',
     'explanation': 'What kind of occupation does the client have',
     'type': 'object',
     'options': ['Cooking staff', 'Not specified', 'Laborers', 'Core staff', 'Security staff', 'Drivers', 'Sales staff',
                 'Accountants', 'High skill tech staff', 'Cleaning staff', 'Low-skill Laborers', 'Managers', 'Medicine staff',
                 'Private service staff', 'Secretaries', 'HR staff', 'Waiters/barmen staff', 'Realty agents', 'IT staff']},

    {'name': 'NUM_PREV_APPS',
     'explanation': 'Number of previous applications at Home Credit',
     'type': 'float64',
     'options': None},

    {'name': 'FLAG_PHONE',
     'explanation': 'Did client provide home phone (1=YES, 0=NO)',
     'type': 'int64',
     'options': [0, 1]},

    {'name': 'AVG_RATE_DOWN_PAYMENT',
     'explanation': 'Average rate of down payment calculated from previous applications',
     'type': 'float64',
     'options': None},

    {'name': 'NAME_FAMILY_STATUS',
     'explanation': 'Family status of the client',
     'type': 'object',
     'options': ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']},

    {'name': 'AVG_CREDIT_GOODS_RATIO',
     'explanation': 'Average credit goods price ratio from previous applications',
     'type': 'float64',
     'options': None},

    {'name': 'YEARS_REGISTRATION',
     'explanation': 'How many years before the application did client change their registration',
     'type': 'float64',
     'options': None},

    {'name': 'OBS_60_CNT_SOCIAL_CIRCLE',
     'explanation': 'How many observation of client\'s social surroundings with observable 60 DPD (days past due) default',
     'type': 'float64',
     'options': None},

    {'name': 'AGE',
     'explanation': 'Client\'s age in years',
     'type': 'float64',
     'options': None},

    {'name': 'YEARS_SINCE_LAST_APP',
     'explanation': 'How many years have passed since the last application',
     'type': 'float64',
     'options': None},

    {'name': 'CODE_GENDER',
     'explanation': 'Gender of the client',
     'type': 'object',
     'options': ['F', 'M']},

    {'name': 'YEARS_EMPLOYED',
     'explanation': 'Years of employment',
     'type': 'float64',
     'options': None},

    {'name': 'EXT_SOURCE_2',
     'explanation': 'Normalized score from external data source #2',
     'type': 'float64',
     'options': None},

    {'name': 'REGION_POPULATION_RELATIVE',
     'explanation': 'Normalized population of region where client lives (higher number means the client lives in more populated region)',
     'type': 'float64',
     'options': None},

    {'name': 'FLAG_WORK_PHONE',
     'explanation': 'Did client provide work phone (1=YES, 0=NO)',
     'type': 'int64',
     'options': [0, 1]},

    {'name': 'AMT_REQ_CREDIT_BUREAU_YEAR',
     'explanation': 'Number of enquiries to Credit Bureau about the client one day year',
     'type': 'float64',
     'options': None},

    {'name': 'AVG_CREDIT_ANNUITY_RATIO',
     'explanation': 'How many times the annuity payment fits into the total loan amount',
     'type': 'float64',
     'options': None},

         {'name': 'WEEKDAY_APPR_PROCESS_START',
     'explanation': 'On which day of the week did the client apply for the loan',
     'type': 'object',
     'options': ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']},

    {'name': 'NAME_CONTRACT_TYPE',
     'explanation': 'Identification if loan is cash or revolving',
     'type': 'object',
     'options': ['Revolving loans', 'Cash loans']},

    {'name': 'OWN_CAR_AGE',
     'explanation': 'Age of client\'s car',
     'type': 'float64',
     'options': None},

    {'name': 'LAST_CONTRACT_STATUS',
     'explanation': 'The status of the last loan contract',
     'type': 'object',
     'options': ['Approved', 'Canceled', 'Refused', 'Not specified', 'Unused offer']},

    {'name': 'CNT_FAM_MEMBERS',
     'explanation': 'How many family members does client have',
     'type': 'float64',
     'options': None},

    {'name': 'NAME_INCOME_TYPE',
     'explanation': 'Clients income type',
     'type': 'object',
     'options': ['Working', 'State servant', 'Pensioner',
                 'Commercial associate', 'Businessman',
                 'Student', 'Unemployed', 'Maternity leave']},

    {'name': 'YEARS_SINCE_LAST_BURR_CREDIT',
     'explanation': 'How many years have passed since the last credit mentioned in the redit Bureau',
     'type': 'float64',
     'options': None},

    {'name': 'YEARS_LAST_PHONE_CHANGE',
     'explanation': 'How many years before application did client change phone',
     'type': 'float64',
     'options': None},

    {'name': 'ORGANIZATION_TYPE',
     'explanation': 'Type of organization where client works',
     'type': 'object',
     'options': ['Business Entity Type 3', 'Other', 'Security Ministries', 'XNA', 'Transport: type 4',
                 'Business Entity Type 1', 'Restaurant', 'Government', 'Medicine', 'Self-employed',
                 'Kindergarten', 'School', 'Industry: type 10', 'Postal', 'Industry: type 9', 'Construction',
                 'Industry: type 4', 'Trade: type 3', 'Trade: type 7', 'Business Entity Type 2', 'Security',
                 'Industry: type 1', 'Industry: type 11', 'Bank', 'Agriculture', 'Services', 'Transport: type 3',
                 'Transport: type 2', 'Culture', 'Industry: type 3', 'Police', 'Hotel', 'Emergency', 'Industry: type 7',
                 'Military', 'Industry: type 2', 'Industry: type 5', 'University', 'Industry: type 12', 'Housing',
                 'Industry: type 13', 'Trade: type 2', 'Electricity', 'Mobile', 'Trade: type 6', 'Telecom',
                 'Trade: type 1', 'Transport: type 1', 'Legal Services', 'Insurance', 'Cleaning', 'Industry: type 6',
                 'Advertising', 'Realtor', 'Trade: type 5', 'Religion', 'Trade: type 4', 'Industry: type 8']},

    {'name': 'YEARS_ID_PUBLISH',
     'explanation': 'How many years before the application did client change the identity document with which he applied for the loan',
     'type': 'float64',
     'options': None},

    {'name': 'REG_CITY_NOT_LIVE_CITY',
     'explanation': 'Flag if client\'s permanent address does not match contact address (1=different, 0=same, at city level)',
     'type': 'int64',
     'options': [0, 1]},

    {'name': 'HOUR_APPR_PROCESS_START',
     'explanation': 'Approximately at what hour did the client apply for the loan',
     'type': 'int64',
     'options': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]},

    {'name': 'NAME_HOUSING_TYPE',
     'explanation': 'What is the housing situation of the client', 
     'type': 'object',
     'options': ['House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment', 'Office apartment', 'Co-op apartment']},

    {'name': 'FLAG_DOCUMENT_13',
     'explanation': 'Did client provide document 13',
     'type': 'int64', 
     'options': [0, 1]}, 

    {'name': 'AMT_CREDIT',
     'explanation': 'Credit amount of the loan',
     'type': 'float64', 
     'options': None},
    
    {'name': 'APPS_APPROVAL_RATE',
     'explanation': 'Application approval rate from previous applications',
     'type': 'float64', 
     'options': None}, 

    {'name': 'FLAG_DOCUMENT_11',
     'explanation': 'Did client provide document 11',
     'type': 'int64', 
     'options': [0, 1]},
    
    {'name': 'LAST_BURR_CREDIT_STATUS',
     'explanation': 'Status of the last credit from Credit Bureau',
     'type': 'object', 
     'options': ['Not specified', 'Closed', 'Active', 'Sold', 'Bad debt']},
    
    {'name': 'FLAG_OWN_REALTY',
     'explanation': 'Flag if client owns a house or flat',
     'type': 'int64', 
     'options': [0, 1]},
    
    {'name': 'EXT_SOURCE_3',
     'explanation': 'Normalized score from external data source #3',
     'type': 'float64', 
     'options': None},
    
    {'name': 'FLAG_DOCUMENT_18',
     'explanation': 'Did client provide document 18',
     'type': 'int64', 
     'options': [0, 1]},
    
    {'name': 'FLAG_OWN_CAR',
     'explanation': 'Flag if the client owns a car',
     'type': 'int64', 
     'options': [0, 1]},
    
    {'name': 'REGION_RATING_CLIENT_W_CITY',
     'explanation': 'Our rating of the region where client lives with taking city into account (1,2,3)',
     'type': 'int64', 
     'options': [1, 2, 3]},
    
    {'name': 'AMT_INCOME_TOTAL',
     'explanation': 'Income of the client',
     'type': 'float64', 
     'options': None},
    
    {'name': 'FLAG_DOCUMENT_3',
     'explanation': 'Did client provide document 3',
     'type': 'int64', 
     'options': [0, 1]},
    
    {'name': 'AVG_DOWN_PAYMENT',
     'explanation': 'The initial payment made by the borrower when obtaining credit for a specific purchase',
     'type': 'float64', 
     'options': None},
]

def preprocess_df(df):
    print("Dataframe shape before preprocessing:", df.shape)
    processed_df = pd.DataFrame(preprocessor_pipeline.transform(df),
                                columns=original_selected_full_col_names)
    print("Dataframe shape after preprocessing:", processed_df.shape)
    return processed_df


def preprocessDataAndPredict(df):
    processed_df = preprocess_df(df)
    print("Processed DataFrame:")
    print(processed_df)

    print("processed_df no reshape", processed_df.shape)

    prediction_status = model_classifier.predict(processed_df)[0]

    return prediction_status


def get_form_data(request, input_features):
    form_data = {}
    for feature in input_features:
        name = feature['name']
        type = feature['type']

        if type == 'int64':
            form_data[name] = int(request.form[name])
        elif type == 'float64':
            form_data[name] = float(request.form[name])
        elif type == 'object':
            form_data[name] = request.form[name]

    return form_data


@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if 'predict_custom' in request.form:
            form_data = get_form_data(request, input_features)
            df = pd.DataFrame([form_data])
            prediction = preprocessDataAndPredict(df)
            print("prediction", prediction)

            return redirect(url_for('predict',
                                    prediction=prediction))

        elif 'predict_random' in request.form:
            idx = random.randint(0, len(test_df)-1)
            sample = test_df.iloc[[idx]]
            prediction = preprocessDataAndPredict(sample)
            print("prediction", prediction)

            return redirect(url_for('predict',
                                    prediction=prediction))

    return render_template('index.html', input_features=input_features)


@app.route('/predict/')
def predict():
    prediction = request.args.get('prediction')

    return render_template('predict.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
