from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)  # Add supports_credentials=True

# Initialize df_data outside of the try block
default_training_data = None

# Default Training Data Normalization
try:
    with open('cvd_training_data.json', 'r') as file:
         # Add this line to print the content
        file_content = file.read()
        data = json.loads(file_content)
        
    # Assuming 'json_data' is the key in your JSON structure
    default_training_data = data.get('cvd_data', [])

    
    
except FileNotFoundError:
    print("File 'cvd_training_data.json' not found.")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")


@app.route('/api/data', methods=['POST'])
def classify_cvd():
    #Recieve the Two dimensional Json from the Flutter App
    request_data = request.json
    # print(data[0].keys())
    # print(data[1].keys())
    prediction_data_list = request_data[0]["predictionData"]

    for data_point in prediction_data_list:
        if "sex" in data_point:
            # Assuming 1 represents male and 0 represents female
            data_point["sex"] = 1 if data_point["sex"].lower() == "male" else 0

    training_data_list = request_data[1]["trainingDataList"]

    for data_point in training_data_list:
        if "sex" in data_point:
            # Assuming 1 represents male and 0 represents female
            data_point["sex"] = 1 if data_point["sex"].lower() == "male" else 0

    #Join the Data of the cvd_training_data.json and the data from the database
    joined_data = default_training_data + training_data_list
    
    #Convert Data to DataFrame
    trainingData_for_filter = pd.DataFrame(joined_data)
    predictionData = pd.DataFrame(prediction_data_list)

    #Filter data 40 to 74 years of age
    trainingData = trainingData_for_filter[(trainingData_for_filter['age'] >= 20) & (trainingData_for_filter['age'] <= 74)]

    #============Data Transformation=======================

    # Convert Age to brackets
    age_br = [20, 44, 49, 54, 59, 64, 69, 80]
    age_lb = [1, 2, 3, 4, 5, 6, 7]

    trainingData['age_bracket'] = pd.cut(trainingData['age'], bins=age_br, labels=age_lb, right=False, ordered=False)
    predictionData['age_bracket'] = pd.cut(predictionData['age'], bins=age_br, labels=age_lb, right=False, ordered=False)


    trainingData = trainingData.drop('age', axis=1)
    predictionData = predictionData.drop('age', axis=1)

    # Convert Systolic to brackets
    systolic_br = [0, 120, 139, 159, 179, float('inf')]
    systolic_lb = [1, 2, 3, 4, 5]
    trainingData['sbp_bracket'] = pd.cut(trainingData['SBP'], bins=systolic_br, labels=systolic_lb, right=False)
    predictionData['sbp_bracket'] = pd.cut(predictionData['SBP'], bins=systolic_br, labels=systolic_lb, right=False)

    trainingData = trainingData.drop('SBP', axis=1)
    predictionData = predictionData.drop('SBP', axis=1)

    # Convert Diastolic to brackets
    diastolic_br = [0, 80, 89, 90, float('inf')]
    diastolic_lb = [1, 2, 3, 4]
    trainingData['dbp_bracket'] = pd.cut(trainingData['DBP'], bins=diastolic_br, labels=diastolic_lb, right=False)
    predictionData['dbp_bracket'] = pd.cut(predictionData['DBP'], bins=diastolic_br, labels=diastolic_lb, right=False)

    trainingData = trainingData.drop('DBP', axis=1)
    predictionData = predictionData.drop('DBP', axis=1)

    # Convert BMI to brackets
    bmi_br = [0, 20, 24, 29, 35, float('inf')]
    bmi_lb = [1, 2, 3, 4, 5]
    trainingData['bmi_bracket'] = pd.cut(trainingData['body_mass_index'], bins=bmi_br, labels=bmi_lb, right=False)
    predictionData['bmi_bracket'] = pd.cut(predictionData['body_mass_index'], bins=bmi_br, labels=bmi_lb, right=False)

    trainingData = trainingData.drop('body_mass_index', axis=1)
    predictionData = predictionData.drop('body_mass_index', axis=1)

    # Convert Total cholestrol to brackets
    tc_br = [0, 4, 4.9, 5.9, 6.9, float('inf')]
    tc_lb = [1, 2, 3, 4, 5,]
    trainingData['tc_bracket'] = pd.cut(trainingData['total_cholesterol'], bins=tc_br, labels=tc_lb, right=False)
    predictionData['tc_bracket'] = pd.cut(predictionData['total_cholesterol'], bins=tc_br, labels=tc_lb, right=False)

    trainingData = trainingData.drop('total_cholesterol', axis=1)
    predictionData = predictionData.drop('total_cholesterol', axis=1)

    #Bracket waist circumference for training data
    trainingData['waist_circumference'] = np.where(
    ((trainingData['sex'] == 0) & (trainingData['waist_circumference (cm)'] < 80))
    | ((trainingData['sex'] == 1) & (trainingData['waist_circumference (cm)'] < 90)),
    0,
    1
    )

    trainingData = trainingData.drop('waist_circumference (cm)', axis=1)

    #Bracket waist circumference for prediction data
    predictionData['waist_circumference'] = np.where(
    ((predictionData['sex'] == 0) & (predictionData['waist_circumference (cm)'] < 80))
    | ((predictionData['sex'] == 1) & (predictionData['waist_circumference (cm)'] < 90)),
    0,
    1
    )

    predictionData = predictionData.drop('waist_circumference (cm)', axis=1)
    

    #=====================End of Data Transformation========================

    # scramble the rows in the training dataframe to prevent biases
    trainingData_scrambled = trainingData.sample(frac=1, random_state=42)
    print('----------------scrambled data---------------')
    print(trainingData_scrambled)

    #Handle null values or Normalize the data
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(trainingData_scrambled), columns=trainingData_scrambled.columns)


    df_imputed['fh_diabetes'] = df_imputed['fh_diabetes'].astype(int)
    df_imputed['fh_stroke'] = df_imputed['fh_stroke'].astype(int)
    df_imputed['fh_hypertension'] = df_imputed['fh_hypertension'].astype(int)
    df_imputed['fh_asthma'] = df_imputed['fh_asthma'].astype(int)
    df_imputed['fh_cancer'] = df_imputed['fh_cancer'].astype(int)
    df_imputed['fh_heartdisease'] = df_imputed['fh_heartdisease'].astype(int)
    df_imputed['fh_kidneydisease'] = df_imputed['fh_kidneydisease'].astype(int)
    df_imputed['sex'] = df_imputed['sex'].astype(int)
    df_imputed['smoker_status'] = df_imputed['smoker_status'].astype(int)
    df_imputed['alcohol_intake'] = df_imputed['alcohol_intake'].astype(int)
    df_imputed['physical_activity'] = df_imputed['physical_activity'].astype(int)
    df_imputed['nutrition_assessment'] = df_imputed['nutrition_assessment'].astype(int)
    df_imputed['hypertension'] = df_imputed['hypertension'].astype(int)
    df_imputed['diabetes'] = df_imputed['diabetes'].astype(int)
    df_imputed['risk_level'] = df_imputed['risk_level'].astype(int)


    # Split the data into features (X) and target variable (y)
    if 'risk_level' in trainingData_scrambled.columns:
        X = df_imputed.drop('risk_level', axis=1)
        y = df_imputed['risk_level']
    else:
        # Handle the case where 'risk_level' column is not present in the DataFrame
        print("Error: 'risk_level' column not found in trainingData_scrambled.")


    # Split the data into training and testing sets 20% of each
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ======================Feature Selection==========================
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    # Feature elimination using Recursive Feature Elimination (RFE)
    #rf_model = RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, n_estimators=100)
    rf_model = GradientBoostingClassifier(max_depth = 10, min_samples_leaf = 1, min_samples_split = 10, n_estimators = 100 )

    rfe = RFE(estimator=rf_model, n_features_to_select=11)  # Adjust the number of features as needed
    X_train_rfe = rfe.fit_transform(X_train_resampled, y_train_resampled)
    X_test_rfe = rfe.transform(X_test)

    selected_features = pd.DataFrame(rfe.support_, index=X_train.columns, columns=['Selected'])
    selected_features = selected_features[selected_features['Selected']].index.tolist()

    # ======================Feature Selection End=======================

    # ======================Prediction========================

    rf_model.fit(X_train_rfe, y_train_resampled)
    dataToPredict = predictionData[selected_features].values.tolist()
    prediction = rf_model.predict(dataToPredict).tolist()

    # Calculate accuracy

    kfold = KFold(n_splits=4, random_state=0, shuffle=True)
    scoring = ['precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovo_weighted']
    cv_results = cross_validate(rf_model, X_train_rfe, y_train_resampled, cv=kfold, scoring=scoring)

    print(f'Precision: {cv_results["test_precision_weighted"].mean():.4f}')
    print(f'Recall: {cv_results["test_recall_weighted"].mean():.4f}')
    print(f'F1 Score: {cv_results["test_f1_weighted"].mean():.4f}')
    print(f'AUC-ROC: {cv_results["test_roc_auc_ovo_weighted"].mean():.4f}')

    result = {
        'Precision': cv_results["test_precision_weighted"].mean(),
        'recall': cv_results["test_recall_weighted"].mean(),
        'F1_Score': cv_results["test_f1_weighted"].mean(),
        'AUC': cv_results["test_roc_auc_ovo_weighted"].mean(),
        'classification_result': prediction,
        'features': selected_features,
        'test_accuracy': cv_results["test_precision_weighted"].mean()
    }
    

    return jsonify(result)   
        
 
# Add CORS headers to allow all origins, methods, and headers
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'  # Add 'Content-Type' to the allowed headers
    return response

if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.7', port=5001)
