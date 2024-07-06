# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 2. Load Data
train_url = 'https://raw.githubusercontent.com/WHPAN0108/BHT-DataScience-S24/main/classification/data/Assigment/aug_train.csv'
test_url = 'https://raw.githubusercontent.com/WHPAN0108/BHT-DataScience-S24/main/classification/data/Assigment/aug_test.csv'

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

print(train_df.head())
print(test_df.head())

# 3. Data Preparation
# to confirmm  the dataset was loaded correctly and to understand its structure before proceeding with further analysis
print(train_df.shape)
print(test_df.shape)

# get a concise summary of a DataFrame. Includes information about the DataFrame's index, column names, non-null counts, and data types of each column.
print(train_df.info())
print(test_df.info())

# Task1 Data clean, imputation
# Imputation = Verfahren, mit denen fehlende Daten in statistischen Erhebungen
# – die sogenannten Antwortausfälle – in der Datenmatrix vervollständigt werden.

# 1. in experience, replace >20 to 21; <1 to 1, and convert this as a numerical column
train_df['experience'] = train_df['experience'].replace({'>20': '21', '<1': '1'}).astype(float)
test_df['experience'] = test_df['experience'].replace({'>20': '21', '<1': '1'}).astype(float)
# use of float instead of int as float can handle NaN values at this point

# control check
print(train_df.info())
print(test_df.info())

# 2. in last_new_job, replace >4 to 5; never to 0, and convert this as a numerical column
train_df['last_new_job'] = train_df['last_new_job'].replace({'>4': '5', 'never': '0'}).astype(float)
test_df['last_new_job'] = test_df['last_new_job'].replace({'>4': '5', 'never': '0'}).astype(float)
# use of float instead of int as float can handle NaN values at this point

# control check
print(train_df.info())
print(test_df.info())

# Export DataFrame to CSV (just to double check)
#train_df.to_csv('C:/Users/User-1/Documents/Master Medieninformatik Beuth/Data Science/SS 24/Einsendeaufgabe 9/train_data.csv', index=False)
#test_df.to_csv('C:/Users/User-1/Documents/Master Medieninformatik Beuth/Data Science/SS 24/Einsendeaufgabe 9/test_data.csv', index=False)

# 3. If the column is categorical, impute the missing value as its mode. If the column is numerical,
# impute the missing value as its median

def impute_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0]) # mode() is used to fill missing values in a DataFrame column with the most frequently occurring value in that column
        else:
            df[column] = df[column].fillna(df[column].median())
    return df

train_df = impute_missing_values(train_df)
test_df = impute_missing_values(test_df)

# Export DataFrame to CSV (just to double check)
#train_df.to_csv('C:/Users/User-1/Documents/Master Medieninformatik Beuth/Data Science/SS 24/Einsendeaufgabe 9/train_data_imputed.csv', index=False)
#test_df.to_csv('C:/Users/User-1/Documents/Master Medieninformatik Beuth/Data Science/SS 24/Einsendeaufgabe 9/test_data_imputed.csv', index=False)


#Task2 Classification
# Goal: predict the probability of a candidate looking for a new job or will continue working for the company,
# as well as interpreting affected factors on employee decision.
# Therefore, value that we wanna predict:
# column target: 0 – Not looking for job change, 1 – Looking for a job change

#----------------
# just as background info for myself:
# Ensembling = Ensemble-Methoden werden in der Statistik und für Machine Learning eingesetzt.
# Sie nutzen eine endliche Menge von verschiedenen Lernalgorithmen, um bessere Ergebnisse zu erhalten
# als mit einem einzelnen Lernalgorithmus.
#----------------

# 1. Build a classification model from the training set ( you can use any algorithms)

# Encode categorical features to receive numerical features
# -------
# Background: there are e.g methods Label Encoder, One hot encoder or pandas dummy.
# Chosen approach will be decided by how many unique values column have.
# -------
# Label Encoding assigns a unique integer to each category. This approach works well when:
# - The categorical variable is ordinal, meaning the categories have a meaningful order
# (e.g., low, medium, high).
# - The number of unique values (categories) is relatively small.
# Disadvantages:
# Imparts a sense of ordinal relationship even if none exists, which can mislead certain
# algorithms (e.g., distance-based methods like KNN).
# -----
# One-Hot Encoding:
# creates a new binary column for each category. This approach is preferred when:
# - The categorical variable is nominal
# - The number of unique values is moderate.
# -----
# nominal data = Den Variablen der Nominalskala ist kein quantitativer Wert zugeordnet.
# Bei den Variablen handelt es sich um Attribute, und es besteht keine Notwendigkeit,
# sie in eine Reihenfolge oder Hierarchie zu bringen. (Bsp. Geschlecht, Herkunft, Familienstand, Land...)

# Conclusion: As the remaining categorical data is mainly nominal data, I chose the One Hot Encoding Method!

# Select categorical columns for one-hot encoding
categorical_columns = train_df.select_dtypes(include=['object']).columns

# Apply one-hot encoding
encoder = OneHotEncoder(drop='first', sparse_output=False)
train_encoded = pd.DataFrame(encoder.fit_transform(train_df[categorical_columns]), index=train_df.index)
test_encoded = pd.DataFrame(encoder.transform(test_df[categorical_columns]), index=test_df.index)

# Assign column names to the encoded DataFrames
train_encoded.columns = encoder.get_feature_names_out(categorical_columns)
test_encoded.columns = encoder.get_feature_names_out(categorical_columns)

# Add encoded columns to the dataframes and drop the original categorical columns
train_df = train_df.drop(categorical_columns, axis=1)
train_df = pd.concat([train_df, train_encoded], axis=1)

test_df = test_df.drop(categorical_columns, axis=1)
test_df = pd.concat([test_df, test_encoded], axis=1)

# Verify the results
print(train_df.head())
print(test_df.head())

# Export DataFrame to CSV (just to double check)
# train_df.to_csv('C:/Users/User-1/Documents/Master Medieninformatik Beuth/Data Science/SS 24/Einsendeaufgabe 9/train_data_encoded.csv', index=False)
# test_df.to_csv('C:/Users/User-1/Documents/Master Medieninformatik Beuth/Data Science/SS 24/Einsendeaufgabe 9/test_data_encoded.csv', index=False)


# Split of data into test and training data set is not necessary anymore, as it is already provided!
# value that we wanna predict:
# column target: 0 – Not looking for job change, 1 – Looking for a job change

# Separate features and target variable
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

# Train a Random Forest model with specified parameters
RFmodel = RandomForestClassifier(n_estimators=100, max_features=3, random_state=42)
RFmodel.fit(X_train, y_train)

# Applying the model in the training set and generating the prediction
y_train_pred = RFmodel.predict(X_train)

# 2. generate the confusion matrix and calculate the accuracy, precision, recall, and F1-score on training set.
train_cm = confusion_matrix(y_train, y_train_pred)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

print(f"Training set results:\nConfusion Matrix:\n{train_cm}\nAccuracy: {train_accuracy}\nPrecision: {train_precision}\nRecall: {train_recall}\nF1 Score: {train_f1}")

# Visualization confusion matrix
train_cm = confusion_matrix(y_train, y_train_pred,labels=RFmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=train_cm,
                               display_labels=RFmodel.classes_)
disp.plot()
plt.show()

# 3. Applying the model in the test set and generating the prediction
y_test_pred = RFmodel.predict(X_test)

# 4. generate the confusion matrix from the test set and calculate the accuracy, precision, recall, and F1-score
test_cm = confusion_matrix(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Test set results:\nConfusion Matrix:\n{test_cm}\nAccuracy: {test_accuracy}\nPrecision: {test_precision}\nRecall: {test_recall}\nF1 Score: {test_f1}")

# Visualization confusion matrix
test_cm = confusion_matrix(y_test, y_test_pred,labels=RFmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=test_cm,
                               display_labels=RFmodel.classes_)
disp.plot()
plt.show()

# 5. compare the results between the training and test set
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Training Set': [train_accuracy, train_precision, train_recall, train_f1],
    'Test Set': [test_accuracy, test_precision, test_recall, test_f1]
})

print(comparison)

# Detailed interpretations
if train_accuracy > test_accuracy:
    print("The training set accuracy is higher than the test set accuracy. This might indicate that the model is overfitting, model is overfitting, meaning it performs well on the training data but poorly on unseen data")
else:
    print("The training set accuracy is lower than or equal to the test set accuracy. This might indicate that the model generalizes well.")

if train_precision > test_precision:
    print("The training set precision is higher than the test set precision. This indicates that the model is better at predicting positive instances in the training set compared to the test set.")
else:
    print(
        "The training set precision is lower than or equal to the test set precision. This might indicate that the model generalizes well.")

if train_recall > test_recall:
    print("The training set recall is higher than the test set recall. This indicates that the model it implies that the model captures more actual positive instances in the training set compared to the test set.")
else:
    print("The training set recall is lower than or equal to the test set recall. This might indicate that the model generalizes well.")

if train_f1 > test_f1:
    print("The training set F1 score is higher than the test set F1 score. This indicates that the model has overall a better performance on the training data. A significant difference between training and test F1 scores suggests overfitting.")
else:
    print("The training set F1 score is lower than or equal to the test set F1 score. This might indicate that the model generalizes well.")

# Extra point: think about what kind of the method can increase the performance (does not need to run):
# Random Forest Model Tune & balance the imbalanced original data of test and training data beforehands.
# Specifying parameters like n_estimators and max_features can impact the performance of the Random Forest model.
# These parameters control the number of trees in the forest and the number of features to consider when looking for
# the best split. Adding these parameters allows us to fine-tune the model for potentially better performance.
# But also a comparision of model performance before deciding on which model to use can be helpful and increase the performance'
# e.g. with using kfold method.

# Example:
parameters_for_testing = {
"n_estimators"    : [50,100, 200] ,
 "max_features"        : [2, 3, 5, 10],
}
RFmodel = RandomForestClassifier()

kfold = KFold(n_splits=5)
grid_cv = GridSearchCV(estimator=RFmodel, param_grid=parameters_for_testing, scoring='accuracy', cv=kfold)
result = grid_cv.fit(X_train, y_train)

print("Best: {} using {}".format(result.best_score_, result.best_params_))
# result: Best: 0.7585714285714286 using {'max_features': 3, 'n_estimators': 100}