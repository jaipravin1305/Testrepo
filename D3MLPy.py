#!/usr/bin/env python
# coding: utf-8

# # Day 3

# # 1. Data Preparation:
# 
#     a. Load the dataset, and provide an overview of the available features, including transaction details, customer information, and labels (fraudulent or non-fraudulent).
# 
#     b. Describe the class distribution of fraudulent and non-fraudulent transactions and discuss the imbalance issue.
# 

# In[2]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Financefraud.csv')

# Question 1a: Overview of available features
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Get information about the dataset, including data types and missing values
print("\nDataset information:")
print(data.info())

# Summary statistics for numerical features
print("\nSummary statistics for numerical features:")
print(data.describe())

# Question 1b: Class distribution and imbalance issue
# Count the number of fraudulent and non-fraudulent transactions
class_distribution = data['isFraud'].value_counts()
print("\nClass distribution (fraudulent vs. non-fraudulent):")
print(class_distribution)

# Calculate the percentage of fraudulent transactions
percentage_fraudulent = (class_distribution[1] / data.shape[0]) * 100
print(f"Percentage of fraudulent transactions: {percentage_fraudulent:.2f}%")

# Plot the class distribution
plt.figure(figsize=(6, 6))
plt.bar(class_distribution.index, class_distribution.values, color=['green', 'red'])
plt.xticks(class_distribution.index, ['Non-Fraudulent', 'Fraudulent'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution (Fraudulent vs. Non-Fraudulent)')
plt.show()

# Discussion of imbalance issue
if percentage_fraudulent < 1:
    imbalance_issue = "Severe Imbalance"
elif percentage_fraudulent < 10:
    imbalance_issue = "Moderate Imbalance"
else:
    imbalance_issue = "Mild Imbalance"

print(f"\nImbalance Issue: {imbalance_issue}")


# # 2. Initial Logistic Regression Model:
# 
#     a. Implement a basic logistic regression model using the raw dataset.
# 
#     b. Evaluate the model's performance using standard metrics like accuracy, precision, recall, and F1-score.

# In[3]:


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Define the features and target variable
X = data[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
y = data['isFraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
logistic_model = LogisticRegression()

# Fit the model to the training data
logistic_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logistic_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print("Model Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Generate and display the confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)


# # 3. Feature Engineering:
# 
#     a. Apply feature engineering techniques to enhance the predictive power of the model. These techniques may include:
# 
#         -Creating new features.
# 
#         - Scaling or normalizing features.
# 
#         - Handling missing values.
# 
#         - Encoding categorical variables.
# 
#     b. Explain why each feature engineering technique is relevant for fraud detection.
# 

# In[ ]:


import pandas as pd

# Load the dataset (assuming the file 'Financefraud.csv' is in the current directory)
data = pd.read_csv('Financefraud.csv')

# Feature engineering
data['transaction_velocity'] = (data['newbalanceOrig'] - data['oldbalanceOrg']) / (data['amount'] + 1)
data['transaction_amount_category'] = pd.cut(data['amount'], bins=[0, 1000, 10000, 1000000], labels=['small', 'medium', 'large'])

# Standardize numerical features
numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'transaction_velocity']
data[numerical_features] = (data[numerical_features] - data[numerical_features].mean()) / data[numerical_features].std()

# Impute missing values
data.fillna(data.mean(), inplace=True)

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['transaction_amount_category', 'type'], drop_first=True)


# # 4. Handling Imbalanced Data:
# 
#     a. Discuss the challenges associated with imbalanced datasets in the context of fraud detection.
# 
#     b. Implement strategies to address class imbalance, such as:
# 
#         -Oversampling the minority class.
# 
#         -Undersampling the majority class.
# 
#         -Using synthetic data generation techniques (e.g., SMOTE).
# 

# In[ ]:


# from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

# Implement oversampling of the minority class
oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
X_over, y_over = oversampler.fit_resample(X, y)

# Implement undersampling of the majority class
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_under, y_under = undersampler.fit_resample(X, y)

# Implement Synthetic Minority Over-sampling Technique (SMOTE)
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Check the class distribution after each method
print("Class distribution after oversampling:")
print(np.bincount(y_over))

print("Class distribution after undersampling:")
print(np.bincount(y_under))

print("Class distribution after SMOTE:")
print(np.bincount(y_smote))


# # 5. Logistic Regression with Feature-Engineered Data:
# 
#     a. Train a logistic regression model using the feature-engineered dataset and the methods for handling imbalanced data.
# 
#     b. Evaluate the model's performance using appropriate evaluation metrics:

# In[ ]:


# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler

# Define the features and target variable
X = data.drop(columns=['isFraud'])
y = data['isFraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement oversampling of the minority class
oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
X_train_over, y_train_over = oversampler.fit_resample(X_train, y_train)

# Create a logistic regression model
logistic_model = LogisticRegression()

# Fit the model to the training data
logistic_model.fit(X_train_over, y_train_over)

# Make predictions on the test data
y_pred = logistic_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print("Model Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Generate and display the confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)


# In[ ]:


# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler

# Define the features and target variable
X = data.drop(columns=['isFraud'])
y = data['isFraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement oversampling of the minority class
oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
X_train_over, y_train_over = oversampler.fit_resample(X_train, y_train)

# Create a logistic regression model
logistic_model = LogisticRegression()

# Fit the model to the training data
logistic_model.fit(X_train_over, y_train_over)

# Interpret the coefficients
# Get the coefficients and corresponding feature names
coefficients = logistic_model.coef_[0]
feature_names = X_train_over.columns

# Create a DataFrame to store the coefficients and feature names
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the DataFrame by coefficient magnitude (absolute value)
coefficients_df['Abs_Coefficient'] = coefficients_df['Coefficient'].abs()
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False)

# Display the top N most influential features
top_n_features = 10  # Adjust this value to show the top N features
print(f"Top {top_n_features} most influential features:")
print(coefficients_df.head(top_n_features))

# Make predictions on the test data
y_pred = logistic_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print("Model Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Generate and display the confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)


# In[ ]:


# import sklearn
import imblearn

print(f"scikit-learn version: {sklearn.__version__}")
print(f"imbalanced-learn version: {imblearn.__version__}")


# In[ ]:




