#!/usr/bin/env python
# coding: utf-8

# # Day 5

# # 1. Data Exploration:
# 	a. Load the dataset using Python libraries like pandas and explore its structure. Describe the features, 
# target variables, and data distribution.
# 	b. Discuss the importance of customer satisfaction and sales prediction in the retail business context.

# In[2]:


import pandas as pd

# Load the dataset
df = pd.read_csv('train.csv')

# Explore the dataset structure
print("Dataset Info:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst 5 Rows:")
print(df.head())

# Descriptive statistics of the dataset
print("\nDescriptive Statistics:")
print(df.describe())

# Data distribution of categorical features
print("\nValue Counts for Categorical Features:")
categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for feature in categorical_features:
    print(f"\n{feature}:\n{df[feature].value_counts()}")

# Data distribution of numerical features
# You can use histograms or box plots to visualize numerical data distribution
import matplotlib.pyplot as plt

numerical_features = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    df[feature].plot(kind='hist', bins=20, title=feature)
    plt.xlabel(feature)
    plt.show()


# # 2. Classification Task - Predicting Customer Satisfaction:
#     a. Implement a decision tree classifier using Python libraries like scikit-learn to predict customer 
# satisfaction.
#     b. Split the dataset into training and testing sets and train the model.
#     c. Evaluate the classification model's performance using relevant metrics such as accuracy, precision, 
# recall, and F1-score.

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('train.csv')

# Select features and target variable
features = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
            'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
            'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
            'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
target = 'satisfaction'

X = df[features]
y = df[target]

# Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
classifier = DecisionTreeClassifier()

# Train the model on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the classification model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display the classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=df[target].unique()))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# # 3. Regression Task - Predicting Sales:
#    a. Implement a decision tree regression model using Python libraries to predict sales based on customer 
# attributes and behavior.
# 	b. Discuss the differences between classification and regression tasks in predictive modeling.
# 	c. Split the dataset into training and testing sets and train the regression model.
# 	d. Evaluate the regression model's performance using metrics such as mean squared error (MSE) and R-squared.

# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('train.csv')

# Select features and target variable
features = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
            'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
            'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
            'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
target = 'satisfaction'

X = df[features]
y = df['Age']  # Assuming 'Sales' is the target variable for the regression task

# Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Regressor
regressor = DecisionTreeRegressor()

# Train the model on the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Evaluate the regression model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")


# # 4. Decision Tree Visualization:
# 	a. Visualize the decision tree for both the classification and regression models. Discuss the interpretability 
# of decision trees in predictive modeling.

# In[ ]:


# import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn import tree

# Assuming you already have a trained classification decision tree model named 'classifier'
plt.figure(figsize=(20, 10))
plot_tree(classifier, filled=True, feature_names=features, class_names=df['Class'].unique(), rounded=True)
plt.show()


# In[8]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Assuming you already have a trained regression decision tree model named 'regressor'
plt.figure(figsize=(20, 10))
plot_tree(regressor, filled=True, feature_names=features, rounded=True)
plt.show()


# # 5. Feature Importance:
#    a. Determine the most important features in both models by examining the decision tree structure. 
# Discuss how feature importance is calculated in decision trees.

# In[12]:


# Get feature importances from the classification decision tree
feature_importances = classifier.feature_importances_

# Create a DataFrame to associate feature names with their importance scores
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the top N most important features
top_n_features = 10  # You can change this value
print(f"Top {top_n_features} Most Important Features for Classification:")
print(feature_importance_df.head(top_n_features))


# In[13]:


# Get feature importances from the regression decision tree
feature_importances = regressor.feature_importances_

# Create a DataFrame to associate feature names with their importance scores
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the top N most important features
top_n_features = 10  # You can change this value
print(f"Top {top_n_features} Most Important Features for Regression:")
print(feature_importance_df.head(top_n_features))


# # 6. Overfitting and Pruning:
# a. Explain the concept of overfitting in the context of decision trees.
# b. Discuss methods for reducing overfitting, such as pruning, minimum samples per leaf, and maximum 
# depth.
# c. Implement pruning or other techniques as necessary and analyze their impact on the model's 
# performance.

# In[ ]:


# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd

# Load the dataset
df = pd.read_csv('train.csv')

# Continue with the rest of your code

# Create a Decision Tree Classifier with ccp_alpha pruning
classifier = DecisionTreeClassifier(ccp_alpha=0.01)  # Adjust the ccp_alpha value as needed

# Train the model
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Classification Report:")
# Check the unique classes in the 'satisfaction' column
unique_classes = df['satisfaction'].unique()
print(classification_report(y_test, y_pred, target_names=unique_classes))

#print(classification_report(y_test, y_pred, target_names=df[target].unique()))
#print(classification_report(y_test, y_pred, target_names=df['satisfaction'].unique()))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# # 7. Real-World Application:
# 	a. Describe the practical applications of customer satisfaction prediction and sales forecasting in the retail 
# industry.
# 	b. Discuss the potential benefits of using predictive models in retail business operations and decisionmaking.

# In[10]:


from sklearn.impute import SimpleImputer

# Instantiate the imputer
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on your training data and transform both training and testing data
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Then, proceed with fitting the RandomForestClassifier as you did before
data.dropna(inplace=True)

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Instantiate the imputer
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on your entire dataset and transform it
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classification Accuracy: {accuracy}")


# # 8. Model Comparison:
# 	a. Compare the performance of the decision tree classification and regression models.
# 	b. Discuss the trade-offs, advantages, and limitations of decision trees for different types of predictive 
# tasks.

# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('train.csv')

# Perform one-hot encoding for categorical variables
data = pd.get_dummies(data, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'])

# Split the data into features and target variable
X = data.drop('satisfaction', axis=1)
y = data['satisfaction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Classification Accuracy: {accuracy}")


# In[ ]:




