#!/usr/bin/env python
# coding: utf-8

# # Day 4

# # 1. Data Exploration:
# 
#     a. Load and explore the medical dataset using Python libraries like pandas. Describe the features, label, and the distribution of diagnoses.

# In[10]:


import pandas as pd

# 1. Load the dataset
data = pd.read_csv("insurance.csv")

# 2. Explore the features and label
# Assuming "charges" is the label and the rest are features
features = data.drop(columns=["charges"])
label = data["charges"]

# 3. Describe the features
print("Features:")
print(features.head())  # Display the first few rows of the features

# 4. Describe the distribution of the label (charges)
print("\nLabel (Charges) Distribution:")
label_distribution = label.describe()
print(label_distribution)


# # 2. Data Preprocessing:
# 
#     a. Explain the necessary data preprocessing steps for preparing the medical data. This may include handling missing values, normalizing or scaling features, and encoding categorical variables.
# 
#     b. Calculate the prior probabilities P(Condition) and P(No Condition) based on the class distribution

# In[4]:


import pandas as pd

# Load the insurance dataset from Kaggle (replace 'insurance.csv' with the actual path)
data = pd.read_csv("insurance.csv")

# 2. Data Preprocessing

# a. Data Preprocessing Steps
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in the dataset:\n", missing_values)

# Handle missing values
data.fillna(0, inplace=True)  # You can replace '0' with an appropriate strategy based on your dataset

# Encode categorical variables (if any) using one-hot encoding
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Assume 'insuranceclaim' is your target variable
# Assuming '1' corresponds to 'Condition' and '0' corresponds to 'No Condition'
# Calculate the prior probabilities P(Condition) and P(No Condition)
diagnosis_distribution = data['charges'].value_counts()
p_condition = diagnosis_distribution[1] / len(data)
p_no_condition = diagnosis_distribution[0] / len(data)

# Print the calculated prior probabilities
print("Prior Probability of Condition (P(Condition)): ", p_condition)
print("Prior Probability of No Condition (P(No Condition)): ", p_no_condition)


# # 3. Feature Engineering:
# 
#     a. Describe how to convert the medical test results and patient information into suitable features for the Naive Bayes model.
# 
#     b. Discuss the importance of feature selection or dimensionality reduction in medical diagnosis

# In[4]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the insurance dataset from 'insurance.csv'
data = pd.read_csv("insurance.csv")

# Define a binary target variable based on a condition
data['target'] = (data['charges'] > data['charges'].mean()).astype(int)

# Encode categorical features using one-hot encoding
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'])

# Features
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target variable

# Train a Random Forest classifier to estimate feature importance
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Get feature importances
feature_importances = clf.feature_importances_

# List feature names and their importance scores
features = X.columns
feature_importance_dict = dict(zip(features, feature_importances))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Display the feature importance
print("Feature Importance:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")


# # 4. Implementing Naive Bayes:
# 
#     a. Choose the appropriate Naive Bayes variant (e.g., Gaussian, Multinomial, or Bernoulli Naive Bayes) for the medical diagnosis task and implement the classifier using Python libraries like scikit-learn.
# 
#     b. Split the dataset into training and testing sets.

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

# Load the insurance dataset from 'insurance.csv'
data = pd.read_csv("insurance.csv")

# Define a binary target variable based on a condition
data['target'] = (data['charges'] > data['charges'].mean()).astype(int)

# Encode categorical features using one-hot encoding
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'])

# Features and target variable
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Bernoulli Naive Bayes classifier
clf = BernoulliNB()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the accuracy and classification report
print("Accuracy:", accuracy)
print("Classification Report:\n", report)


# # 5. Model Training:
# 
#     a. Train the Naive Bayes model using the feature-engineered dataset. Explain the probability estimation process in Naive Bayes for medical diagnosis..

# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# Load the insurance dataset from 'insurance.csv'
data = pd.read_csv("insurance.csv")

# Define a binary target variable based on a condition
data['target'] = (data['charges'] > data['charges'].mean()).astype(int)

# Encode categorical features using one-hot encoding
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'])

# Features and target variable
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Bernoulli Naive Bayes classifier
clf = BernoulliNB()
clf.fit(X_train, y_train)

# Probability Estimation for Test Data
# Probabilities for each class (0: low premium, 1: high premium)
# For binary classification, you can use predict_proba to get probabilities
probabilities = clf.predict_proba(X_test)

# Example: Get the probability of the first data point being in the "high premium" class
first_data_point_probability = probabilities[0][1]
print("Probability of the first data point being in the 'high premium' class:", first_data_point_probability)


# # 6. Model Evaluation:
# 
#     a. Assess the performance of the medical diagnosis model using relevant evaluation metrics, such as accuracy, precision, recall, and F1-score..
# 
#     b. Interpret the results and discuss the model's ability to accurately classify medical conditions.
# 

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the insurance dataset from 'insurance.csv'
data = pd.read_csv("insurance.csv")

# Define a binary target variable based on a condition
data['target'] = (data['charges'] > data['charges'].mean()).astype(int)

# Encode categorical features using one-hot encoding
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'])

# Features and target variable
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Bernoulli Naive Bayes classifier
clf = BernoulliNB()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Classification Report:\n", report)


# # 7. Laplace Smoothing:
# 
#     a. Explain the concept of Laplace (add-one) smoothing and discuss its potential application in the context of medical diagnosis.
# 
#     b. Discuss the impact of Laplace smoothing on model performance.

# In[ ]:





# # 8. Real-World Application:
# 
#     a. Describe the importance of accurate medical diagnosis in healthcare and research.
# 
#     b. Discuss the practical implications of implementing a diagnostic system based on Naive Bayes.

# In[ ]:





# # 9. Model Limitations:
# 
#     a. Identify potential limitations of the Naive Bayes approach to medical diagnosis and discuss scenarios in which it may not perform well.

# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the insurance dataset from 'insurance.csv'
data = pd.read_csv("insurance.csv")

# Define a binary target variable based on a condition
data['target'] = (data['charges'] > data['charges'].mean()).astype(int)

# Encode categorical features using one-hot encoding
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'])

# Features and target variable
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Bernoulli Naive Bayes classifier
clf = BernoulliNB()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# # Qn 2

# # 1. Data Exploration:
# 
#     a. Load the customer dataset using Python libraries like pandas and explore its structure. Describe the features, target variable, and data distribution.
# 
#     b. Explain the importance of customer segmentation in the retail industry.

# In[3]:


import pandas as pd

# Load the dataset
customer_data = pd.read_csv("customer_dataset.csv")

# Display the structure of the dataset
customer_data.head()

# Describe the features
features = customer_data.drop("Payment Method", axis=1)
print("Features:")
print(features.head())

# Describe the target variable
target_variable = customer_data["Payment Method"]
print("\nTarget Variable:")
print(target_variable.head())

# Data distribution
print("\nData Distribution:")
print(customer_data.describe())


# # 2. Data Preprocessing:
# 
#     a. Prepare the customer data for analysis. Discuss the steps involved in data preprocessing, such as scaling handling missing values, and encoding categorical variables.

# In[7]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load the customer dataset (replace 'customer_data.csv' with your actual file path or URL)
data = pd.read_csv("customer_dataset.csv")

# Define your numerical and categorical features
numerical_features = ['Previous Purchases']  # Replace with your actual numerical feature column names
categorical_features = ['Review Rating', 'Item Purchased', 'Shipping Type']  # Replace with your actual categorical feature column names

# Check if the specified categorical features exist in the dataset
for col in categorical_features:
    if col not in data.columns:
        raise ValueError(f"Column '{col}' specified in categorical_features does not exist in the dataset.")

# Step 1: Handle Missing Values
# For numerical features, fill missing values with the mean
imputer_numeric = SimpleImputer(strategy='mean')
data[numerical_features] = imputer_numeric.fit_transform(data[numerical_features])

# For categorical features, fill missing values with the most frequent category
imputer_categorical = SimpleImputer(strategy='most_frequent')
data[categorical_features] = imputer_categorical.fit_transform(data[categorical_features])

# Step 2: Encode Categorical Variables
# We'll use one-hot encoding for categorical features to create binary columns for each category.
encoder = OneHotEncoder(sparse=False, drop='first')  # Set 'drop' to 'first' to avoid multicollinearity
encoded_features = encoder.fit_transform(data[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(input_features=categorical_features)
encoded_data = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Step 3: Combine Encoded Categorical Features with Numerical Features
X = pd.concat([data[numerical_features], encoded_data], axis=1)

# Step 4: Scale Numerical Features
# It's important to scale numerical features when using K-Nearest Neighbors (KNN) as it's distance-based.
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Now, your data is preprocessed and ready for analysis.


# # 3. implementing KNN:
# 
#     a. Implement the K-Nearest Neighbors algorithm using Python libraries like scikit-learn to segment customers based on their features.
# 
#     b. Choose an appropriate number of neighbors (K) for the algorithm and explain your choice.
# 

# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming 'segments' is the target variable
X = customer_data[['Purchase Amount (USD)', 'Age', 'Previous Purchases']]
y = customer_data['Subscription Status']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Implement KNN
k_value = 5  # You can adjust this based on the dataset and cross-validation results
knn_model = KNeighborsClassifier(n_neighbors=k_value)
knn_model.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# # 3. implementing KNN:
# 
#     a. Implement the K-Nearest Neighbors algorithm using Python libraries like scikit-learn to segment customers based on their features.
# 
#     b. Choose an appropriate number of neighbors (K) for the algorithm and explain your choice.
# 

# In[5]:


from sklearn.model_selection import cross_val_score

# Define a range of K values to test
k_values = list(range(1, 11))

# Evaluate each K value using cross-validation
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_scaled, y, cv=5)  # 5-fold cross-validation
    average_accuracy = scores.mean()
    print(f'K = {k}, Average Accuracy: {average_accuracy}')


# # 4. Model Training:
# 
#     a. Train the KNN model using the preprocessed customer dataset.
# 
#     b. Discuss the distance metric used for finding the nearest neighbors and its significance in customer segmentation.

# In[8]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming X_scaled is the preprocessed feature matrix, and y is the target variable
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Assuming k_value is the chosen number of neighbors
k_value = 5
knn_model = KNeighborsClassifier(n_neighbors=k_value)

# Train the KNN model
knn_model.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# # 5. Customer Segmentation:
# 
#     a. Segment the customers based on their purchase behavior, age, and income.
#     b. Visualize the customer segments to gain insights into the distribution and characteristics of each segment.

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your data (replace 'your_data.csv' with your actual data)
data = pd.read_csv("customer_dataset.csv")

# Select the features for segmentation (e.g., 'Review Rating', 'Age', 'Payment Method')
selected_features = data[['Review Rating', 'Age', 'Payment Method']]

# Define which columns are categorical and which are numerical
categorical_features = ["Payment Method"]
numerical_features = ["Review Rating", "Age"]

# Create transformers for preprocessing
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse=False))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine the transformers using a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply the preprocessing
preprocessed_features = preprocessor.fit_transform(selected_features)

# Determine the optimal number of clusters (K) using the Elbow Method
wcss = []  # Within-Cluster Sum of Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(preprocessed_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow Method, choose an appropriate K (number of clusters)
k = 3  # Adjust this value based on the Elbow Method plot

# Apply K-Means clustering with the chosen K
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
data['Cluster'] = kmeans.fit_predict(preprocessed_features)

# Visualize the customer segments
plt.figure(figsize=(10, 8))
for cluster in range(k):
    plt.scatter(data[data['Cluster'] == cluster]['Age'], data[data['Cluster'] == cluster]['Review Rating'], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=300, c='red', label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Age')
plt.ylabel('Review Rating')
plt.legend()
plt.show()


# # 6. Hyperparameter Tuning:
# 
#     a. Explain the role of the hyperparameter (K) in the KNN algorithm and suggest strategies for selecting the optimal value of K.
# 
#     b. Conduct hyperparameter tuning for the KNN model and discuss the impact of different values of K on segmentation results.

# In[12]:


from sklearn.model_selection import GridSearchCV

# Define a range of K values to test
param_grid = {'n_neighbors': list(range(1, 21))}

# Create KNN model
knn_model = KNeighborsClassifier()

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_scaled, y)

# Get the best hyperparameter values
best_k = grid_search.best_params_['n_neighbors']
best_accuracy = grid_search.best_score_

print(f'Best K: {best_k}, Best Accuracy: {best_accuracy}')


# # 7. Model Evaluation:
# 
#     a. Evaluate the KNN model's performance in customer segmentation. Discuss the criteria and metrics used for evaluating unsupervised learning models.
# 
#     b. Interpret the results and provide insights on how the customer segments can be leveraged for marketing strategies.

# In[14]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load your preprocessed data
data = pd.read_csv("customer_dataset.csv")

# Define your features for clustering (e.g., using numerical columns)
X = data[['Review Rating', 'Previous Purchases']]

# Choose the number of clusters (you can experiment with different values)
n_clusters = 5

# Fit K-Means clustering model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Calculate silhouette score to evaluate the model
silhouette_avg = silhouette_score(X, data['Cluster'])
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Analyze the clusters
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Review Rating', 'Previous Purchases'])
cluster_centers['Cluster'] = range(n_clusters)
print(cluster_centers)

# Interpretation and marketing insights can be provided here.


# # 8. Real-World Application:
# 
#     a. Describe the practical applications of customer segmentation in the retail industry.
# 
#     b. Discuss how customer segmentation can lead to improved customer engagement and increased sales.

# In[15]:


from sklearn.cluster import KMeans
import pandas as pd

# Assuming customer_data is the preprocessed DataFrame with features
X = customer_data[['Purchase Amount (USD)', 'Age', 'Previous Purchases']]

# Choose the number of clusters (segments)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
customer_data['Subscription Status'] = kmeans.fit_predict(X)

# Assume you have a product catalog DataFrame
product_catalog = pd.DataFrame({
    'product_id': range(1, 11),
    'product_name': [f'Product_{i}' for i in range(1, 11)]
})

# Define a function to recommend products for a given customer segment
def recommend_products(segment):
    segment_products = product_catalog.sample(3)  # Recommend 3 random products
    return segment_products

# Example: Recommend products for customers in segment 0
segment_0_recommendations = recommend_products(0)
print("Product Recommendations for Segment 0:")
print(segment_0_recommendations)


# # 9. Model Limitations:
# 
#     a. Identify potential limitations of the KNN algorithm in customer segmentation and discuss scenarios in which it may not perform well.

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

# Create a synthetic dataset with outliers
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X[-1] = [4, 2]  # Adding an outlier

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X, y)

# Plot decision boundary
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the dataset and decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
plt.title("KNN Decision Boundary with Outlier")
plt.show()

