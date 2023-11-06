#!/usr/bin/env python
# coding: utf-8

# # Day 1

# # 1. Data Preparation:
# 
# a. Load the dataset using pandas.
# 
# b. Explore and clean the data. Handle missing values and outliers.
# 
# c. Split the dataset into training and testing sets.

# In[6]:


import pandas as pd
import numpy as np

# Load the dataset from Housing.csv
data = pd.read_csv('Housing.csv')

# Display the first few rows of the dataset to get an overview
print(data.head())


# In[4]:


data.dropna(inplace=True)


# In[8]:


from scipy import stats

z_scores = stats.zscore(data[['price', 'area', 'bedrooms', 'bathrooms']])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data = data[filtered_entries]


# In[ ]:


# from sklearn.model_selection import train_test_split

# Define your feature columns and target variable
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishing']]
y = data['price']

# Split the dataset into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Now, you have X_train, X_test, y_train, and y_test for your machine learning model.


# # 2. Implement Simple Linear Regression:
# 
# a. Choose a feature (e.g., square footage) as the independent variable (X) and house prices as the dependent variable (y).
# b. Implement a simple linear regression model using sklearn to predict house prices based on the selected feature.

# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset from Housing.csv
data = pd.read_csv('Housing.csv')

# Define the feature (independent variable) and target (dependent variable)
X = data[['area']]  # Independent variable
y = data['price']  # Dependent variable

# Split the dataset into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Visualize the data and regression line
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.show()

# Coefficients and intercept of the linear regression model
slope = model.coef_[0]
intercept = model.intercept_

print("Linear Regression Equation: Price = {} * Area + {}".format(slope, intercept))


# # 3. Evaluate the Simple Linear Regression Model:
# 
# a. Use scikit-learn to calculate the R-squared value to assess the goodness of fit.
# 
# b. Interpret the R-squared value and discuss the model's performance.

# In[14]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset from Housing.csv
data = pd.read_csv('Housing.csv')

# Define the feature (independent variable) and target (dependent variable)
X = data[['area']]  # Independent variable
y = data['price']  # Dependent variable

# Split the dataset into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the R-squared value
r_squared = r2_score(y_test, y_pred)

print("R-squared value:", r_squared)

# Interpret the R-squared value and discuss the model's performance
if r_squared == 1:
    print("The model perfectly fits the data.")
elif r_squared > 0.7:
    print("The model has a strong fit.")
elif r_squared > 0.5:
    print("The model has a moderate fit.")
else:
    print("The model has a weak fit. It may not be a good predictor of house prices.")


# # 4. Implement Multiple Linear Regression:
# 
# a. Select multiple features (e.g., square footage, number of bedrooms, number of bathrooms) as independent variables (X) and house prices as the dependent variable (y).
# 
# b. Implement a multiple linear regression model using scikit-learn to predict house prices based on the selected features.
# 

# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset from Housing.csv
data = pd.read_csv('Housing.csv')

# Define the features (independent variables) and target (dependent variable)
X = data[['area', 'bedrooms', 'bathrooms']]  # Independent variables
y = data['price']  # Dependent variable

# Split the dataset into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Multiple Linear Regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) as a measure of model performance
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)

# Coefficients and intercept of the multiple linear regression model
coefficients = model.coef_
intercept = model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)


# # 5. Evaluate the Multiple Linear Regression Model:
# 
# a. Calculate the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to assess the model's accuracy.
# 
# b. Discuss the advantages of using multiple features in regression analysis.

# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the dataset from Housing.csv
data = pd.read_csv('Housing.csv')

# Define the features (independent variables) and target (dependent variable)
X = data[['area', 'bedrooms', 'bathrooms']]  # Independent variables
y = data['price']  # Dependent variable

# Split the dataset into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Multiple Linear Regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics: MAE, MSE, RMSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Advantages of using multiple features in regression analysis
print("Advantages of using multiple features:")
print("1. Improved model accuracy: Multiple features can capture complex relationships in the data that a single feature may miss.")
print("2. Better predictive power: More features can provide a more comprehensive view of the factors influencing the target variable.")
print("3. Reduced bias: Multiple features help reduce bias in the model, leading to more accurate predictions.")
print("4. Enhanced model interpretability: Including domain-specific features can lead to more interpretable models.")


# # 6. Model Comparison:
# 
# 	a. Compare the results of the simple linear regression and multiple linear regression models.
# 
# 	b. Discuss the advantages and limitations of each model.

# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset from Housing.csv
data = pd.read_csv('Housing.csv')

# Simple Linear Regression
X_simple = data[['area']]
y_simple = data['price']
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train_simple)
y_pred_simple = simple_model.predict(X_test_simple)

# Multiple Linear Regression
X_multi = data[['area', 'bedrooms', 'bathrooms']]
y_multi = data['price']
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

multi_model = LinearRegression()
multi_model.fit(X_train_multi, y_train_multi)
y_pred_multi = multi_model.predict(X_test_multi)

# Evaluate Simple Linear Regression
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)

# Evaluate Multiple Linear Regression
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)

# Display the evaluation results
print("Simple Linear Regression Results:")
print("Mean Squared Error (MSE):", mse_simple)
print("R-squared (R2):", r2_simple)

print("\nMultiple Linear Regression Results:")
print("Mean Squared Error (MSE):", mse_multi)
print("R-squared (R2):", r2_multi)

# Discuss the advantages and limitations of each model
print("\nAdvantages of Simple Linear Regression:")
print("- Simplicity: Easy to understand and implement.")
print("- High interpretability: Clear understanding of the relationship between a single predictor and the target variable.")
print("- Lower risk of overfitting: Simpler models are less prone to overfitting.")

print("\nAdvantages of Multiple Linear Regression:")
print("- Improved accuracy: Can capture complex relationships involving multiple predictors.")
print("- Better predictive power: Utilizes multiple variables for a more comprehensive view.")
print("- Enhanced interpretability: Can accommodate domain-specific factors and interactions.")

print("\nLimitations of Simple Linear Regression:")
print("- Limited explanatory power: Not suitable for complex relationships.")
print("- Restricted use: Only applicable when there is a clear, strong predictor.")

print("\nLimitations of Multiple Linear Regression:")
print("- Complexity: More challenging to understand and explain due to multiple predictors.")
print("- Overfitting risk: Prone to overfitting when dealing with many predictors without proper feature selection.")


# In[ ]:




