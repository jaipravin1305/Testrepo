#!/usr/bin/env python
# coding: utf-8

# # 1. Reading and Understanding the Data

# In[2]:


import warnings
warnings.filterwarnings('ignore')

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cars = pd.read_csv('CarPrice_Assignment.csv')
cars.head()


# In[3]:


cars.shape


# In[4]:


cars.describe()


# In[5]:


cars.info()


# # 2. Data Cleaning and Preparation

# In[6]:


#Splitting company name from CarName column
CompanyName = cars['CarName'].apply(lambda x : x.split(' ')[0])
cars.insert(3,"CompanyName",CompanyName)
cars.drop(['CarName'],axis=1,inplace=True)
cars.head()


# In[7]:


cars.CompanyName.unique()


# In[8]:


cars.CompanyName = cars.CompanyName.str.lower()

def replace_name(a,b):
    cars.CompanyName.replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

cars.CompanyName.unique()


# In[9]:


#Checking for duplicates
cars.loc[cars.duplicated()]


# In[10]:


cars.columns


# # 3. Visualizing the data

# In[11]:


plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Car Price Distribution Plot')
sns.distplot(cars.price)

plt.subplot(1,2,2)
plt.title('Car Price Spread')
sns.boxplot(y=cars.price)

plt.show()


# In[12]:


print(cars.price.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))


# In[67]:


plt.figure(figsize=(25, 6))

plt.subplot(1, 3, 1)
plt1 = cars['CompanyName'].value_counts().plot(kind='bar')
plt.title('Companies Histogram')
plt1.set(xlabel='Car company', ylabel='Frequency of company')

plt.subplot(1, 3, 2)
plt2 = cars['fueltype'].value_counts().plot(kind='bar')
plt.title('Fuel Type Histogram')
plt2.set(xlabel='Fuel Type', ylabel='Frequency of fuel type')

plt.subplot(1, 3, 3)
plt3 = cars['carbody'].value_counts().plot(kind='bar')
plt.title('Car Type Histogram')
plt3.set(xlabel='Car Type', ylabel='Frequency of Car type')

plt.show()


# In[14]:


plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Symboling Histogram')
sns.countplot(cars.symboling, palette=("cubehelix"))

plt.subplot(1,2,2)
plt.title('Symboling vs Price')
sns.boxplot(x=cars.symboling, y=cars.price, palette=("cubehelix"))

plt.show()


# In[70]:


plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
plt.title('Engine Type Histogram')
sns.countplot(x='enginetype', data=cars, palette='Blues_d')

plt.subplot(1, 2, 2)
plt.title('Engine Type vs Price')
sns.boxplot(x='enginetype', y='price', data=cars, palette='PuBuGn')

plt.show()

df = pd.DataFrame(cars.groupby(['enginetype'])['price'].mean().sort_values(ascending=False))
df.plot.bar(figsize=(8, 6))
plt.title('Engine Type vs Average Price')
plt.show()


# In[16]:


plt.figure(figsize=(25, 6))

df = pd.DataFrame(cars.groupby(['CompanyName'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Company Name vs Average Price')
plt.show()

df = pd.DataFrame(cars.groupby(['fueltype'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Fuel Type vs Average Price')
plt.show()

df = pd.DataFrame(cars.groupby(['carbody'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Car Type vs Average Price')
plt.show()


# In[68]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title('Door Number Histogram')
sns.countplot(x='doornumber', data=cars, palette='plasma')

plt.subplot(1, 2, 2)
plt.title('Door Number vs Price')
sns.boxplot(x='doornumber', y='price', data=cars, palette='plasma')

plt.show()


# In[69]:


# Convert categorical variables to numerical using label encoding
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
categorical_cols = ['enginelocation', 'cylindernumber', 'fuelsystem', 'drivewheel']

for col in categorical_cols:
    cars[col] = encoder.fit_transform(cars[col])

def plot_count(x, fig):
    plt.subplot(4, 2, fig)
    plt.title(x + ' Histogram')
    sns.countplot(cars[x], palette="magma")
    plt.subplot(4, 2, fig + 1)
    plt.title(x + ' vs Price')
    sns.boxplot(x=cars[x], y=cars.price, palette="magma")

plt.figure(figsize=(15, 20))

plot_count('enginelocation', 1)
plot_count('cylindernumber', 3)
plot_count('fuelsystem', 5)
plot_count('drivewheel', 7)

plt.tight_layout()
plt.show()


# In[19]:


def scatter(x,fig):
    plt.subplot(5,2,fig)
    plt.scatter(cars[x],cars['price'])
    plt.title(x+' vs Price')
    plt.ylabel('Price')
    plt.xlabel(x)

plt.figure(figsize=(10,20))

scatter('carlength', 1)
scatter('carwidth', 2)
scatter('carheight', 3)
scatter('curbweight', 4)

plt.tight_layout()


# In[20]:


def pp(x,y,z):
    sns.pairplot(cars, x_vars=[x,y,z], y_vars='price',size=4, aspect=1, kind='scatter')
    plt.show()

pp('enginesize', 'boreratio', 'stroke')
pp('compressionratio', 'horsepower', 'peakrpm')
pp('wheelbase', 'citympg', 'highwaympg')


# In[21]:


np.corrcoef(cars['carlength'], cars['carwidth'])[0, 1]


# # 4. Deriving new features

# In[22]:


cars['fueleconomy'] = (0.55 * cars['citympg']) + (0.45 * cars['highwaympg'])


# In[23]:


cars['price'] = cars['price'].astype('int')
temp = cars.copy()
table = temp.groupby(['CompanyName'])['price'].mean()
temp = temp.merge(table.reset_index(), how='left',on='CompanyName')
bins = [0,10000,20000,40000]
cars_bin=['Budget','Medium','Highend']
cars['carsrange'] = pd.cut(temp['price_y'],bins,right=False,labels=cars_bin)
cars.head()


# # 5. Bivariate Analysis

# In[24]:


plt.figure(figsize=(8,6))

plt.title('Fuel economy vs Price')
sns.scatterplot(x=cars['fueleconomy'],y=cars['price'],hue=cars['drivewheel'])
plt.xlabel('Fuel Economy')
plt.ylabel('Price')

plt.show()
plt.tight_layout()


# In[25]:


plt.figure(figsize=(25, 6))

df = pd.DataFrame(cars.groupby(['fuelsystem','drivewheel','carsrange'])['price'].mean().unstack(fill_value=0))
df.plot.bar()
plt.title('Car Range vs Average Price')
plt.show()


# In[26]:


cars_lr = cars[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 
                    'fueleconomy', 'carlength','carwidth', 'carsrange']]
cars_lr.head()


# In[27]:


sns.pairplot(cars_lr)
plt.show()


# # 6. Dummy Variables

# In[72]:


# Defining the map function
def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df
# Applying the function to the cars_lr

cars_lr = dummies('fueltype',cars_lr)
cars_lr = dummies('aspiration',cars_lr)
cars_lr = dummies('carbody',cars_lr)
cars_lr = dummies('drivewheel',cars_lr)
cars_lr = dummies('enginetype',cars_lr)
cars_lr = dummies('cylindernumber',cars_lr)
cars_lr = dummies('carsrange',cars_lr)


# In[30]:


cars_lr.head()


# In[31]:


cars_lr.shape


# # 7. Train-Test Split and feature scaling

# In[32]:


from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(cars_lr, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[33]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[34]:


df_train.head()


# In[35]:


df_train.describe()


# In[36]:


plt.figure(figsize = (30, 25))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[37]:


#Dividing data into X and y variables
y_train = df_train.pop('price')
X_train = df_train


# # 8. Model Building

# In[62]:


#RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[63]:


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)


# In[ ]:


# def evaluate_model(predictions, model_name):
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    

    print(f"{model_name} - Mean Squared Error: {mse:.2f}")
    print(f"{model_name} - R-squared: {r2:.2f}")

evaluate_model(lr_predictions, "Linear Regression")


# In[ ]:


# Create and fit the Lasso model
lasso_model = Lasso(alpha=0.01)  # Define and configure your Lasso model
lasso_model.fit(X_train, y_train)

# Generate predictions using the Lasso model
lasso_predictions = lasso_model.predict(X_test)

# Evaluate the Lasso model
evaluate_model(lasso_predictions, "Lasso")


# In[ ]:


# Create and fit the Ridge model
ridge_model = Ridge(alpha=0.01)  # Define and configure your Ridge model
ridge_model.fit(X_train, y_train)

# Generate predictions using the Ridge model
ridge_predictions = ridge_model.predict(X_test)

# Evaluate the Ridge model
evaluate_model(ridge_predictions, "Ridge")


# # 9. Residual Analysis of Model

# In[60]:


import statsmodels.api as sm

# Extract a single feature as X_train_new
X_train_new = X_train["wheelbase"]

# Fit the OLS model
X_train_new = sm.add_constant(X_train_new)  # Add a constant (intercept) if needed
lm = sm.OLS(y_train, X_train_new).fit()

# Generate predictions using the OLS model
y_train_price = lm.predict(X_train_new)


# In[61]:


# Fit your regression model (for example, Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Generate predictions for the training data
y_train_price = model.predict(X_train)

# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins=20)
fig.suptitle('Error Terms', fontsize=20)  # Plot heading
plt.xlabel('Errors', fontsize=18)


# In[87]:


print(df_test.columns)


# # 10. Prediction and Evaluation

# In[13]:


import pandas as pd

# Load the DataFrame from a CSV file
df_test = pd.read_csv("CarPrice_Assignment.csv")

#df_test = pd.DataFrame(data)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','carlength','carwidth','price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])


# In[14]:


#Dividing into X and y
y_test = df_test.pop('price')
X_test = df_test


# In[ ]:


# import pandas as pd

# Assuming you have X_train and X_test defined
# Assuming you have your training and testing data in X_train and X_test
# Make sure to load or define these DataFrames with your actual data
# Example:
X_train = pd.read_csv("CarPrice_Assignment.csv")
X_test = pd.read_csv("CarPrice_Assignment.csv")

# Create X_train_new and X_test_new
X_train_new = X_train.copy()  # Create a copy of X_train
X_test_new = X_test.copy()    # Create a copy of X_test
# Check if 'const' exists in X_train_new before dropping
if 'const' in X_train_new.columns:
    X_train_new = X_train_new.drop('const', axis=1)
else:
    print("'const' column not found in X_train_new")

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test_new[X_train_new.columns]

# Drop 'const' from X_train_new (if 'const' is a column you want to remove)
#X_train_new = X_train_new.drop('const', axis=1)

# Creating X_test_new dataframe by dropping variables from X_test
#X_test_new = X_test_new[X_train_new.columns]

# Adding a constant variable to X_test_new
X_test_new = sm.add_constant(X_test_new)

# Now you can use X_train_new and X_test_new as intended


# In[ ]:


# import pandas as pd
from sklearn.linear_model import LinearRegression
# Load your training data into df_train from a CSV file (replace 'your_training_data.csv' with your file path)
df_train = pd.read_csv('CarPrice_Assignment.csv')

# Define your target variable y_train
y_train = df_train['price']  # Assuming 'price' is the name of your target variable

# Continue with the rest of your code, including defining and training the linear regression model



# Define your target variable y_train
y_train = df_train['price']  # Assuming 'price' is the name of your target variable

# Define and train the linear regression model
lm = LinearRegression()
lm.fit(X_train_new, y_train)

# Now you can make predictions using the trained model
y_pred = lm.predict(X_test_new)


# In[ ]:


# from sklearn.metrics import r2_score 
r2_score(y_test,y_pred)


# In[ ]:


# EVALUATION OF THE MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)


# In[ ]:




