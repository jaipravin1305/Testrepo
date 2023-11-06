#!/usr/bin/env python
# coding: utf-8

# # Pandas:
# 1. Make a bar plot of the months in which movies with "Christmas" in their title tend to be released in the USA.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title contains "Christmas" and the country is USA
christmas_movies_usa = df[(df['title'].str.contains('Christmas', case=False, na=False)) & (df['country'] == 'USA')]

# Extract the release month from the 'year' column and create a new column 'release_month'
christmas_movies_usa['release_month'] = pd.to_datetime(christmas_movies_usa['year'], format='%Y').dt.month

# Count the number of movies released in each month
monthly_counts = christmas_movies_usa['release_month'].value_counts().sort_index()

# Create a bar plot of the months
plt.figure(figsize=(10, 6))
monthly_counts.plot(kind='bar', color='skyblue')
plt.title('Movies with "Christmas" in Title Released in the USA by Month')
plt.xlabel('Month')
plt.ylabel('Number of Movies')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# # Pandas:
# 2. Make a bar plot of the months in which movies whose titles start with "The Hobbit" are realeased in the USA.

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title starts with "The Hobbit" and the country is USA
hobbit_movies_usa = df[(df['title'].str.startswith('The Hobbit', na=False)) & (df['country'] == 'USA')]

# Extract the release month from the 'year' column and create a new column 'release_month'
hobbit_movies_usa['release_month'] = pd.to_datetime(hobbit_movies_usa['year'], format='%Y').dt.month

# Count the number of movies released in each month
monthly_counts = hobbit_movies_usa['release_month'].value_counts().sort_index()

# Create a bar plot of the months
plt.figure(figsize=(10, 6))
monthly_counts.plot(kind='bar', color='skyblue')
plt.title('Movies Starting with "The Hobbit" Released in the USA by Month')
plt.xlabel('Month')
plt.ylabel('Number of Movies')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# # Pandas:
# 3. Make a bar plot of the day of the week  which movies whose titles start with "Romance" are realeased in the USA.

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title starts with "Romance" and the country is USA
romance_movies_usa = df[(df['title'].str.startswith('Romance', na=False)) & (df['country'] == 'USA')]

# Convert the 'date' column to datetime to extract the day of the week
romance_movies_usa['date'] = pd.to_datetime(romance_movies_usa['date'])

# Extract the day of the week and create a new column 'day_of_week'
romance_movies_usa['day_of_week'] = romance_movies_usa['date'].dt.day_name()

# Count the number of movies released on each day of the week
day_of_week_counts = romance_movies_usa['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Create a bar plot of the day of the week
plt.figure(figsize=(10, 6))
day_of_week_counts.plot(kind='bar', color='skyblue')
plt.title('Movies Starting with "Romance" Released in the USA by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.show()


# # Pandas:
# 4. Make a bar plot of the day of the week  which movies with "Action" in their title tend to be  realeased in the USA.

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title contains "Action" (case-insensitive) and the country is USA
action_movies_usa = df[(df['title'].str.contains('Action', case=False, na=False)) & (df['country'] == 'USA')]

# Convert the 'date' column to datetime to extract the day of the week
action_movies_usa['date'] = pd.to_datetime(action_movies_usa['date'])

# Extract the day of the week and create a new column 'day_of_week'
action_movies_usa['day_of_week'] = action_movies_usa['date'].dt.day_name()

# Count the number of movies released on each day of the week
day_of_week_counts = action_movies_usa['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Create a bar plot of the day of the week
plt.figure(figsize=(10, 6))
day_of_week_counts.plot(kind='bar', color='skyblue')
plt.title('Movies with "Action" in Title Released in the USA by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.show()


# # Pandas:
# 5. On which date was each Judi Dench movie from the 1990s realeased in the USA?

# In[5]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title contains "Judi Dench" (case-insensitive) and the year is in the 1990s
judi_dench_movies_1990s = df[(df['title'].str.contains('Judi Dench', case=False, na=False)) & (df['year'] >= 1990) & (df['year'] <= 1999)]

# Filter further to select only movies released in the USA
judi_dench_movies_usa = judi_dench_movies_1990s[judi_dench_movies_1990s['country'] == 'USA']

# Display the release date of each movie
print(judi_dench_movies_usa[['title', 'date']])


# # Pandas:
# 6. In which month do films with Judi Dench tend to be realeased in the USA?

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title contains "Judi Dench" (case-insensitive) and the country is USA
judi_dench_movies_usa = df[(df['title'].str.contains('Judi Dench', case=False, na=False)) & (df['country'] == 'USA')]

# Check if there are any matching movies before proceeding
if not judi_dench_movies_usa.empty:
    # Convert the 'date' column to datetime to extract the release month
    judi_dench_movies_usa['date'] = pd.to_datetime(judi_dench_movies_usa['date'])

    # Extract the release month and create a new column 'release_month'
    judi_dench_movies_usa['release_month'] = judi_dench_movies_usa['date'].dt.month

    # Count the number of movies released in each month
    monthly_counts = judi_dench_movies_usa['release_month'].value_counts().sort_index()

    # Create a bar plot of the release months
    plt.figure(figsize=(10, 6))
    monthly_counts.plot(kind='bar', color='skyblue')
    plt.title('Release Months of Films with Judi Dench in the USA')
    plt.xlabel('Month')
    plt.ylabel('Number of Films')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.show()
else:
    print("No matching movies found.")


# # Pandas:
# 7. In which month do films with Tom Cruise tend to be realeased in the USA?

# In[ ]:


# import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('release_dates.csv')

# Filter rows where the title contains "Tom Cruise" (case-insensitive) and the country is USA
tom_cruise_movies_usa = df[(df['title'].str.contains('Tom Cruise', case=False, na=False)) & (df['country'] == 'USA')]

# Convert the 'date' column to datetime to extract the release month
tom_cruise_movies_usa['date'] = pd.to_datetime(tom_cruise_movies_usa['date'])

# Extract the release month and create a new column 'release_month'
tom_cruise_movies_usa['release_month'] = tom_cruise_movies_usa['date'].dt.month

# Count the number of movies released in each month
monthly_counts = tom_cruise_movies_usa['release_month'].value_counts().sort_index()

# Create a bar plot of the release months
plt.figure(figsize=(10, 6))
monthly_counts.plot(kind='bar', color='skyblue')
plt.title('Release Months of Films with Tom Cruise in the USA')
plt.xlabel('Month')
plt.ylabel('Number of Films')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# In[ ]:





# In[ ]:




