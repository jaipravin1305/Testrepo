#!/usr/bin/env python
# coding: utf-8

# # Pandas:
# 1. Define a year as a "Superman year" whose films feature more Superman characters than Batman. How many years in film history have been Superman years?

# In[1]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with "Superman" or "Batman" characters
superman_batman_roles = df[df['character'].isin(['Superman', 'Batman'])]

# Group the data by 'year' and 'character' and count the number of each character in each year
character_counts = superman_batman_roles.groupby(['year', 'character']).size().unstack(fill_value=0)

# Determine the years where the count of "Superman" characters is greater than "Batman" characters
superman_years = character_counts[character_counts['Superman'] > character_counts['Batman']]

# Count the number of "Superman years"
num_superman_years = len(superman_years)

# Print the result
print("Number of 'Superman years' in film history:", num_superman_years)


# # Pandas:
# 2. How many years have been "Batman years, with more Batman. characters than Superman characters?

# In[2]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with "Superman" or "Batman" characters
superman_batman_roles = df[df['character'].isin(['Superman', 'Batman'])]

# Group the data by 'year' and 'character' and count the number of each character in each year
character_counts = superman_batman_roles.groupby(['year', 'character']).size().unstack(fill_value=0)

# Determine the years where the count of "Batman" characters is greater than "Superman" characters
batman_years = character_counts[character_counts['Batman'] > character_counts['Superman']]

# Count the number of "Batman years"
num_batman_years = len(batman_years)

# Print the result
print("Number of 'Batman years' in film history with more Batman characters than Superman characters:", num_batman_years)


# # Pandas:
# 3. Plot the number of actor roles each year and the number of actress roles each year over the history of film.

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with 'type' as "actor" or "actress"
actor_actress_roles = df[df['type'].isin(['actor', 'actress'])]

# Group the data by 'year' and 'type' and count the number of roles of each type in each year
roles_by_year_and_type = actor_actress_roles.groupby(['year', 'type']).size().unstack(fill_value=0)

# Create separate DataFrames for actor and actress roles
actor_roles = roles_by_year_and_type['actor']
actress_roles = roles_by_year_and_type['actress']

# Plot the number of actor and actress roles each year
plt.figure(figsize=(12, 6))
plt.plot(actor_roles.index, actor_roles.values, label='Actor Roles', color='blue')
plt.plot(actress_roles.index, actress_roles.values, label='Actress Roles', color='pink')
plt.xlabel('Year')
plt.ylabel('Number of Roles')
plt.title('Number of Actor and Actress Roles Each Year')
plt.legend()
plt.grid()
plt.show()


# # Pandas:
# 4. Plot the number of actor roles each year and the number of actress roles each year, but this time as a kind='area' plot.

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with 'type' as "actor" or "actress"
actor_actress_roles = df[df['type'].isin(['actor', 'actress'])]

# Group the data by 'year' and 'type' and count the number of roles of each type in each year
roles_by_year_and_type = actor_actress_roles.groupby(['year', 'type']).size().unstack(fill_value=0)

# Create separate DataFrames for actor and actress roles
actor_roles = roles_by_year_and_type['actor']
actress_roles = roles_by_year_and_type['actress']

# Plot the number of actor and actress roles each year as an area plot
plt.figure(figsize=(12, 6))
plt.fill_between(actor_roles.index, actor_roles.values, label='Actor Roles', color='blue', alpha=0.5)
plt.fill_between(actress_roles.index, actress_roles.values, label='Actress Roles', color='pink', alpha=0.5)
plt.xlabel('Year')
plt.ylabel('Number of Roles')
plt.title('Number of Actor and Actress Roles Each Year (Area Plot)')
plt.legend()
plt.grid()
plt.show()


# # Pandas:
# 5. Plot the difference between the number of actor roles each year and the number of actress roles each year over the history of film.

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with 'type' as "actor" or "actress"
actor_actress_roles = df[df['type'].isin(['actor', 'actress'])]

# Group the data by 'year' and 'type' and count the number of roles of each type in each year
roles_by_year_and_type = actor_actress_roles.groupby(['year', 'type']).size().unstack(fill_value=0)

# Calculate the difference between the number of actor roles and actress roles each year
difference_roles = roles_by_year_and_type['actor'] - roles_by_year_and_type['actress']

# Plot the difference between actor and actress roles each year
plt.figure(figsize=(12, 6))
plt.plot(difference_roles.index, difference_roles.values, label='Difference (Actor - Actress)', color='green')
plt.xlabel('Year')
plt.ylabel('Difference in Number of Roles')
plt.title('Difference Between Actor and Actress Roles Each Year')
plt.legend()
plt.grid()
plt.show()


# # Pandas:
# 6. Plot the fraction of roles that have been 'actor' roles each year in the history of film.

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with 'type' as "actor" or "actress"
actor_actress_roles = df[df['type'].isin(['actor', 'actress'])]

# Group the data by 'year' and 'type' and count the number of roles of each type in each year
roles_by_year_and_type = actor_actress_roles.groupby(['year', 'type']).size().unstack(fill_value=0)

# Calculate the fraction of 'actor' roles each year
total_roles_each_year = roles_by_year_and_type['actor'] + roles_by_year_and_type['actress']
fraction_actor_roles = roles_by_year_and_type['actor'] / total_roles_each_year

# Plot the fraction of 'actor' roles each year
plt.figure(figsize=(12, 6))
plt.plot(fraction_actor_roles.index, fraction_actor_roles.values, label='Fraction of Actor Roles', color='blue')
plt.xlabel('Year')
plt.ylabel('Fraction of Actor Roles')
plt.title('Fraction of Actor Roles Each Year')
plt.legend()
plt.grid()
plt.show()


# # Pandas:
# 7. Plot the fraction of supporting (n=2) roles that have been 'actor' roles each year in the history of film.

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with 'type' as "actor," 'n' as 2 (supporting roles), and valid 'year'
supporting_actor_roles = df[(df['type'] == 'actor') & (df['n'] == 2) & ~df['year'].isna()]

# Group the data by 'year' and 'type' and count the number of supporting roles of each type in each year
supporting_roles_by_year_and_type = supporting_actor_roles.groupby(['year', 'type']).size().unstack(fill_value=0)

# Calculate the fraction of 'actor' supporting roles each year
total_supporting_roles_each_year = (
    supporting_roles_by_year_and_type['actor'] + supporting_roles_by_year_and_type['actor']
)
fraction_actor_supporting_roles = supporting_roles_by_year_and_type['actor'] / total_supporting_roles_each_year

# Plot the fraction of 'actor' supporting roles each year
plt.figure(figsize=(12, 6))
plt.plot(fraction_actor_supporting_roles.index, fraction_actor_supporting_roles.values, label='Fraction of Actor Supporting Roles', color='blue')
plt.xlabel('Year')
plt.ylabel('Fraction of Actor Supporting Roles')
plt.title('Fraction of Actor Supporting Roles Each Year')
plt.legend()
plt.grid()
plt.show()


# # Pandas:
# 8. Build a plot with a line for each rank n=1 through n=3, where the line shows what fraction of that rank's roles were 'actor' roles for each year in the history of film.

# In[ ]:


# import pandas as pd
import matplotlib.pyplot as plt

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows with 'type' as "actor," valid 'year,' and 'n' in the range of 1 to 3
actor_roles_by_rank = df[(df['type'] == 'actor') & ~df['year'].isna() & (df['n'] >= 1) & (df['n'] <= 3)]

# Group the data by 'year,' 'n,' and 'type,' and count the number of roles of each type for each rank (n=1, n=2, n=3) in each year
roles_by_year_rank_type = actor_roles_by_rank.groupby(['year', 'n', 'type']).size().unstack(fill_value=0)

# Calculate the fraction of 'actor' roles for each rank each year
total_roles_each_year = roles_by_year_rank_type.sum(axis=1)
fraction_actor_roles_by_rank = roles_by_year_rank_type['actor'] / total_roles_each_year

# Plot the fractions for each rank over the years
plt.figure(figsize=(12, 6))
for rank in range(1, 4):  # Iterate through ranks 1, 2, and 3
    rank_label = f'n={rank}'
    plt.plot(fraction_actor_roles_by_rank[rank_label].index, fraction_actor_roles_by_rank[rank_label].values, label=f'Fraction of Actor Roles ({rank_label})')
plt.xlabel('Year')
plt.ylabel('Fraction of Actor Roles')
plt.title('Fraction of Actor Roles by Rank Each Year')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




