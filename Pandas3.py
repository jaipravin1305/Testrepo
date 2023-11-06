#!/usr/bin/env python
# coding: utf-8

# # Pandas:
# 1. Using groupby(), plot the number of films that have been released each decade in the history of cinema.

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Create a new column 'decade' to represent the decade for each movie
df['decade'] = (df['year'] // 10) * 10

# Group the data by 'decade' and count the number of films in each decade
film_counts_by_decade = df.groupby('decade').size().reset_index(name='film_count')

# Plot the number of films released each decade
plt.figure(figsize=(10, 6))
plt.bar(film_counts_by_decade['decade'], film_counts_by_decade['film_count'], width=8)
plt.xlabel('Decade')
plt.ylabel('Number of Films')
plt.title('Number of Films Released Each Decade')
plt.xticks(film_counts_by_decade['decade'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# # Pandas:
# 2. Using groupby(), plot the number of "Hamlet" films made each decade.

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only "Hamlet" films
hamlet_films = df[df['title'] == 'Hamlet']

# Create a new column 'decade' to represent the decade for each "Hamlet" movie
hamlet_films['decade'] = (hamlet_films['year'] // 10) * 10

# Group the data by 'decade' and count the number of "Hamlet" films in each decade
hamlet_counts_by_decade = hamlet_films.groupby('decade').size().reset_index(name='hamlet_count')

# Plot the number of "Hamlet" films made each decade
plt.figure(figsize=(10, 6))
plt.bar(hamlet_counts_by_decade['decade'], hamlet_counts_by_decade['hamlet_count'], width=8)
plt.xlabel('Decade')
plt.ylabel('Number of "Hamlet" Films')
plt.title('Number of "Hamlet" Films Made Each Decade')
plt.xticks(hamlet_counts_by_decade['decade'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# # Pandas:
# 3. How many leading (n=1) roles were available to actors, and how many to actresses, in each year of the 1950s?

# In[4]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'n' column is 1 (leading roles)
leading_roles = df[df['n'] == 1]

# Create a new column 'decade' to represent the decade for each movie
leading_roles['decade'] = (leading_roles['year'] // 10) * 10

# Filter the data to include only rows where the 'decade' column is in the 1950s
leading_roles_1950s = leading_roles[(leading_roles['decade'] >= 1950) & (leading_roles['decade'] < 1960)]

# Group the data by 'decade' and 'type' (actor/actress) and count the number of leading roles
leading_roles_by_year_1950s = leading_roles_1950s.groupby(['decade', 'type']).size().reset_index(name='count')

# Pivot the data to have 'decade' as rows, 'type' as columns, and 'count' as values
pivot_table = leading_roles_by_year_1950s.pivot(index='decade', columns='type', values='count')

# Fill NaN values with 0 (in case no leading roles of a specific type were found in a year)
pivot_table = pivot_table.fillna(0)

# Print the resulting pivot table
print(pivot_table)


# # Pandas:
# 4. In the 1950s decade taken as a whole, how many total roles were available to actors, and how many to actresses, for each "n" number 1 through 5?

# In[5]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows from the 1950s
roles_1950s = df[(df['year'] >= 1950) & (df['year'] < 1960)]

# Group the data by 'n' (role number) and 'type' (actor/actress) and count the number of roles
roles_by_n_and_type = roles_1950s.groupby(['n', 'type']).size().reset_index(name='count')

# Pivot the data to have 'n' as rows, 'type' as columns, and 'count' as values
pivot_table = roles_by_n_and_type.pivot(index='n', columns='type', values='count')

# Fill NaN values with 0 (in case no roles of a specific type were found for a particular "n" number)
pivot_table = pivot_table.fillna(0)

# Print the resulting pivot table
print(pivot_table)


# # Pandas:
# 5. Use groupby() to determine how many roles are listed for each of the Pink Panther movies.

# In[6]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'title' column contains "Pink Panther"
pink_panther_movies = df[df['title'].str.contains('Pink Panther', case=False)]

# Group the data by 'title' and count the number of roles for each movie
roles_per_pink_panther_movie = pink_panther_movies.groupby('title')['character'].count().reset_index()

# Print the result
print(roles_per_pink_panther_movie)


# # Pandas:
# 6. List, in order by year, each of the films in which Frank Oz has played more than 1 role.

# In[7]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'name' column is "Frank Oz"
frank_oz_movies = df[df['name'] == 'Frank Oz']

# Group the data by 'year' and 'title' and count the number of roles Frank Oz played in each movie
roles_per_frank_oz_movie = frank_oz_movies.groupby(['year', 'title'])['character'].count().reset_index()

# Filter the results to include only movies where Frank Oz played more than one role
multiple_role_movies = roles_per_frank_oz_movie[roles_per_frank_oz_movie['character'] > 1]

# Sort the filtered results by 'year' in ascending order
sorted_multiple_role_movies = multiple_role_movies.sort_values(by='year')

# Print the list of movies
print(sorted_multiple_role_movies)


# # Pandas:
# 7. List each of the characters that Frank Oz has portrayed at least twice.

# In[8]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'name' column is "Frank Oz"
frank_oz_roles = df[df['name'] == 'Frank Oz']

# Group the data by 'character' and count the number of times each character has been portrayed by Frank Oz
character_counts = frank_oz_roles.groupby('character').size().reset_index(name='count')

# Filter the results to include only characters portrayed at least twice
characters_portrayed_at_least_twice = character_counts[character_counts['count'] >= 2]

# Print the list of characters
print(characters_portrayed_at_least_twice)


# In[ ]:




