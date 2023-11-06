#!/usr/bin/env python
# coding: utf-8

# # Pandas:
# 1. What are the ten most common movie names of all time?

# In[1]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Group the DataFrame by movie title and count the occurrences of each title
movie_name_counts = df['title'].value_counts()

# Get the top ten most common movie names
top_ten_common_movie_names = movie_name_counts.head(10)

print("The ten most common movie names of all time:")
print(top_ten_common_movie_names)


# # Pandas:
# 2.Which three years of the 1930s saw the most films released?

# In[2]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year falls within the 1930s (1930 to 1939)
films_in_1930s = df[(df['year'] >= 1930) & (df['year'] <= 1939)]

# Count the number of films released in each year of the 1930s
film_counts_by_year = films_in_1930s['year'].value_counts()

# Get the top three years with the most films released
top_three_years = film_counts_by_year.head(3)

print("The three years of the 1930s with the most films released:")
print(top_three_years)


# # Pandas:
# 3. Plot the number of films that have been released each decade over the history of cinema.

# In[3]:


# import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Extract the decade from the 'year' column and create a new column 'decade'
df['decade'] = (df['year'] // 10) * 10

# Group the DataFrame by 'decade' and count the number of films in each decade
film_counts_by_decade = df['decade'].value_counts().sort_index()

# Plot the number of films released each decade
plt.figure(figsize=(10, 6))
film_counts_by_decade.plot(kind='bar', color='skyblue')
plt.title('Number of Films Released Each Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Films')
plt.xticks(rotation=45)
plt.show()


# # Pandas:
# 4.Plot the number of hamlet films made each decade

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Hamlet"
hamlet_films = df[df['title'] == 'Hamlet']

# Extract the decade from the 'year' column and create a new column 'decade'
hamlet_films['decade'] = (hamlet_films['year'] // 10) * 10

# Group the DataFrame by 'decade' and count the number of "Hamlet" films in each decade
hamlet_counts_by_decade = hamlet_films['decade'].value_counts().sort_index()

# Plot the number of "Hamlet" films made each decade
plt.figure(figsize=(10, 6))
hamlet_counts_by_decade.plot(kind='bar', color='skyblue')
plt.title('Number of "Hamlet" Films Made Each Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Films')
plt.xticks(rotation=45)
plt.show()


# # Pandas:
# 5. Plot the number of "Rustler" characters in each decade of the history of the film.

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the character name is "Rustler"
rustler_characters = df[df['character'] == 'Rustler']

# Extract the decade from the 'year' column and create a new column 'decade'
rustler_characters['decade'] = (rustler_characters['year'] // 10) * 10

# Group the DataFrame by 'decade' and count the number of "Rustler" characters in each decade
rustler_counts_by_decade = rustler_characters['decade'].value_counts().sort_index()

# Plot the number of "Rustler" characters in each decade
plt.figure(figsize=(10, 6))
rustler_counts_by_decade.plot(kind='bar', color='skyblue')
plt.title('Number of "Rustler" Characters in Each Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Characters')
plt.xticks(rotation=45)
plt.show()


# # Pandas:
# 6.Plot the number of "Hamlet" characters each decade

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Hamlet"
hamlet_characters = df[df['title'] == 'Hamlet']

# Extract the decade from the 'year' column and create a new column 'decade'
hamlet_characters['decade'] = (hamlet_characters['year'] // 10) * 10

# Group the DataFrame by 'decade' and count the number of "Hamlet" characters in each decade
hamlet_counts_by_decade = hamlet_characters['decade'].value_counts().sort_index()

# Plot the number of "Hamlet" characters in each decade
plt.figure(figsize=(10, 6))
hamlet_counts_by_decade.plot(kind='bar', color='skyblue')
plt.title('Number of "Hamlet" Characters in Each Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Characters')
plt.xticks(rotation=45)
plt.show()


# # Pandas:
# 7. What are the 11 most common character names in movie history?

# In[7]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Group the DataFrame by character name and count the occurrences of each character name
character_name_counts = df['character'].value_counts()

# Get the top 11 most common character names
top_11_common_character_names = character_name_counts.head(11)

print("The 11 most common character names in movie history:")
print(top_11_common_character_names)


# # Pandas:
# 8. Who are the 10 people most often credited as "Harself" in film history?

# In[8]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the character name is "Herself"
herself_credits = df[df['character'] == 'Herself']

# Group the DataFrame by actor name and count the number of times each actor was credited as "Herself"
top_10_herself_actors = herself_credits['name'].value_counts().head(10)

print("The 10 people most often credited as 'Herself' in film history:")
print(top_10_herself_actors)


# # Pandas:
# 9. Who are the 10 people most often credited as "Himself" in film history?

# In[9]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the character name is "Himself"
himself_credits = df[df['character'] == 'Himself']

# Group the DataFrame by actor name and count the number of times each actor was credited as "Himself"
top_10_himself_actors = himself_credits['name'].value_counts().head(10)

print("The 10 people most often credited as 'Himself' in film history:")
print(top_10_himself_actors)


# # Pandas:
# 10. Which actors or actressess appeared in the most movies in the year 1945?

# In[10]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is 1945
movies_in_1945 = df[df['year'] == 1945]

# Group the DataFrame by actor/actress name and count the number of movies for each
most_appearances_1945 = movies_in_1945['name'].value_counts().reset_index()
most_appearances_1945.columns = ['Actor/Actress', 'Number of Movies']

# Find the actor/actress with the most movie appearances in 1945
top_actor_1945 = most_appearances_1945.iloc[0]

print("Actor/Actress with the most movie appearances in 1945:")
print(top_actor_1945)


# # Pandas:
# 11. Which actors or actressess appeared in the most movies in the year 1985?

# In[11]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is 1985
movies_in_1985 = df[df['year'] == 1985]

# Group the DataFrame by actor/actress name and count the number of movies for each
most_appearances_1985 = movies_in_1985['name'].value_counts().reset_index()
most_appearances_1985.columns = ['Actor/Actress', 'Number of Movies']

# Find the actor/actress with the most movie appearances in 1985
top_actor_1985 = most_appearances_1985.iloc[0]

print("Actor/Actress with the most movie appearances in 1985:")
print(top_actor_1985)


# # Pandas:
# 12. Plot how many roles Mammotty has played in each year of his carrer

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the name is "Mammootty"
mammootty_roles = df[df['name'] == 'Mammootty']

# Group the DataFrame by year and count the number of roles in each year
roles_by_year = mammootty_roles.groupby('year').size()

# Plot the number of roles Mammootty has played in each year of his career
plt.figure(figsize=(12, 6))
roles_by_year.plot(kind='bar', color='skyblue')
plt.title('Number of Roles Played by Mammootty Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Roles')
plt.xticks(rotation=45)
plt.show()


# # Pandas:
# 13. What are the 10 most frequent roles that start with the phrase "Parton in"?

# In[13]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the character name starts with "Parton in"
parton_in_roles = df[df['character'].str.startswith('Parton in')]

# Count the occurrences of each role and get the top 10 most frequent roles
top_10_parton_in_roles = parton_in_roles['character'].value_counts().head(10)

print("The 10 most frequent roles that start with 'Parton in':")
print(top_10_parton_in_roles)


# # Pandas:
# 14. What are the 10 most frequent roles that start with the word "Science"?

# In[14]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the character name starts with "Science"
science_roles = df[df['character'].str.startswith('Science')]

# Count the occurrences of each role and get the top 10 most frequent roles
top_10_science_roles = science_roles['character'].value_counts().head(10)

print("The 10 most frequent roles that start with 'Science':")
print(top_10_science_roles)


# # Pandas:
# 15. Plot n-values of the roles that Judi Dench has played over her carrer.

# In[15]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the name is "Judi Dench"
judi_dench_roles = df[df['name'] == 'Judi Dench']

# Remove rows where the 'n' column is not numeric
judi_dench_roles = judi_dench_roles[pd.to_numeric(judi_dench_roles['n'], errors='coerce').notna()]

# Convert the 'n' column to numeric
judi_dench_roles['n'] = pd.to_numeric(judi_dench_roles['n'])

# Plot the n-values of the roles Judi Dench has played
plt.figure(figsize=(12, 6))
plt.scatter(judi_dench_roles['year'], judi_dench_roles['n'], color='skyblue', alpha=0.5)
plt.title('n-Values of Roles Played by Judi Dench Over Her Career')
plt.xlabel('Year')
plt.ylabel('n-Value')
plt.grid(True)
plt.show()


# # Pandas:
# 16. Plot the n-values of Cary Grants roles through his carrer

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the name is "Cary Grant"
cary_grant_roles = df[df['name'] == 'Cary Grant']

# Remove rows where the 'n' column is not numeric
cary_grant_roles = cary_grant_roles[pd.to_numeric(cary_grant_roles['n'], errors='coerce').notna()]

# Convert the 'n' column to numeric
cary_grant_roles['n'] = pd.to_numeric(cary_grant_roles['n'])

# Plot the n-values of Cary Grant's roles throughout his career
plt.figure(figsize=(12, 6))
plt.scatter(cary_grant_roles['year'], cary_grant_roles['n'], color='skyblue', alpha=0.5)
plt.title('n-Values of Roles Played by Cary Grant Throughout His Career')
plt.xlabel('Year')
plt.ylabel('n-Value')
plt.grid(True)
plt.show()


# # Pandas:
# 17. Plot the n-values of the roles that Sidney Poitier has acted over the years.

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the name is "Sidney Poitier"
sidney_poitier_roles = df[df['name'] == 'Sidney Poitier']

# Remove rows where the 'n' column is not numeric
sidney_poitier_roles = sidney_poitier_roles[pd.to_numeric(sidney_poitier_roles['n'], errors='coerce').notna()]

# Convert the 'n' column to numeric
sidney_poitier_roles['n'] = pd.to_numeric(sidney_poitier_roles['n'])

# Plot the n-values of Sidney Poitier's roles over the years
plt.figure(figsize=(12, 6))
plt.scatter(sidney_poitier_roles['year'], sidney_poitier_roles['n'], color='skyblue', alpha=0.5)
plt.title('n-Values of Roles Played by Sidney Poitier Over the Years')
plt.xlabel('Year')
plt.ylabel('n-Value')
plt.grid(True)
plt.show()


# # Pandas:
# 18. How many leading(n==1) roles were available to actors and how many to actresses in the 1950s?

# In[18]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the 'year' is in the 1950s
roles_in_1950s = df[(df['year'] >= 1950) & (df['year'] <= 1959)]

# Count the number of leading roles (n==1) for actors and actresses separately
leading_roles_actors = roles_in_1950s[(roles_in_1950s['n'] == 1) & (roles_in_1950s['type'] == 'actor')]
leading_roles_actresses = roles_in_1950s[(roles_in_1950s['n'] == 1) & (roles_in_1950s['type'] == 'actress')]

# Get the counts
num_leading_roles_actors = len(leading_roles_actors)
num_leading_roles_actresses = len(leading_roles_actresses)

print(f"Number of leading roles (n==1) for actors in the 1950s: {num_leading_roles_actors}")
print(f"Number of leading roles (n==1) for actresses in the 1950s: {num_leading_roles_actresses}")


# # Pandas:
# 19. How many supporting(n==2) roles were available to actors and how many to actresses in the 1950s?

# In[19]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the 'year' is in the 1950s
roles_in_1950s = df[(df['year'] >= 1950) & (df['year'] <= 1959)]

# Count the number of supporting roles (n==2) for actors and actresses separately
supporting_roles_actors = roles_in_1950s[(roles_in_1950s['n'] == 2) & (roles_in_1950s['type'] == 'actor')]
supporting_roles_actresses = roles_in_1950s[(roles_in_1950s['n'] == 2) & (roles_in_1950s['type'] == 'actress')]

# Get the counts
num_supporting_roles_actors = len(supporting_roles_actors)
num_supporting_roles_actresses = len(supporting_roles_actresses)

print(f"Number of supporting roles (n==2) for actors in the 1950s: {num_supporting_roles_actors}")
print(f"Number of supporting roles (n==2) for actresses in the 1950s: {num_supporting_roles_actresses}")


# In[ ]:




