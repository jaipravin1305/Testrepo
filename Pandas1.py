#!/usr/bin/env python
# coding: utf-8

# # Pandas:
# 1. How many movies are listed in the titles dataframe.

# In[7]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Count the number of unique titles in the 'title' column
number_of_titles = df['title'].nunique()

print("Number of unique titles listed in titles.csv:", number_of_titles)


# # Pandas:
# 2. What are the earliest two films listed in the titles dataframe 

# In[8]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Sort the DataFrame by the 'year' column in ascending order
sorted_df = df.sort_values(by='year')

# Get the first two rows (earliest two films) from the sorted DataFrame
earliest_films = sorted_df.head(2)

# Print the earliest two films
print("The earliest two films listed:")
print(earliest_films)


# # Pandas:
# 3. How many movies have the title "Hamlet"

# In[9]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'title' column contains "Hamlet"
hamlet_movies = df[df['title'] == 'Hamlet']

# Count the number of rows (movies) in the filtered DataFrame
number_of_hamlet_movies = len(hamlet_movies)

print("Number of movies with the title 'Hamlet':", number_of_hamlet_movies)


# # Pandas:
# 4. How many movies have the title "North by Northwest"

# In[10]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'title' column contains "North by Northwest"
north_by_northwest_movies = df[df['title'] == 'North by Northwest']

# Count the number of rows (movies) in the filtered DataFrame
number_of_north_by_northwest_movies = len(north_by_northwest_movies)

print("Number of movies with the title 'North by Northwest':", number_of_north_by_northwest_movies)


# # Pandas:
# 5. When was the first movie titled "Hamlet" made.

# In[11]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'title' column contains "Hamlet"
hamlet_movies = df[df['title'] == 'Hamlet']

# Sort the filtered DataFrame by the 'year' column in ascending order
sorted_hamlet_movies = hamlet_movies.sort_values(by='year')

# Get the first row (the earliest "Hamlet" movie)
first_hamlet_movie = sorted_hamlet_movies.iloc[0]

# Extract the release year from the row
release_year = first_hamlet_movie['year']

print("The first movie titled 'Hamlet' was released in:", release_year)


# # Pandas:
# 6. List all of the "Treasure Island" movies from earliest to most recent.

# In[12]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'title' column contains "Treasure Island"
treasure_island_movies = df[df['title'] == 'Treasure Island']

# Sort the filtered DataFrame by the 'year' column in ascending order
sorted_treasure_island_movies = treasure_island_movies.sort_values(by='year')

# Print the list of "Treasure Island" movies from earliest to most recent
print("List of 'Treasure Island' movies from earliest to most recent:")
print(sorted_treasure_island_movies)


# # Pandas:
# 7. How many movies were made in the year 1950.

# In[13]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'year' column is equal to 1950
movies_1950 = df[df['year'] == 1950]

# Count the number of rows (movies) in the filtered DataFrame
number_of_movies_1950 = len(movies_1950)

print("Number of movies made in the year 1950:", number_of_movies_1950)


# # Pandas:
# 8. How many movies were made in the year 1960.

# In[14]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'year' column is equal to 1960
movies_1960 = df[df['year'] == 1960]

# Count the number of rows (movies) in the filtered DataFrame
number_of_movies_1960 = len(movies_1960)

print("Number of movies made in the year 1960:", number_of_movies_1960)


# # Pandas:
# 9. How many movies were made from 1950 through 1959.

# In[15]:


Pa


# # Pandas:
# 10. In what years has a movie titled "Batman" been released.

# In[16]:


import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'titles.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'title' column is "Batman"
batman_movies = df[df['title'] == 'Batman']

# Extract and print the unique release years of "Batman" movies
release_years = batman_movies['year'].unique()

print("Years in which a movie titled 'Batman' has been released:")
print(sorted(release_years))


# # Pandas:
# 11. How many roles were there in the movie "inception"

# In[17]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('titles.csv')

# Filter rows where the title is "Inception"
inception_roles = df[df['title'] == 'Inception']

# Count the number of roles in "Inception"
number_of_roles = len(inception_roles)

print(f"Number of roles in 'Inception': {number_of_roles}")


# # Pandas:
# 12. How many roles in the movie "inception" are NOT ranked by an "n" value?

# In[22]:


# import pandas as pd

# Replace 'titles.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only roles in the movie "Inception"
inception_roles = df[(df['title'] == 'Inception')]

# Count the number of roles that do not have an "n" value (NaN)
roles_without_n_value = inception_roles['n'].isna().sum()
roles_without_n_value = inception_roles['n'].isna().sum()

print("Number of roles in the movie 'Inception' without an 'n' value:", roles_without_n_value)


# # Pandas:
# 13. How many roles in the movie "inception" did receive  an "n" value?

# In[23]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Inception"
inception_roles = df[df['title'] == 'Inception']

# Filter the rows where the "n" value is not NaN
inception_roles_with_n_value = inception_roles[~pd.isnull(inception_roles['n'])]

# Count the number of roles in "Inception" that received an "n" value
number_of_roles_with_n_value = len(inception_roles_with_n_value)

print(f"Number of roles in 'Inception' with 'n' value: {number_of_roles_with_n_value}")


# # Pandas:
# 14. Display the cast of "North by Northwest in their correct "n"-value order, ignoring roles that did not eam a numeric "n" value.

# In[25]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "North by Northwest"
north_by_northwest_cast = df[(df['title'] == 'North by Northwest') & (pd.to_numeric(df['n'], errors='coerce').notna())]

# Sort the cast by the "n" values in ascending order
north_by_northwest_cast = north_by_northwest_cast.sort_values(by='n')

# Display the cast
print(north_by_northwest_cast[['n', 'name']])


# # Pandas:
# 15. Display the entire cast in "n"-order of the 1972 film "sleuth"

# In[26]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Sleuth" and the year is 1972
sleuth_1972_cast = df[(df['title'] == 'Sleuth') & (df['year'] == 1972)]

# Sort the cast by the "n" values in ascending order
sleuth_1972_cast = sleuth_1972_cast.sort_values(by='n')

# Display the entire cast
print(sleuth_1972_cast[['n', 'name']])


# # Pandas:
# 16. Now display the entire cast in "n" -order of the 2007 version of "sleuth"

# In[28]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Sleuth" and the year is 2007
sleuth_2007_cast = df[(df['title'] == 'Sleuth') & (df['year'] == 2007)]

# Sort the cast by the "n" values in ascending order
sleuth_2007_cast = sleuth_2007_cast.sort_values(by='n')

# Display the entire cast
print(sleuth_2007_cast[['n', 'name']])


# # Pandas:
# 17. How many roles were credited in the silent 1921 version of hamlet?

# In[33]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Hamlet" and the year is 1921
hamlet_1921_cast = df[(df['title'] == 'Hamlet') & (df['year'] == 1921)]

# Count the number of credited roles in the silent 1921 version of "Hamlet"
number_of_credited_roles = len(hamlet_1921_cast)

print(f"Number of credited roles in the silent 1921 version of 'Hamlet': {number_of_credited_roles}")


# # Pandas:
# 18. How many roles were credited in Branagh's 1996 hamlet?

# In[34]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Hamlet," the year is 1996, and the name is "Kenneth Branagh"
hamlet_1996_branagh_cast = df[(df['title'] == 'Hamlet') & (df['year'] == 1996) & (df['name'] == 'Kenneth Branagh')]

# Count the number of credited roles for Kenneth Branagh in the 1996 version of "Hamlet"
number_of_credited_roles = len(hamlet_1996_branagh_cast)

print(f"Number of roles credited to Kenneth Branagh in the 1996 version of 'Hamlet': {number_of_credited_roles}")


# # Pandas:
# 19. How many "Hamlet" roles have been listed in all film credits through history?

# In[35]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the title is "Hamlet"
hamlet_roles = df[df['title'] == 'Hamlet']

# Count the number of "Hamlet" roles listed in all film credits through history
number_of_hamlet_roles = len(hamlet_roles)

print(f"Number of 'Hamlet' roles listed in all film credits through history: {number_of_hamlet_roles}")


# # Pandas:
# 20. How many people have played an "Ophelia"?

# In[36]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the character name is "Ophelia"
ophelia_roles = df[df['name'] == 'Ophelia']

# Count the number of people who have played the role of "Ophelia"
number_of_ophelia_actors = len(ophelia_roles)

print(f"Number of people who have played the role of 'Ophelia': {number_of_ophelia_actors}")


# # Pandas:
# 21. How many people have played a role called "The Dude".

# In[27]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFram
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'character' column is "The Dude"
the_dude_actors = df[df['character'] == 'The Dude']

# Count the number of unique actors who played the role "The Dude"
number_of_dude_actors = the_dude_actors['name'].nunique()

print("Number of people who played a role called 'The Dude':", number_of_dude_actors)


# # Pandas:
# 22. How many people have played a role called "The Stranger".

# In[29]:


# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFram
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'character' column is "The Dude"
the_dude_actors = df[df['character'] == 'The Stranger']

# Count the number of unique actors who played the role "The Dude"
number_of_dude_actors = the_dude_actors['name'].nunique()

print("Number of people who played a role called 'The Stranger':", number_of_dude_actors)


# # Pandas:
# 23. How many roles has Sidney Poitier played throughout his career.

# In[30]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'name' column is "Sidney Poitier"
sidney_poitier_roles = df[df['name'] == 'Sidney Poitier']

# Count the number of unique roles played by Sidney Poitier
number_of_roles_played = sidney_poitier_roles['character'].nunique()

print("Number of roles played by Sidney Poitier throughout his career:", number_of_roles_played)


# # Pandas:
# 24. How many roles has Judi Dench played.

# In[31]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'name' column is "Judi Dench"
judi_dench_roles = df[df['name'] == 'Judi Dench']

# Count the number of unique roles played by Judi Dench
number_of_roles_played = judi_dench_roles['character'].nunique()

print("Number of roles played by Judi Dench throughout her career:", number_of_roles_played)


# # Pandas:
# 25. List the supporting roles (having n=2) played by Cary Grant in the 1940s, in order by year.

# In[32]:


import pandas as pd

# Replace 'cast.csv' with the actual path to your CSV file
file_path = 'cast.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter the DataFrame to include only rows where the 'name' column is "Cary Grant"
# and 'n' column is 2 (supporting roles), and the 'year' column is in the 1940s
cary_grant_supporting_roles_1940s = df[(df['name'] == 'Cary Grant') & (df['n'] == 2) & (df['year'] // 10 == 194)]

# Sort the filtered DataFrame by 'year' in ascending order
sorted_cary_grant_supporting_roles_1940s = cary_grant_supporting_roles_1940s.sort_values(by='year')

# Print the list of supporting roles played by Cary Grant in the 1940s
print("Supporting roles played by Cary Grant in the 1940s:")
print(sorted_cary_grant_supporting_roles_1940s)


# # Pandas:
# 26. List the leading roles that cary Grant played in the 1940s in order by year.

# In[37]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the name is "Cary Grant" and the year is in the 1940s
cary_grant_1940s_roles = df[(df['name'] == 'Cary Grant') & (df['year'] >= 1940) & (df['year'] <= 1949)]

# Filter rows where the "n" value is 1, indicating a leading role
leading_roles = cary_grant_1940s_roles[cary_grant_1940s_roles['n'] == 1]

# Sort the leading roles by year
leading_roles_sorted = leading_roles.sort_values(by='year')

# Display the list of leading roles
print(leading_roles_sorted[['year', 'title']])


# # Pandas:
# 27. How many roles were available for actors in the 1950s?

# In[38]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is in the 1950s
roles_in_1950s = df[(df['year'] >= 1950) & (df['year'] <= 1959)]

# Count the number of roles available for actors in the 1950s
number_of_roles_in_1950s = len(roles_in_1950s)

print(f"Number of roles available for actors in the 1950s: {number_of_roles_in_1950s}")


# # Pandas:
# 28. How many roles were available for actorsses in the 1950s?

# In[39]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is in the 1950s
roles_in_1950s = df[(df['year'] >= 1950) & (df['year'] <= 1959)]

# Count the number of roles available for actors in the 1950s
number_of_roles_in_1950s = len(roles_in_1950s)

print(f"Number of roles available for actors in the 1950s: {number_of_roles_in_1950s}")


# # Pandas:
# 29. How many leading roles (n==1) were available from the beginning of film history through 1980?

# In[40]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is less than or equal to 1980 and n is 1 (indicating a leading role)
leading_roles_through_1980 = df[(df['year'] <= 1980) & (df['n'] == 1)]

# Count the number of leading roles available from the beginning of film history through 1980
number_of_leading_roles_through_1980 = len(leading_roles_through_1980)

print(f"Number of leading roles available from the beginning of film history through 1980: {number_of_leading_roles_through_1980}")


# # Pandas:
# 30. How many non -leading roles  were available through from the beginning of film history through 1980?

# In[41]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is less than or equal to 1980 and n is 1 (indicating a leading role)
leading_roles_through_1980 = df[(df['year'] <= 1980) & (df['n'] == 1)]

# Count the number of leading roles available from the beginning of film history through 1980
number_of_leading_roles_through_1980 = len(leading_roles_through_1980)

print(f"Number of leading roles available from the beginning of film history through 1980: {number_of_leading_roles_through_1980}")


# # Pandas:
# 31. How many roles through 1980 were minor enough that they did not warrant a numeric "n" rank?

# In[42]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cast.csv')

# Filter rows where the year is less than or equal to 1980 and the "n" value is not a numeric rank (NaN)
minor_roles_without_numeric_rank = df[(df['year'] <= 1980) & pd.to_numeric(df['n'], errors='coerce').isna()]

# Count the number of minor roles without a numeric "n" rank through 1980
number_of_minor_roles_without_numeric_rank = len(minor_roles_without_numeric_rank)

print(f"Number of roles through 1980 that did not warrant a numeric 'n' rank: {number_of_minor_roles_without_numeric_rank}")


# In[ ]:




