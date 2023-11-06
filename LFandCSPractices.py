#!/usr/bin/env python
# coding: utf-8

# # Programming Fundamentals
# 
# 1. Manipulate Using a List

# In[1]:


# i) To add new elements to the end of list.
lst = [15, 286, 786]
lst.append(1040)
print("The New Elements is:\t", lst)


# In[2]:


# ii) To reverse element in the List
lst.reverse()
print("The reverse value of list is:\t", lst)


# In[3]:


# iii) To display the same list of elements in multiple times
for i in lst:
    print("The multiple printed list are:\t", lst)


# In[5]:


# iv) To concatenate two list
lst = [10, 11, 12, 13]
lst2 = [14, 15, 16, 17]
lst.extend(lst2)
print("The Concatenate value of list is:\t", lst)


# In[7]:


# v) To sort the element in the list 
lst = [1023, 8795, 564, 645,15,98]
lst.sort()
print("The Ascending sort is:\t", lst)


# # Programming Fundamentals
# 2. Write a Python Program to do in the tuples

# In[1]:


# i) Manipulate using tuples
# ii) To add the elements in the end of tuples
tup = (45, 85, 96, 33, 84)
tup2 = (54, 869, 568, 77)
tup3 = tup + tup2
print("The added to added in end of tuples:\t", tup3)


# In[2]:


# iii) To reverse element in the tuple
reversed_tup = tuple(reversed(tup3))
print("The reverse element is:\t", reversed_tup)


# In[3]:


# iv) To display elements of the same tuple multiple times
for i in tup3:
    print("The multiple printed Elements are:\t", tup3)


# In[4]:


# v) To Concatenate two tuples
concatenated_tuple = tup + tup2
print("The Concatenate of two tuples is:\t", concatenated_tuple)


# In[5]:


# To sort the elements in the list
ascend_sort = tuple(sorted(tup3))
print("The ascending sorting is:\t", ascend_sort)


# # Programming Fundamentals
# 3.Write a python program to implement following using list

# In[6]:


# i) Create a list with integer(minimum 10 numbers)
lst = [89,1254,36,58,12,15,87,56,98,65,47,33]
print("The list with integers are: \t", lst)


# In[7]:


# ii) How to display last number in list
print("To be print the last number:\t", lst[-1])


# In[8]:


# iii) Command for displaying the values from the list[0:4]
print("To be print the certain index:\t", lst[0:4])


# In[9]:


# iv) Command for displaying the values from the list[2:]
print("To be print the certain index:\t", lst[2:])


# In[10]:


# v) Command for displaying the values from the list[:6]
print("To be print the certain index:\t", lst[:6])


# # Programming Fundamentals\
# 4. Write a Python program tuple1 = (10, 50, 20, 40, 30)

# In[11]:


# i) To display the elements 10 and 50 from tuple
tuple1 = (10, 50, 20, 40, 30)
print("To print the selected index alone:\t",tuple1[0:2]) 


# In[12]:


# ii) To display the length of tuple 
print("The length of the alligned tuple is:\t", len(tuple1))


# In[13]:


# iii) To find the minimum elelment in tuple
min_element = tuple1[0]
for i in tuple1:
    if(i < min_element):
        min_element = i
print("The minimum element in the tuple is:\t", min_element)


# In[14]:


# iv) To add all elements in the tuple1.
sum = 0
for i in tuple1:
    sum+=i
print("To add all the elements in the tuple is:\t", sum)


# In[15]:


# v) To display the same tuple1 multiple times
for i in tuple1:
    print("The display of multiple times in tuples:\t", tuple1)


# # Programming Fumdamentals
# 5. Write a Python Program

# In[16]:


# i) To calculate the length of a String
string = "I'm doing Practice Exercise in Jupyter"
print("The length of the above string is: ", len(string))


# In[17]:


# ii) To reverse words in string
words = string.split()
reversed_string = ' '.join(reversed(words))
print("Reversed string:", reversed_string)


# In[18]:


# iii) To display string multiple times
mult_str = string*3
print("The same string is displayed Multiple times:\n ", mult_str)


# In[19]:


# iv) To concatenate two strings
str2 = " And the Exercise are little bit tough to do"
str3 = string + str2
print("The Concatenate of two string: ", str3)


# In[20]:


# v) Str1 = "South India ", using string slicing to display "India"
str1 = "South India"
print("To print certain word: ",str1[6:])


# # Programming Fundamentals
# 6. Perform the Following

# In[21]:


# i) Creating Dictionary

member = {"f_name" : "Virat", "l_name" : "Kholi", "age" : 34, "Location" : "Delhi"}
print("The Dictionary items are: ",member)


# In[22]:


# ii) Access values and Keys in Dictionary
first_name = member["f_name"]
last_name = member["l_name"]
age = member["age"]
location = member["Location"]

keys = member.keys()

print("First Name:", first_name)
print("Last Name:", last_name)
print("Age:", age)
print("Location:", location)
print("Keys:", keys)


# In[23]:


# iii) Update the dictionary using function
member.update({"age" : 36})
print("Update Dictionary: ",member) 


# In[24]:


# iv) Clear and Delete the Dictionary values
member.clear()
print("Delete and Clear Dictionary: ", member)


# # Programming Fundamentals
# 7. Pythom program to insert a number to any position in a list

# In[25]:


lst = [15, 84, 56, 78, 123]
lst.insert(1, 13)
print(lst)


# # Programming Fundamentals
# 8. Python program to delete an element from a list by index

# In[26]:


lst = ['delhi', 23, 'Chennai', 7, 'Jai', 40]
del lst[0]
print("List after deletion is: ", lst)


# # Programming Fundamentals
# 9. Write a program to display a number from 1 to 100

# In[28]:


for i in range(1, 101):
    print(i, end =" ")


# # Programming Fundamentals
# 10. Write a python program to find the sum of all items in tuples

# In[29]:


tup1 = (23, 45 ,98,24, 56)
tup2 = (67, 90, 123, 456, 897)
tup3 = tup1 + tup2
sum = 0
for i in tup3:
    sum = sum + i
print("Sum of all items in tuple is: ",sum)


# # Programming Fundamentals
# 11. Create a dictionary containing three lambda functions square, cube and square root.
# 
# i) E.g. dict = {'Square': function for squaring, 'Cube': function for cube, 'Squareroot': function for square root}
# 
# ii) Pass the values (input from the user) to the functions in the dictionary respectively.
# 
# iii) Then add the outputs of each function and print it.

# In[30]:


def main():
  functions = {
    'Square': lambda x: x ** 2,
    'Cube': lambda x: x ** 3,
    'Squareroot': lambda x: x ** (1 / 2)
  }

  number = int(input("Enter a number: "))

  results = []
  for function_name in functions:
    results.append(functions[function_name](number))

  print("The square of {} is {}.".format(number, results[0]))
  print("The cube of {} is {}.".format(number, results[1]))
  print("The square root of {} is {}.".format(number, results[2]))

if __name__ == "__main__":
 main()


# # Programming Fundamentals
# 12. A list of words is given. Find the words from the list that have their second character in uppercase. Is = ['hello', 'Dear', 'how', 'ARe', 'You']

# In[31]:


def find_words_with_uppercase_second_character(words):

  results = []

  for word in words:
    if word[1].isupper():
      results.append(word)

  return results

words = ['hello', 'Dear', 'hOw', 'ARe', 'You']

results = find_words_with_uppercase_second_character(words)

print(results)


# # Programming Fundamentals
# 13. A dictionary of names and their weights on earth is given. Find how much they will weigh on the moon. (Use map and lambda functions) Formula: wMoon = (wEarth GMoon) / GEarth
# 
# i) #Weight of people in kg
# 
#   WeightOnEarth (John':45, 'Shelly':65, 'Marry':35)
# ii) # Gravitational force on the Moon: 1.622 m/s2 GMoon = 1.622
# 
# iii) # Gravitational force on the Earth: 9.81 m/s2 GEarth = 9.81

# In[32]:


WeightOnEarth = {'John': 45, 'Shelly': 65, 'Marry': 35}

GMoon = 1.622
GEarth = 9.81

WeightOnMoon = map(lambda weight: (weight * GMoon) / GEarth, WeightOnEarth.values())

WeightOnMoonDict = dict(zip(WeightOnEarth.keys(), WeightOnMoon))

print("Weight on the Moon:")
for name, weight in WeightOnMoonDict.items():
    print(f"{name}: {weight} kg")


# # Control Structures
# 1. Write a Python program to find first N prime numbers.

# In[33]:


i = 1
x = int(input("Enter the number:"))
print("The N prime numbers are: ")
for k in range(1, x+1):
    c = 0
    for j in range(1, i+1):
        a = i % j
        if a == 0:
            c = c + 1

    if c == 2:
        print(i)
    else:
        k = k - 1

    i = i + 1


# # Control Structures
# 2. Write the python code that calculates the salary of an employee. Prompt the user to enter the Basic Salary, HRA, TA, and DA. Add these components to calculate the Gross Salary. Also, deduct 10% of salary from the Gross Salary to be paid as tax and display gross minus tax as net salary.

# In[34]:


Basic_Salary = int(input("Enter the Basic Salary: "))
HRA = int(input("Enter the House Rent Allowance: "))
TA = int(input("Enter the Travel Allowance: "))
DA = int(input("Enter the DA: "))
Gross_Salary = Basic_Salary + HRA + TA + DA 
Tax = (Gross_Salary*10) / 100
Salary = Gross_Salary - Tax
print("The Gross Salary is: ", Gross_Salary)
print("The deduction of Tax is: ", Tax)
print("The Salary After deduction of Tax is: ", Salary)


# # Control Structures
# 3. Write the python program to search for a string in the given list.

# In[35]:


my_list = ['apple', 'banana', 'cherry', 'date', 'fig']

search_string = 'cherry'

# Initialize a variable to store the result
found = False

for item in my_list:
    if item == search_string:
        found = True
        break

if found:
    print(f"'{search_string}' found in the list.")
else:
    print(f"'{search_string}' not found in the list.")


# # Control Structures
# 4. Write a Python function that accepts a string and calculates the number of upper-case letters and lower-case letters

# In[36]:


def count_upper_lower(string):
    upper_count = 0
    lower_count = 0
    for char in string:
        if char.isupper():
            upper_count += 1
        elif char.islower():
            lower_count += 1

    return upper_count, lower_count

# Test the function
input_string = "Hello World"
upper, lower = count_upper_lower(input_string)

print(f"Uppercase letters: {upper}")
print(f"Lowercase letters: {lower}")


# # Control Structures
# 5. Write a program to display the sum of odd numbers and even numbers that fall between 12 and 37

# In[37]:


sum_of_odd = 0
sum_of_even = 0
for i in range(12, 37+1):
    if i%2 != 0:
        sum_of_odd+=i
    else:
        sum_of_even+=i
print("The Sum of Odd: ",sum_of_odd)
print("The Sum of Even: ", sum_of_even)


# # Control Structures
# 6. Write a python program to print the table of any number

# In[38]:


def print_table(number):
  # Iterate over the numbers from 1 to 10
  for i in range(1, 11):
    print(f"{number} x {i} = {number * i}")


# Get the number from the user
number = int(input("Enter the number: "))

# Print the table of the number
print_table(number)


# # Control Structures
# 7. Write a python program to sum the first 10 prime numbers

# In[40]:


#Python program to find sum of prime numbers from 1 to n
maximum=int(input("Please enter the maximum value: "))
total=0
for Number in range(1,maximum+1):
    count=0;
    for i in range(2,(Number//2+1)):
        if(Number%i==0):
          count=count+1
          break
    if(count==0 and Number !=1):
        
        total=total+Number
print("Sum of prime numbers from 1 to %d = %d"%(maximum,total))


# # Control Structures
# 8. Write a python program to implement the arithmetic operarations using nested if statement

# In[41]:


def arithmetic_operations(num1, num2, operator):
  # Check the operator
  if operator == "+":
    return num1 + num2
  elif operator == "-":
    return num1 - num2
  elif operator == "*":
    return num1 * num2
  elif operator == "/":
    return num1 / num2
  else:
    print("Invalid operator")

num1 = int(input("Enter the first number: "))
num2 = int(input("Enter the second number: "))
operator = input("Enter the operator: ")

result = arithmetic_operations(num1, num2, operator)

print(f"The result is {result}")


# # Control Structures
# 9. Write a program to take a Temperature in Celsius and convert it to a Fahrenheit.

# In[42]:


Celsius = 40
Fahrenheit = (Celsius * 1.8) + 32
print("The value of Fahrenheit is: ", Fahrenheit)


# # Control Structures
# 10. Write a python program to find a maximum and minimum number in a list without using in-built function.

# In[ ]:


# numbers = [4, 7, 2, 9, 1, 5]
max_num = numbers[0]  
min_num = numbers[0]  
for num in numbers:
    if num > max_num:
        max_num = num  
    elif num < min_num:
        min_num = num  
print("Maximum number:", max_num)
print("Minimum number:", min_num)


# # Control Structures
# 11. Write a program in python to print out the number of seconds in 30-day month 30 days, 24 hours in a day, 60 minutes per day, 60 seconds in a minute.

# In[45]:


seconds_in_day = 60 * 60 * 24
seconds_in_month = 30 * seconds_in_day
print("The Seconds in Month: ", seconds_in_month)


# # Control Structures
# 12. Write a program in Python to print out the number of seconds in a year.

# In[ ]:


# seconds_in_day = (60 * 60) * 24
seconds_in_year = 365 * seconds_in_day 
print("The seconds in year is: ", seconds_in_year)


# # Control Structures
# 13. A high-speed train can travel at an average speed of 150 mph, how long will it take a train travelling at this speed to travel from London to Glasgow which is 414 miles away?

# In[47]:


Distance = 414
Speed = 150
Time = Distance / Speed
print("The time Travel for Glasgow is " + str(float(Time)) + " hours")


# # Control Structures
# 14. Write a python program that defines a variable called days_in_each_school_year and assign 192 to the variable. The program should then print out the total hours that you spend in school from year 7 to year 11, if each day you spend 6 hours in school days_in_each_school_year = 192

# In[48]:


days_in_each_school_year = 192
hours_spend_in_school = 6
total_hours = 192 * 6
print("The total hours that spend from year 7 to year 11: ", total_hours)


# # Control Structures
# 15. If the age of Ram, Sam and Khan are input through the keyboard, write a python program to determine the eldest and youngest of the three.

# In[49]:


def find_eldest_and_youngest(ram_age, sam_age, khan_age):
  eldest_age = ram_age
  youngest_age = ram_age

  if sam_age > eldest_age:
    eldest_age = sam_age

  if khan_age > eldest_age:
    eldest_age = khan_age

  if sam_age < youngest_age:
    youngest_age = sam_age

  if khan_age < youngest_age:
    youngest_age = khan_age

  return eldest_age, youngest_age

ram_age = int(input("Enter Ram's age: "))
sam_age = int(input("Enter Sam's age: "))
khan_age = int(input("Enter Khan's age: "))

eldest_age, youngest_age = find_eldest_and_youngest(ram_age, sam_age, khan_age)

# Print the results
print("The eldest is {} years old.".format(eldest_age))
print("The youngest is {} years old.".format(youngest_age))


# # Control Structures
# 16. Write a python program to rotate a list by right n times with and without slicing technique.

# In[50]:


def rotate_list_with_slicing(list, n):
  length = len(list)

  rotated_list = list[-n:] + list[:length - n]

  return rotated_list


def rotate_list_without_slicing(list, n):
  
  rotated_list = []

  for i in range(len(list) - n, len(list)):
    rotated_list.append(list[i])

  for i in range(0, len(list) - n):
    rotated_list.append(list[i])

  return rotated_list

list = [1, 2, 3, 4, 5]

rotated_list_with_slicing = rotate_list_with_slicing(list, 2)
rotated_list_without_slicing = rotate_list_without_slicing(list, 2)

print("Rotated list with slicing: ", rotated_list_with_slicing)
print("Rotated list without slicing: ", rotated_list_without_slicing)


# # Control Structures
# 17. Python program to print the patterns

# In[51]:


# Left triangle star pattern
n = 5

for i in range(1, n+1):
    print("*" * i)


# In[52]:


a = int(input("Enter the number: "))
k = a - 1
for i in range(a):
    for j in range(k):
        print(end=" ")
    k -= 1
    for j in range(1, i + 2):
        print("*", "", end="")
    print("\r")


# In[53]:


def printPascal(n) :
    for line in range(0, n) :
        for i in range(0, line + 1) :
            print(binomialCoeff(line, i)," ", end = "")
        print()
def binomialCoeff(n, k) :
    res = 1
    if (k > n - k) :
        k = n - k
    for i in range(0 , k) :
        res = res * (n - i)
        res = res // (i + 1)

    return res
n = 7
printPascal(n)


# In[54]:


n = 6
for i in range(1, n + 1):
    for j in range(n - i):
        print(end="")
    for k in range(1, i + 1):
        print("Python"[k - 1], end="")
    print()


# In[ ]:




