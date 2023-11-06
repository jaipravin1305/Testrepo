#!/usr/bin/env python
# coding: utf-8

# # Functions:
# 1. Write a python function to list even and odd numbers in list. 

# In[12]:


def list_even_odd():
    even_list = []
    odd_list = []
    for i in list1:
        if i % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return even_list, odd_list
list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Even numbers in the list are:", list_even_odd()[0])
print("Odd numbers in the list are:", list_even_odd()[1])


# # Functions:
# 2. Write and run a Python program that asks the user to enter 8 integers (one at a time), and then prints out how many of those integers were even numbers. For example, if the user entered 19, 6, 9, 20, 13, 7, 6, and 1, then your program should print out 3 since 3 of those numbers were even.

# In[1]:


n = int(input("Enter an integer: "))
numbers = []
for i in range(n):
    number = int(input("Enter the number you want in list: "))
    numbers.append(number)
even_count = 0
for number in numbers:
    if number % 2 == 0:
        even_count += 1
print("The total number of even numbers is", even_count)


# # Function:
# 3. Write a Python program where you take any positive integer n, if n is even, divide it by 2 to get n/2. If n is odd, multiply it by 3 and add 1 to obtain 3n+ 1. Repeat the process until you reach 1.

# In[2]:


def collatz(n):
  while n > 1:
    print(n)
    if n % 2 == 0:
      n //= 2
    else:
      n = 3 * n + 1
  print(n)
collatz(int(input("Enter a positive integer: ")))


# # Function:
# 4. Write a Python program to find the sum of all the multiples of 3 or 5 below 500.

# In[4]:


def sum_of_multiples(n):
    add = 0
    for i in range(1, n):
        if i%3 == 0 or i%5 == 0:
            add = add + i
    return add
print("The Sum of Multiples is: ", sum_of_multiples(500))


# # Functions:
# 5. To write a Python program to find first 'n' prime numbers from a list of given numbers.

# In[5]:


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


# # Functions:
# 6. To write a Python Program to compute matrix multiplication

# In[6]:


def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix")
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            dot_product = 0
            for k in range(len(B)):
                dot_product += A[i][k] * B[k][j]
            row.append(dot_product)
        result.append(row)
    return result
matrix1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

matrix2 = [
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
]

result_matrix = matrix_multiply(matrix1, matrix2)
print("The Matrix Multiplication is: ")
for row in result_matrix:
    print(row)


# # Functions:
# 7. Write a python Function to count the number of vowels.
# 

# In[7]:


def count_vowels(str1):
  vowels = "aeiouAEIOU"
  count = 0
  for i in range(len(str1)):
    if str1[i] in vowels:
      count += 1
  return count
count_vowels("Alone is Most Powerful")


# # Functions:
# 8. Write a python Function for finding factorial for the given number using a recursive

# In[8]:


def find_fact(n):
    if n == 1:
        return n
    else:
        return n*find_fact(n-1)
find_fact(int(input("Enter the number: ")))


# # Function:
# 9. Write a Python Function for generating the Fibonacci Series using the function

# In[9]:


def fibonacci(n):
  if n == 0:
    return []
  elif n == 1:
    return [0]
  else:
    fibonacci_numbers = [0, 1]
    for i in range(2, n):
      fibonacci_numbers.append(fibonacci_numbers[i - 1] + fibonacci_numbers[i - 2])
    return fibonacci_numbers
print("The Fibonacci Series is: ",fibonacci(10))


# # Functions:
# 10. Python Program to display the given integer in reverse order using function without in-built function.

# In[10]:


def reverse_integer(n):
  rev_num = 0
  while n != 0:
    rev_num = rev_num * 10 + n % 10
    n //= 10
  return rev_num
num = int(input("Enter the number: "))
reversed_num = reverse_integer(num)
print("The Reversed Number is: ",reversed_num)


# # Functions:
# 11. Write a Python function to display all integers within the range 200 - 300 whose sum digit is an even number

# In[11]:


def sum_of_digits_is_even(num):
    digit_sum = sum(int(digit) for digit in str(num))
    return digit_sum % 2 == 0

def find_numbers_with_even_digit_sum(start, end):
    for num in range(start, end + 1):
        if sum_of_digits_is_even(num):
            print(num)

start_range = 200
end_range = 300

print("Numbers with even digit sums in the range", start_range, "to", end_range, "are:")
find_numbers_with_even_digit_sum(start_range, end_range)


# # Functions:
# 12. Write a python function to find the number of digits and sum of digits in given integer

# In[12]:


def number_of_digits_and_sum(number):
  number_of_digits = 0
  sum_of_digits = 0
  while number != 0:
    number_of_digits += 1
    sum_of_digits += number % 10
    number //= 10
  return number_of_digits, sum_of_digits


number = 12345

number_of_digits, sum_of_digits = number_of_digits_and_sum(number)

print("The number of digits in", number, "is", number_of_digits)
print("The sum of digits in", number, "is", sum_of_digits)


# # Functions:
# 13. Write function called is_sorted takes a list as a parameter and returns True if the list is sorted in ascending order and False otherwise and has_duplicates that takes a list and returns True if there is any element that appears more than once. It should not modify the original list.

# In[13]:


def is_sorted(list1):
  is_sorted = True
  for i in range(len(list1) - 1):
    if list1[i] > list1[i + 1]:
      is_sorted = False
      break
  return is_sorted


def has_duplicates(list1):
  has_duplicates = False
  seen = set()
  for element in list1:
    if element in seen:
      has_duplicates = True
      break
    seen.add(element)
  return has_duplicates
lst = [1,2,3,4,5,6,6,7,7]
print(is_sorted(lst))
print(has_duplicates(lst))


# # Functions:
# 14. Write functions called nested sum that takes a list of integers and adds up the elements from all the nested lists and cumsum that takes a list of numbers and returns the cumulative sum; that is, a new list where the ith element is the sum of the first i + 1 elements from the original list.

# In[14]:


def nested_sum(list1):
  total = 0
  for sublist in list1:
    total += sum(sublist)
  return total

def cumsum(list1):
  cumsum_list = []
  for i in range(len(list1)):
    if i == 0:
      cumsum_list.append(list1[i])
    else:
      cumsum_list.append(cumsum_list[i - 1] + list1[i])
  return cumsum_list

list1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
total = nested_sum(list1)
cumsum_list = cumsum(list1)

print(cumsum_list)


# In[ ]:




