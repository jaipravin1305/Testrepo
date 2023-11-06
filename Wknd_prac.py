#!/usr/bin/env python
# coding: utf-8

# # 1. Case Study: Online Shopping Cart Exception Handling
# 
# You are working as a Python developer for an e-commerce company, and your team is responsible for building and maintaining the shopping cart module of the website. Customers can add items to their cart, view the cart contents, and proceed to checkout.
# 
# Recently, there have been reports of unexpected crashes and errors when customers interact with their shopping carts. Your task is to investigate these issues and improve the exception handling in the shopping cart code to make it more robust.
# 
# Requirements and Scenarios:
# 
# Scenario 1 - Adding Items to Cart:
# 
# When a customer adds an item to their cart, they provide the product ID and quantity. Handle exceptions that may occur during this process, such as:
# 
# 1. Product ID not found in the product catalog.
# 
# ii. Invalid quantity (eg, negative quantity or non-integer input).
# 
# Scenario 2- Viewing Cart Contents:
# 
# When a customer views their cart, display the list ohitems and their quantities. Handle exceptions that may occur during this process, such as
# 
# 1. Empty cart (no items added).
# 
# II. Unexpected errors (eg, network issues when fetching cart data).
# 
# Scenario 3- Proceeding to Checkout:
# 
# When a customer proceeds to checkout, validate the cart and process the payment. Handle exceptions that may occur during this process, such as:
# 
# 1. Insufficient stock for some items in the cart.
# 
# ii. Payment gateway errors.
# 
# lil Customer payment method declined.
# 
# Your Tasks:
# 
# 1. Review the existing shopping cart code to identify potential areas where exceptions may occur.
# 
# 2. Enhance the exception handling in the code by adding appropriate try, except, and finally blocks to handle exceptions gracefully. Provide helpful error messages to the user where applicable.
# 
# 3. Ensure that the program continues to run smoothly even when exceptions occur, and customers receive informative feedback.
# 
# 4. Test the shopping cart thoroughly with different scenarios to ensure that it handles exceptions correctly.

# In[3]:


class ProductNotFoundException(Exception):
    pass

class InvalidQuantityException(Exception):
    pass

class EmptyCartException(Exception):
    pass

class NetworkErrorException(Exception):
    pass

class InsufficientStockException(Exception):
    pass

class PaymentGatewayException(Exception):
    pass

class PaymentDeclinedException(Exception):
    pass

class Product:
    def __init__(self, product_id, name, price, stock):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.stock = stock

class ShoppingCart:
    def __init__(self):
        self.cart = []

    def add_item(self, product, quantity):
        if quantity <= 0 or not isinstance(quantity, int):
            raise InvalidQuantityException("Invalid quantity. Please provide a positive integer.")
        
        if product.stock < quantity:
            raise InsufficientStockException(f"Insufficient stock for {product.name}. Only {product.stock} available.")
        
        self.cart.append({"product": product, "quantity": quantity})

    def view_cart(self):
        if not self.cart:
            raise EmptyCartException("Your cart is empty.")
        
        for item in self.cart:
            print(f"Product: {item['product'].name}, Quantity: {item['quantity']}")

    def proceed_to_checkout(self, payment_method):
        total_price = sum(item['product'].price * item['quantity'] for item in self.cart)
        
        # Simulate payment gateway errors
        if payment_method == "credit_card_error":
            raise PaymentGatewayException("Payment gateway error. Please try again.")
        
        # Simulate payment declined
        if payment_method == "declined":
            raise PaymentDeclinedException("Payment declined. Please check your payment method.")
        
        # Process payment here (not implemented in this example)
        print(f"Payment of ${total_price:.2f} using {payment_method} was successful.")
        

# Example usage:
try:
    product_catalog = [
        Product(1, "Laptop", 800, 10),
        Product(2, "Phone", 400, 5),
        Product(3, "Tablet", 200, 8)
    ]

    cart = ShoppingCart()
    cart.add_item(product_catalog[0], 2)
    cart.add_item(product_catalog[1], 1)  # Invalid quantity
    cart.view_cart()
    cart.proceed_to_checkout("credit_card_error")  # Payment gateway error
except (ProductNotFoundException, InvalidQuantityException, EmptyCartException, NetworkErrorException,
        InsufficientStockException, PaymentGatewayException, PaymentDeclinedException) as e:
    print(f"Error: {str(e)}")
except Exception as e:
    print("An unexpected error occurred. Please try again later.")


# # Wknd:
# 2. Create a Python function that checks if two given strings are anagrams of each other.

# In[4]:


def are_anagrams(str1, str2):
    # Remove spaces and convert both strings to lowercase
    str1 = str1.replace(" ", "").lower()
    str2 = str2.replace(" ", "").lower()
    
    # Check if the sorted characters of both strings are the same
    return sorted(str1) == sorted(str2)

# Example usage:
string1 = "listen"
string2 = "silent"
result = are_anagrams(string1, string2)

if result:
    print(f"'{string1}' and '{string2}' are anagrams.")
else:
    print(f"'{string1}' and '{string2}' are not anagrams.")


# # Wknd:
# 4.Case Study: Online Bookstore Database Connectivity
# 
# You are a Python developer working on the backend of an online bookstore website. The website's database stores information about books, customers, orders, and Inventory. Your task is to develop and maintain the database connectivity and interaction components.
# 
# Requirements and Scenarios:
# 
# Scenario 1-Customer Registration:
# 
# When a new customer registers on the website, their information (name, email, password) should be stored in the database.
# 
# Handle exceptions that may occur during the registration process, such as:
# 
# 1. Duplicate email addresses.
# 
# 2. Database connection errors.
# 
# Scenario 2-Book Inventory Management:
# 
# Implement functionality to add new books to the inventory, update existing book details, and delete books.
# 
# Handle exceptions that may occur during these operations, such as:
# 
# 1. Invalid book data.
# 
# 2. Database errors when updating or deleting books.
# 
# Scenario 3- Customer Orders:
# 
# Allow customers to place orders for books. Each order includes customer details and a list of ordered books.
# 
# Handle exceptions that may occur during order placement, such as:
# 
# 1. Insufficient stock for some books.
# 
# 2. Database errors when recording orders.
# 
# Scenario 4-Order History:
# 
# Customers should be able to view their order history, which includes details of past orders
# 
# Handle exceptions that may occur when retrieving order history, such as:
# 
# 1. No orders found for the customer.
# 
# 2. Database connection issues.
# 
# Your Tasks:
# 1. Review the existing database interaction code to identify potential areas where exceptions may occur. 
# 2. Enhance the exception handling in the code by adding appropriate try, except, and finally
# blocks to handle exceptions gracefully. Provide helpful error messages to the user where
# applicable.
# 
# 3. Ensure that the program continues to run smoothly even when exceptions occur, and customers receive informative feedback.
# 
# 4. Implement database queries and transactions following best practices to maintain data integrity.
# 
# 5. Test the website's database interactions thoroughly with different scenarios to ensure that it handles exceptions correctly

# In[1]:


import sqlite3

# Database Initialization (You can create a separate script to create the database)
def initialize_database():
    conn = sqlite3.connect('bookstore.db')
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY,
            title TEXT,
            author TEXT,
            price REAL,
            stock INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date DATE,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER,
            book_id INTEGER,
            quantity INTEGER,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (book_id) REFERENCES books(id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Scenario 1 - Customer Registration
def register_customer(name, email, password):
    try:
        conn = sqlite3.connect('bookstore.db')
        cursor = conn.cursor()
        
        # Check for duplicate email
        cursor.execute('SELECT id FROM customers WHERE email = ?', (email,))
        existing_customer = cursor.fetchone()
        
        if existing_customer:
            raise Exception("Email already registered")
        
        # Insert new customer
        cursor.execute('INSERT INTO customers (name, email, password) VALUES (?, ?, ?)', (name, email, password))
        conn.commit()
        
        print("Registration successful.")
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

# Scenario 2 - Book Inventory Management
def add_book(title, author, price, stock):
    try:
        conn = sqlite3.connect('bookstore.db')
        cursor = conn.cursor()
        
        cursor.execute('INSERT INTO books (title, author, price, stock) VALUES (?, ?, ?, ?)', (title, author, price, stock))
        conn.commit()
        
        print("Book added to inventory.")
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

# Scenario 3 - Customer Orders
def place_order(customer_id, book_id_list):
    try:
        conn = sqlite3.connect('bookstore.db')
        cursor = conn.cursor()
        
        # Check stock and deduct quantities
        for book_id in book_id_list:
            cursor.execute('SELECT stock FROM books WHERE id = ?', (book_id,))
            stock = cursor.fetchone()
            
            if not stock or stock[0] <= 0:
                raise Exception(f"Insufficient stock for book id {book_id}")
            
            cursor.execute('UPDATE books SET stock = stock - 1 WHERE id = ?', (book_id,))
        
        # Insert order and order_items
        cursor.execute('INSERT INTO orders (customer_id, order_date) VALUES (?, CURRENT_TIMESTAMP)', (customer_id,))
        order_id = cursor.lastrowid
        
        for book_id in book_id_list:
            cursor.execute('INSERT INTO order_items (order_id, book_id, quantity) VALUES (?, ?, 1)', (order_id, book_id))
        
        conn.commit()
        
        print("Order placed successfully.")
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

# Scenario 4 - Order History
def get_order_history(customer_id):
    try:
        conn = sqlite3.connect('bookstore.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT orders.id, order_date, GROUP_CONCAT(books.title) as book_titles
            FROM orders
            JOIN order_items ON orders.id = order_items.order_id
            JOIN books ON order_items.book_id = books.id
            WHERE orders.customer_id = ?
            GROUP BY orders.id
            ORDER BY order_date DESC
        ''', (customer_id,))
        
        order_history = cursor.fetchall()
        
        if not order_history:
            raise Exception("No orders found for this customer.")
        
        for order in order_history:
            print(f"Order ID: {order[0]}, Order Date: {order[1]}, Books: {order[2]}")
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

# Initialize the database
initialize_database()

# Example usage:
register_customer("John Doe", "john@example.com", "password123")
add_book("The Great Gatsby", "F. Scott Fitzgerald", 12.99, 50)
add_book("To Kill a Mockingbird", "Harper Lee", 10.99, 30)
place_order(1, [1, 2])
get_order_history(1)


# # Wknd:
# 6. Read a text file containing a list of names or numbers, sort the data, and write the sorted data back to a new file.

# In[3]:


def sort_and_write(input_file, output_file):
    try:
        # Read the data from the input file
        with open(input_file, 'r') as file:
            data = file.readlines()

        # Sort the data
        data.sort()

        # Write the sorted data to the output file
        with open(output_file, 'w') as file:
            file.writelines(data)

        print(f"Data sorted and saved to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    input_file = "input1.txt"    # Replace with the name of your input file
    output_file = "output1.txt"  # Replace with the name of the output file where sorted data will be saved

    sort_and_write(input_file, output_file)


# # Wknd:
# 7. Write a Python script that compares two text files and identifies the differences between them, including added, modified, and deleted lines

# In[ ]:


# import difflib

def compare_files(file1_path, file2_path):
    # Read the contents of the first file
    with open(file1_path, 'r') as file1:
        file1_lines = file1.readlines()
    
    # Read the contents of the second file
    with open(file2_path, 'r') as file2:
        file2_lines = file2.readlines()

    # Calculate the differences between the two files
    differ = difflib.Differ()
    diff = list(differ.compare(file1_lines, file2_lines))

    added_lines = [line[2:] for line in diff if line.startswith('+ ')]
    modified_lines = [line[2:] for line in diff if line.startswith('? ')]
    deleted_lines = [line[2:] for line in diff if line.startswith('- ')]
    
    return added_lines, modified_lines, deleted_lines

if __name__ == "__main__":
    file1_path = "file1.txt"  # Replace with the path to your first text file
    file2_path = "file2.txt"  # Replace with the path to your second text file

    added, modified, deleted = compare_files(file1_path, file2_path)

    if added:
        print("Added Lines:")
        for line in added:
            print(f"+ {line}", end="")

    if modified:
        print("\nModified Lines:")
        for line in modified:
            print(f"? {line}", end="")

    if deleted:
        print("\nDeleted Lines:")
        for line in deleted:
            print(f"- {line}", end="")
    
    if not (added or modified or deleted):
        print("No differences found.")


# # Wknd:
# 8. Develop a Python program that compresses a large text file using a compression algorithm (e.g., gzip) and then decompresses it back to its original form.

# In[4]:


import gzip
import shutil

def compress_file(input_file, compressed_file):
    try:
        with open(input_file, 'rb') as f_in:
            with gzip.open(compressed_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"File '{input_file}' compressed to '{compressed_file}'")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def decompress_file(compressed_file, decompressed_file):
    try:
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(decompressed_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"File '{compressed_file}' decompressed to '{decompressed_file}'")

    except FileNotFoundError:
        print(f"Error: File '{compressed_file}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    input_file = "large_text_file.txt"           # Replace with your input file name
    compressed_file = "compressed_file.gz"       # Replace with the name for the compressed file
    decompressed_file = "decompressed_file.txt"  # Replace with the name for the decompressed file

    # Compress the input file
    compress_file(input_file, compressed_file)

    # Decompress the compressed file
    decompress_file(compressed_file, decompressed_file)


# # Wknd:
# 9. Read a binary file (e.g., an image or audio file) in Python and perform an operation, such as resizing an image or modifying audio data.

# In[5]:


from PIL import Image

def resize_image(input_image_path, output_image_path, size):
    try:
        with Image.open(input_image_path) as img:
            img = img.resize(size)
            img.save(output_image_path)
        print(f"Image resized and saved to '{output_image_path}'")

    except FileNotFoundError:
        print(f"Error: File '{input_image_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    input_image_path = "input_image.jpg"           # Replace with your input image file
    output_image_path = "resized_image.jpg"       # Replace with the name for the resized image
    target_size = (300, 200)                      # Replace with the desired dimensions

    # Resize the image
    resize_image(input_image_path, output_image_path, target_size)


# # Wknd:
# 10. Write a python program to Combine the contents of multiple text files into a single file using Python. Each file should be appended to the end of the resulting file

# In[6]:


def combine_text_files(input_files, output_file):
    try:
        with open(output_file, 'w') as output:
            for input_file in input_files:
                with open(input_file, 'r') as input:
                    output.write(input.read())
                    output.write('\n')  # Add a newline between the contents of each file

        print(f"Combined files into '{output_file}'")

    except FileNotFoundError:
        print(f"Error: One or more input files not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    input_files = ["file3.txt", "file4.txt", "file5.txt"]  # Replace with your input file names
    output_file = "combined_file.txt"                    # Replace with the name for the combined file

    # Combine the text files
    combine_text_files(input_files, output_file)


# # Wknd:
# 11. Create a Python script that accepts a text file as a command-line argument and counts the number of words, lines, and characters in the file.

# In[ ]:


# import argparse

def count_file_statistics(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        lines = content.splitlines()
        words = content.split()
        characters = len(content)

        print(f"File: {file_path}")
        print(f"Number of lines: {len(lines)}")
        print(f"Number of words: {len(words)}")
        print(f"Number of characters: {characters}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count words, lines, and characters in a text file.")
    parser.add_argument("file_path", type=str, help="Path to the input text file")

    args = parser.parse_args()

    # Count statistics for the specified file
    count_file_statistics(args.file_path)


# # Wknd:
# 12. Build a command-line calculator that accepts a mathematical expression as a string argument and evaluates it, then prints the result.

# In[ ]:


# import argparse

def calculate(expression):
    try:
        result = eval(expression)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line calculator")
    parser.add_argument("expression", type=str, help="Mathematical expression to evaluate")

    args = parser.parse_args()

    # Calculate the expression and print the result
    calculate(args.expression)


# # Wknd:
# 13. Implement a Python script that takes a CSV file and two column names as command-line arguments. The script should calculate the average of values in one column and store the result in another column in the same file.

# In[ ]:


import argparse
import pandas as pd

def calculate_average(input_csv, output_csv, source_column, result_column):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(input_csv)
        
        # Calculate the average of values in the source column
        avg = df[source_column].mean()
        
        # Add the average to the result column
        df[result_column] = avg
        
        # Save the DataFrame back to the output CSV file
        df.to_csv(output_csv, index=False)
        
        print(f"Average calculated and saved to '{output_csv}'")

    except FileNotFoundError:
        print(f"Error: File '{input_csv}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and store the average of values in a CSV column")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file")
    parser.add_argument("source_column", type=str, help="Name of the source column to calculate the average")
    parser.add_argument("result_column", type=str, help="Name of the result column to store the average")

    args = parser.parse_args()

    # Calculate the average and store it in the result column
    calculate_average(args.input_csv, args.output_csv, args.source_column, args.result_column)


# # Wknd:
# 14. Write a Python script that takes two integer command-line arguments and prints their sum.

# In[ ]:


# import argparse

def add_numbers(num1, num2):
    try:
        result = num1 + num2
        print(f"Sum of {num1} and {num2} is: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add two integers")
    parser.add_argument("num1", type=int, help="First integer")
    parser.add_argument("num2", type=int, help="Second integer")

    args = parser.parse_args()

    # Calculate and print the sum
    add_numbers(args.num1, args.num2)


# # Wknd:
# 15. Create a custom Python module that includes functions to calculate the factorial of a number and to check if a number is prime. Import and use this module in another Python script

# In[11]:


# math_operations.py

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# main.py
import math_operations

# Calculate factorial
n = 5
fact = math_operations.factorial(n)
print(f"The factorial of {n} is {fact}")

# Check if a number is prime
num = 17
if math_operations.is_prime(num):
    print(f"{num} is a prime number")
else:
    print(f"{num} is not a prime number")


# # Wknd:
# 16. Create a Python module named calculator.py that contains functions for each of the four operations (addition, subtraction, multiplication, and division). Each function should take two arguments, perform the respective operation, and return the result.

# In[12]:


# main.py
import calculator1

# Perform addition
result_add = calculator1.add(5, 3)
print(f"Addition: 5 + 3 = {result_add}")

# Perform subtraction
result_subtract = calculator1.subtract(10, 4)
print(f"Subtraction: 10 - 4 = {result_subtract}")

# Perform multiplication
result_multiply = calculator1.multiply(6, 7)
print(f"Multiplication: 6 * 7 = {result_multiply}")

# Perform division
try:
    result_divide = calculator1.divide(15, 3)
    print(f"Division: 15 / 3 = {result_divide}")
except ValueError as e:
    print(f"Error: {e}")


# In[ ]:




