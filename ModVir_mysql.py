#!/usr/bin/env python
# coding: utf-8

# # Modules & Var:
# 1.Module Import and Management
# 
# Scenario: You are developing a complex Python project with multiple modules To manage the project effectively, you need to import and use various modules. Additionally, you want to organize your code using namespaces and avoid naming conflicts.
# 
# Design a Python program that demonstrates the following:
# 
# 1. Import multiple modules within your project
# 
# 2. Use the import statement to access functions, classes, and variables from imported modules.
# 
# 3. Create your custom module and use it in your main program.
# 
# 4. Handle naming conflicts and ensure proper namespacing
# 
# 5. Implement error handling for missing modules or incorrect module usage

# In[ ]:


# module1.py
def say_hello():
    print("Hello from module1")

def add_numbers(a, b):
    return a + b

module1_variable = "Variable from module1"


# In[ ]:


# module2.py
def multiply_numbers(a, b):
    return a * b

module2_variable = "Variable from module2"


# In[12]:


# main.py
import module1
import module2

# Use functions and variables from module1
module1.say_hello()
result = module1.add_numbers(5, 3)
print(f"Result from module1: {result}")

# Use functions and variables from module2
result = module2.multiply_numbers(4, 6)
print(f"Result from module2: {result}")

print(module1.module1_variable)  # Accessing module1's variable
print(module2.module2_variable)  # Accessing module2's variable

try:
    import non_existent_module  # Try importing a non-existent module
except ImportError as e:
    print(f"Error importing module: {e}")


# # 2.Virtual Envi:
# Scenario: You are working on multiple Python projects with different dependencies and versions. To avoid conflicts and ensure project isolation, you decide to use virtual environments.
# 
# Create a Python program that demonstrates the following:
# 
# 1. Create a virtual environment for a specific project.
# 
# 2. Activate and deactivate virtual environments.
# 
# 3. Install, upgrade, and uninstall packages within a virtual environment.
# 
# 4. List the installed packages in a virtual environment.
# 
# 5. Implement error handling for virtual environment operations.

# In[2]:


import os
import subprocess
import sys

def create_virtualenv(env_name):
    try:
        subprocess.check_call([sys.executable, '-m', 'venv', env_name])
        print(f"Virtual environment '{env_name}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to create virtual environment. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def activate_virtualenv(env_name):
    try:
        activate_script = os.path.join(env_name, 'Scripts' if sys.platform == 'win32' else 'bin', 'activate')
        subprocess.run(activate_script, shell=True, check=True)
        print(f"Activated virtual environment '{env_name}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to activate virtual environment. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def deactivate_virtualenv():
    try:
        subprocess.run('deactivate', shell=True, check=True)
        print("Deactivated virtual environment.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to deactivate virtual environment. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def install_package(env_name, package_name):
    try:
        subprocess.check_call([os.path.join(env_name, 'Scripts' if sys.platform == 'win32' else 'bin', 'pip'), 'install', package_name])
        print(f"Package '{package_name}' installed successfully in '{env_name}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install package in virtual environment. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def upgrade_package(env_name, package_name):
    try:
        subprocess.check_call([os.path.join(env_name, 'Scripts' if sys.platform == 'win32' else 'bin', 'pip'), 'install', '--upgrade', package_name])
        print(f"Package '{package_name}' upgraded successfully in '{env_name}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to upgrade package in virtual environment. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def uninstall_package(env_name, package_name):
    try:
        subprocess.check_call([os.path.join(env_name, 'Scripts' if sys.platform == 'win32' else 'bin', 'pip'), 'uninstall', '-y', package_name])
        print(f"Package '{package_name}' uninstalled successfully from '{env_name}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to uninstall package from virtual environment. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def list_installed_packages(env_name):
    try:
        output = subprocess.check_output([os.path.join(env_name, 'Scripts' if sys.platform == 'win32' else 'bin', 'pip'), 'freeze']).decode('utf-8')
        installed_packages = output.strip().split('\n')
        print(f"Installed packages in '{env_name}':\n{output}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to list installed packages in virtual environment. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    project_name = "my_project"  # Change this to your project name
    package_name = "requests"    # Change this to the package you want to install/upgrade/uninstall
    env_name = f"{project_name}_env"

    print("Creating virtual environment...")
    create_virtualenv(env_name)

    print("\nActivating virtual environment...")
    activate_virtualenv(env_name)

    print("\nInstalling a package in the virtual environment...")
    install_package(env_name, package_name)

    print("\nUpgrading a package in the virtual environment...")
    upgrade_package(env_name, package_name)

    print("\nUninstalling a package from the virtual environment...")
    uninstall_package(env_name, package_name)

    print("\nListing installed packages in the virtual environment...")
    list_installed_packages(env_name)

    print("\nDeactivating virtual environment...")
    deactivate_virtualenv()

if __name__ == "__main__":
    main()


# # 3.Modules:
# Module Dependency Resolution
# 
# Scenario: You are developing a Python application that relies on third-party packages. Managing dependencies and ensuring compatibility is crucial for your project's success.
# 
# Design a Python program that demonstrates the following:
# 
# 1. Use a requirements.txt file to specify project dependencies.
# 
# 2. Automatically install all project dependencies from the requirements.txt file.
# 
# 3. Ensure that the versions of installed packages are compatible.
# 
# 4. Implement error handling for dependency resolution and installatio

# In[ ]:


# import subprocess
import sys

def install_dependencies():
    try:
        # Install dependencies from requirements.txt using pip
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install dependencies. {e}")
    except FileNotFoundError:
        print("Error: 'pip' not found. Please ensure it's installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    try:
        # Check if 'requirements.txt' exists
        with open('requirements.txt') as f:
            pass
    except FileNotFoundError:
        print("Error: 'requirements.txt' not found.")
        return

    print("Installing project dependencies...")
    install_dependencies()

if __name__ == "__main__":
    main()


# # DB MySQL
# 1. Implement Inventory Management in Python with MySQL
# 
# a) Inventory management, a critical element of the supply chain, is the tracking of inventory from manufacturers to warehouses and from these facilities to a point of sale. The goal of inventory management is to have the right products in the right place at the right time.
# 
# b) The required Database is Inventory, and the required Tables are Purchases, Sales and Inventory
# 
# c) Note: Apply your thoughts to demonstrate the DB Operation in Python.

# In[1]:


import mysql.connector

# Define database connection parameters
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'Pravin@1305',
    'database': 'Inventory'
}

def add_purchase(product_id, purchase_date, quantity_purchased, cost_per_unit):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Insert the purchase record
        cursor.execute("INSERT INTO Purchases (product_id, purchase_date, quantity_purchased, cost_per_unit) VALUES (%s, %s, %s, %s)",
                       (product_id, purchase_date, quantity_purchased, cost_per_unit))

        # Update the inventory
        cursor.execute("UPDATE Inventory SET current_quantity = current_quantity + %s WHERE product_id = %s",
                       (quantity_purchased, product_id))

        # Commit the transaction
        connection.commit()

        print("Purchase added successfully.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def record_sale(product_id, sale_date, quantity_sold, selling_price_per_unit):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Check if there is enough inventory to record the sale
        cursor.execute("SELECT current_quantity FROM Inventory WHERE product_id = %s", (product_id,))
        current_quantity = cursor.fetchone()[0]

        if current_quantity >= quantity_sold:
            # Insert the sale record
            cursor.execute("INSERT INTO Sales (product_id, sale_date, quantity_sold, selling_price_per_unit) VALUES (%s, %s, %s, %s)",
                           (product_id, sale_date, quantity_sold, selling_price_per_unit))

            # Update the inventory
            cursor.execute("UPDATE Inventory SET current_quantity = current_quantity - %s WHERE product_id = %s",
                           (quantity_sold, product_id))

            # Commit the transaction
            connection.commit()

            print("Sale recorded successfully.")
        else:
            print("Not enough inventory to complete the sale.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


# # DB MySQL:
# 2. Customer Order Processing
# 
#    Scenario: You are building a customer order processing system for an e-commerce company. The system needs to interact      with a MySQL database to store customer orders, products, and order details.
# 
#    1. Design a MySQL database schema for the order processing system, including tables for customers, products, and orders.
# 
#    2. Write a Python program that connects to the database and allows customers to place new orders.
# 
#    3. Implement a feature that calculates the total cost of an order and updates product quantities in the database.
# 
#    4. How would you handle cases where a product is no longer available when a customer places an order?
# 
#    5. Develop a function to generate order reports for the company's finance department.

# In[29]:


import mysql.connector

# Define database connection parameters
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'Pravin@1305',
    'database': 'mydb'
}

def place_order(customer_id, product_id, quantity):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Check product availability
        cursor.execute("SELECT quantity_in_stock FROM products WHERE product_id = %s", (product_id,))
        stock_quantity = cursor.fetchone()[0]

        if stock_quantity >= quantity:
            # Calculate total cost
            cursor.execute("SELECT price FROM products WHERE product_id = %s", (product_id,))
            price = cursor.fetchone()[0]
            total_cost = price * quantity

            # Insert order into the database
            cursor.execute("INSERT INTO orders (customer_id) VALUES (%s)", (customer_id,))
            order_id = cursor.lastrowid

            # Insert order details
            cursor.execute("INSERT INTO order_details (order_id, product_id, quantity, total_cost) VALUES (%s, %s, %s, %s)",
                           (order_id, product_id, quantity, total_cost))

            # Update product quantity in stock
            cursor.execute("UPDATE products SET quantity_in_stock = quantity_in_stock - %s WHERE product_id = %s",
                           (quantity, product_id))

            # Commit the transaction
            connection.commit()

            print("Order placed successfully.")
        else:
            print("Product is not available in the desired quantity.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def generate_order_report(start_date, end_date):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Query orders within the specified date range
        cursor.execute("SELECT order_id, order_date, customer_id FROM orders WHERE order_date BETWEEN %s AND %s",
                       (start_date, end_date))
        orders = cursor.fetchall()

        # Print the report
        print("Order Report:")
        for order in orders:
            order_id, order_date, customer_id = order
            print(f"Order ID: {order_id}, Order Date: {order_date}, Customer ID: {customer_id}")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Example usage
#place_order(customer_id, product_id, quantity)  # Place an order
#generate_order_report('2023-01-01', '2023-12-31')  # Generate an order report


# # DB MySQL:
# 3. You are tasked with developing a Python program that connects to a MySQL database, retrieves data from a table, performs some operations on the data, and updates the database with the modified data. Please write Python code to accomplish this task.
# 
# Instructions:
# 
# 1. Assume that you have a MySQL database server running with the following details:
# 
# i. Host: localhost
# 
# ii. Port: 3306
# 
# iii. Username: your username
# 
# iv. Password: your password
# 
# v. Database Name: your database
# 
# vi. Table Name: your_table
# 
# vii. The table has the following columns: id (int), name (varchar), quantity (int).
# 
# 2. Your Python program should:
# 
# i. Connect to the MySQL database.
# 
# ii. Retrieve all records from the your table table.
# 
# iii. Calculate the total quantity of all records retrieved.
# 
# iv. Update the quantity column of each record by doubling its value.
# 
# v. Commit the changes to the database.
# 
# vi. Close the database connection.
# 
# 3. Handle any potential errors that may occur during the database connection and data manipulation, such as connection failures or SQL errors.
# 
# 4. Provide comments in your code to explain each

# In[8]:


import mysql.connector

# Define the database connection parameters
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'Pravin@1305',
    'database': 'order_processing_db'
}

try:
    # Connect to the MySQL database
    connection = mysql.connector.connect(**db_config)
    
    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    # Step 2(ii): Retrieve all records from the table
    cursor.execute("SELECT * FROM order_details")
    records = cursor.fetchall()

    # Initialize a variable to store the total quantity
    total_quantity = 0

    # Step 2(iii): Calculate the total quantity
    for record in records:
        total_quantity += record[2]

    # Step 2(iv): Update the quantity column by doubling its value
    for record in records:
        new_quantity = record[2] * 2
        cursor.execute("UPDATE your_table SET quantity = %s WHERE id = %s", (new_quantity, record[0]))

    # Commit the changes to the database
    connection.commit()

    # Step 2(vi): Close the database connection
    cursor.close()
    connection.close()

    print("Operation completed successfully.")
except mysql.connector.Error as err:
    print(f"Error: {err}")
finally:
    if connection.is_connected():
        cursor.close()
        connection.close()


# # DB MySQL:
# 4.You are developing an employee management system for a company. The database should store employee information, including name, salary, department, and hire date. Managers should be able to view and update employee details. I
# 
# 1. Design the database schema for the employee management system.
# 
# 2. Write Python code to connect to the database and retrieve a list of employees in a specific department.
# 
# 3. Implement a feature to update an employee's salary.

# In[23]:


import mysql.connector

# Define the database connection parameters
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'Pravin@1305',
    'database': 'mydb'
}

try:
    # Connect to the MySQL database
    connection = mysql.connector.connect(**db_config)

    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    # Define the department you want to retrieve employees for
    target_department = 'IT'

    # Retrieve employees in the specified department
    cursor.execute("SELECT name FROM employees WHERE department = %s", (target_department,))
    employees = cursor.fetchall()

    # Print the list of employees in the department
    print(f"Employees in the {target_department} department:")
    for employee in employees:
        print(employee[0])

    # Define the employee whose salary you want to update and the new salary
    employee_name_to_update = 'John Doe'
    new_salary = 65000.00

    # Update the salary of the specified employee
    cursor.execute("UPDATE employees SET salary = %s WHERE name = %s", (new_salary, employee_name_to_update))

    # Commit the changes to the database
    connection.commit()

    print(f"Salary of {employee_name_to_update} updated successfully.")
except mysql.connector.Error as err:
    print(f"Error: {err}")
finally:
    if connection.is_connected():
        cursor.close()
        connection.close()


# In[19]:





# In[ ]:





# In[ ]:




