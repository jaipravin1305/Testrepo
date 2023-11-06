#!/usr/bin/env python
# coding: utf-8

# # OOPS:
# 1. Write a python program to create a base class "Shape" with methods to calculate area and perimeter. Then, create derived classes "Circle" and "Rectangle" that inherit from the base class and calculate their respective areas and perimeters. Demonstrate their usage in a program.
# 
# You are developing an online quiz application where users can take quizzes on various topics and receive scores.
# 
# 1. Create a class for quizzes and questions.
# 
# 2. Implement a scoring system that calculates the user's score on a quiz.
# 
# 3. How would you store and retrieve user progress, including quiz history and scores?

# In[2]:


import math

class Shape:
    def calculate_area(self):
        pass

    def calculate_perimeter(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def calculate_area(self):
        return math.pi * self.radius ** 2

    def calculate_perimeter(self):
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def calculate_area(self):
        return self.length * self.width

    def calculate_perimeter(self):
        return 2 * (self.length + self.width)

# Demonstrate usage
circle = Circle(5)
rectangle = Rectangle(4, 6)

print(f"Circle - Area: {circle.calculate_area()}, Perimeter: {circle.calculate_perimeter()}")
print(f"Rectangle - Area: {rectangle.calculate_area()}, Perimeter: {rectangle.calculate_perimeter()}")


# In[3]:


class Question:
    def __init__(self, text, correct_answer):
        self.text = text
        self.correct_answer = correct_answer

    def check_answer(self, user_answer):
        return user_answer == self.correct_answer

class Quiz:
    def __init__(self, name, questions):
        self.name = name
        self.questions = questions

    def take_quiz(self):
        score = 0
        for question in self.questions:
            user_answer = input(question.text + " ")
            if question.check_answer(user_answer):
                score += 1
        print(f"Your score for {self.name}: {score}/{len(self.questions)}")

# Example usage
question1 = Question("What is 2 + 2?", "4")
question2 = Question("What is the capital of France?", "Paris")
quiz = Quiz("Math and Geography Quiz", [question1, question2])
quiz.take_quiz()


# # OOPS:
# 2. Write a python script to create a class "Person" with private attributes for age and name. Implement a method to calculate a person's eligibility for voting based on their age. Ensure that age cannot be accessed directly but only through a getter method.

# In[5]:


class Person:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age
    
    def get_name(self):
        return self.__name

    def get_age(self):
        return self.__age
    
    def is_eligible_to_vote(self):
        if self.__age >= 18:
            return True
        else:
            return False

# Example usage:
person1 = Person("Alice", 25)
print(f"{person1.get_name()} is eligible to vote: {person1.is_eligible_to_vote()}")


# # OOPS:
# 3. You are tasked with designing a Python class hierarchy for a simple hanking system. The system should be able to handle different types of accounts, such as Savings Accounts and Checking Accounts. Both account types should have common atributes like an account number, account holder's name, and balance. However, Savings Accounts should have an additional attribute for interest rate, while Checking Accounts should have an atribe for overdraft limit
# 
# 1. Create a Python class called Bank Account with the following attributes and
# 
# methods Attributes: account number, account holder name, balance
# 
# h. Methods: init (constructor), deposit(), and withdraw()
# 
# 2. Create two subclasses, Savings Account and Checking Account, that inherit from the BankAccount class 1
# 
# . Add the following attributes and methods in each subcla a Savings Account
# 
# Additional attribute: interest rate Method: calculate interest), which calculates and adds interest to the account based on the interest rate.
# 
# Checking Account
# 
# 1. Additional attribute: overdraft lima
# 
# i. Method: withdraw(), which allows withdrawing money up the overdraft limit (if available) without additional fees
# 
# 4. Write a program that creates instances of both Savings Account and Checking Account and demonstrates the use of their methods
# 
# Python-Practice Exercise
# 
# MLA-AIML Batch
# 
# 5. Implement proper encapsulation by making the annbuses private where necessary and providing gener and sener methods as needed.
# 
# 6 Handle any posemal ertoes i exceptions that may occur during operations like withdrawal, deposits, or interest calculations,
# 
# 7. Provide comments in your code to explain the purpose of each class, aurbine, and method
# 
# Mphasis
# 
# Note: Your code should create instances of the classes, le traction, showcase the differences here on Savings Accounts and Checking Accounts

# In[8]:


class BankAccount:
    def __init__(self, account_number, account_holder_name, balance=0):
        self.__account_number = account_number
        self.__account_holder_name = account_holder_name
        self.__balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited ${amount}. New balance: ${self.__balance}")
        else:
            print("Invalid deposit amount.")

    def withdraw(self, amount):
        if amount > 0 and amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew ${amount}. New balance: ${self.__balance}")
        else:
            print("Invalid withdrawal amount or insufficient funds.")

    def get_balance(self):
        return self.__balance

    def get_account_number(self):
        return self.__account_number


class SavingsAccount(BankAccount):
    def __init__(self, account_number, account_holder_name, balance=0, interest_rate=0.02):
        super().__init__(account_number, account_holder_name, balance)
        self.__interest_rate = interest_rate

    def calculate_interest(self):
        interest = self.get_balance() * self.__interest_rate
        self.deposit(interest)
        print(f"Interest of ${interest} added to the account.")


class CheckingAccount(BankAccount):
    def __init__(self, account_number, account_holder_name, balance=0, overdraft_limit=100):
        super().__init__(account_number, account_holder_name, balance)
        self.__overdraft_limit = overdraft_limit

    def withdraw(self, amount):
        if amount <= self.get_balance() + self.__overdraft_limit:
            super().withdraw(amount)
        else:
            print("Withdrawal exceeds overdraft limit.")


# Usage example:
if __name__ == "__main__":
    savings_account = SavingsAccount("SA123", "John Doe", 1000, 0.03)
    checking_account = CheckingAccount("CA456", "Jane Smith", 500, 200)

    savings_account.deposit(500)
    savings_account.calculate_interest()
    savings_account.withdraw(200)

    checking_account.withdraw(700)


# # OOPS:
# 4. You are developing an employee management system for a company. Ensure that the
# 
# system utilizes encapsulation and polymorphism to handle different types of employees,
# 
# such as full-time and part-time employees. 1. Create a base class called "Employee" with private attributes for name, employee ID, and salary. Implement getter and setter methods for these attributes.
# 
# 2. Design two subclasses, "FullTimeEmployee" and "PartTimeEmployee," that inherit from "Employee." These subclasses should encapsulate specific properties like hours worked (for part-time employees) and annual salary (for full-time employees).
# 
# 3. Override the salary calculation method in both subclasses to account for different payment structures.
# 
# 4. Write a program that demonstrates polymorphism by creating instances of both "FullTimeEmployee" and "PartTimeEmployee." Calculate their salaries and display employee information.

# In[9]:


class Employee:
    def __init__(self, name, employee_id, salary):
        self.__name = name
        self.__employee_id = employee_id
        self.__salary = salary

    # Getter methods
    def get_name(self):
        return self.__name

    def get_employee_id(self):
        return self.__employee_id

    def get_salary(self):
        return self.__salary

    # Setter methods
    def set_name(self, name):
        self.__name = name

    def set_employee_id(self, employee_id):
        self.__employee_id = employee_id

    def set_salary(self, salary):
        self.__salary = salary

    # Salary calculation method (to be overridden by subclasses)
    def calculate_salary(self):
        pass

# Step 2: Create subclasses "FullTimeEmployee" and "PartTimeEmployee"
class FullTimeEmployee(Employee):
    def __init__(self, name, employee_id, annual_salary):
        super().__init__(name, employee_id, annual_salary)

    # Override the salary calculation method
    def calculate_salary(self):
        return self.get_salary()

class PartTimeEmployee(Employee):
    def __init__(self, name, employee_id, hours_worked, hourly_rate):
        super().__init__(name, employee_id, 0)  # Initialize salary to 0 for part-time employees
        self.__hours_worked = hours_worked
        self.__hourly_rate = hourly_rate

    # Getter method for hours worked
    def get_hours_worked(self):
        return self.__hours_worked

    # Setter method for hours worked
    def set_hours_worked(self, hours_worked):
        self.__hours_worked = hours_worked

    # Override the salary calculation method
    def calculate_salary(self):
        return self.__hours_worked * self.__hourly_rate

# Step 4: Demonstrate polymorphism
full_time_employee = FullTimeEmployee("John Doe", 101, 50000)
part_time_employee = PartTimeEmployee("Jane Smith", 102, 20, 15.0)

employees = [full_time_employee, part_time_employee]

for employee in employees:
    print(f"Employee Name: {employee.get_name()}")
    print(f"Employee ID: {employee.get_employee_id()}")
    print(f"Salary: ${employee.calculate_salary()}")
    print()


# # OOPS:
# 5. Library Management System-Scenario: You are developing a library management system where you need to handle books, patrons, and library transactions.
# 
# 1. Create a class hierarchy that includes classes for books (e.g., Book), patrons (e.g.. Patron), and transactions (e.g.. Transaction). Define attributes and methods for each class.
# 
# 2. Implement encapsulation by making relevant attributes private and providing getter and setter methods where necessary. 3. Use inheritance to represent different types of books (e.g., fiction, non-fiction) as
# 
# subclasses of the Book class. Ensure that each book type can have specific attributes
# 
# and methods. 4. Demonstrate polymorphism by allowing patrons to check out and return books, regardless of the book type.
# 
# 5. Implement a method for tracking overdue books and notifying patrons.
# 
# 6. Consider scenarios like book reservations, late fees, and library staff interactions in your design.

# In[12]:


import datetime

class Book:
    def __init__(self, title, author, publication_date, isbn):
        self.__title = title
        self.__author = author
        self.__publication_date = publication_date
        self.__isbn = isbn
        self.__checked_out = False

    def get_title(self):
        return self.__title

    def get_author(self):
        return self.__author

    def get_publication_date(self):
        return self.__publication_date

    def get_isbn(self):
        return self.__isbn

    def is_checked_out(self):
        return self.__checked_out

    def check_out(self):
        self.__checked_out = True

    def return_book(self):
        self.__checked_out = False

class FictionBook(Book):
    def __init__(self, title, author, publication_date, isbn, genre):
        super().__init__(title, author, publication_date, isbn)
        self.__genre = genre

    def get_genre(self):
        return self.__genre

class NonFictionBook(Book):
    def __init__(self, title, author, publication_date, isbn, topic):
        super().__init__(title, author, publication_date, isbn)
        self.__topic = topic

    def get_topic(self):
        return self.__topic

class Patron:
    def __init__(self, name, patron_id):
        self.__name = name
        self.__patron_id = patron_id

    def get_name(self):
        return self.__name

    def get_patron_id(self):
        return self.__patron_id

class Transaction:
    def __init__(self, book, patron):
        self.__book = book
        self.__patron = patron
        self.__checkout_date = datetime.date.today()
        self.__due_date = self.__checkout_date + datetime.timedelta(days=14)

    def get_book(self):
        return self.__book

    def get_patron(self):
        return self.__patron

    def get_checkout_date(self):
        return self.__checkout_date

    def get_due_date(self):
        return self.__due_date

    def is_overdue(self):
        return datetime.date.today() > self.__due_date

# Sample usage of the library management system
if __name__ == "__main__":
    fiction_book = FictionBook("The Great Gatsby", "F. Scott Fitzgerald", "1925", "978-0743273565", "Classics")
    nonfiction_book = NonFictionBook("Sapiens: A Brief History of Humankind", "Yuval Noah Harari", "2014", "978-0062316097", "History")
    patron = Patron("Alice", "12345")

    transaction1 = Transaction(fiction_book, patron)
    transaction2 = Transaction(nonfiction_book, patron)

    fiction_book.check_out()
    print(fiction_book.is_checked_out())  # Output: True

    nonfiction_book.return_book()
    print(nonfiction_book.is_checked_out())  # Output: False

    print(transaction1.is_overdue())  # Output: False
    print(transaction2.is_overdue())  # Output: True


# # OOPS:
# 6.Online Shopping Cart
# 
# Scenario: You are tasked with designing a class hierarchy for an online shopping cart system. The system should handle products, shopping carts, and orders. Consider various OOP principles while designing this system.
# 
# 1. Create a class hierarchy that includes classes for products (e.g., Product), shopping carts (e.g., ShoppingCart), and orders (e.g., Order). Define attributes and methods for each class.
# 
# 2 Implement encapsulation by making relevant attributes private and providing getter and setter methods where necessary. 3. Use inheritance to represent different types of products (e.g., electronics, clothing) as
# 
# subclasses of the Product class. Ensure that each product type can have specific attributes
# 
# and methods.
# 
# 4. Demonstrate polymorphism by allowing various product types to be added to a shopping cart and calculate the total cost of items in the cart.
# 
# 5. Implement a method for placing an order, which transfers items from the shopping cart to an order.
# 
# Consider scenarios like out-of-stock products, discounts, and shipping costs in your design.

# In[13]:


class Product:
    def __init__(self, product_id, name, price):
        self.product_id = product_id
        self.name = name
        self.price = price

    def get_product_id(self):
        return self.product_id

    def get_name(self):
        return self.name

    def get_price(self):
        return self.price

class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_product(self, product, quantity=1):
        self.items.append({"product": product, "quantity": quantity})

    def remove_product(self, product):
        self.items = [item for item in self.items if item["product"] != product]

    def calculate_total_cost(self):
        total_cost = 0
        for item in self.items:
            total_cost += item["product"].get_price() * item["quantity"]
        return total_cost

class Order:
    def __init__(self, order_id, customer):
        self.order_id = order_id
        self.customer = customer
        self.items = []

    def add_item(self, product, quantity=1):
        self.items.append({"product": product, "quantity": quantity})

    def calculate_order_total(self):
        total_cost = 0
        for item in self.items:
            total_cost += item["product"].get_price() * item["quantity"]
        return total_cost

# Inheritance example
class Electronics(Product):
    def __init__(self, product_id, name, price, brand):
        super().__init__(product_id, name, price)
        self.brand = brand

# Usage
iphone = Electronics("1", "iPhone 13", 999.99, "Apple")
laptop = Electronics("2", "Dell XPS 15", 1499.99, "Dell")

cart = ShoppingCart()
cart.add_product(iphone, 2)
cart.add_product(laptop)

order = Order("123", "John Doe")
order.add_item(iphone, 2)
order.add_item(laptop)

total_cart_cost = cart.calculate_total_cost()
total_order_cost = order.calculate_order_total()

print(f"Total Cart Cost: ${total_cart_cost}")
print(f"Total Order Cost: ${total_order_cost}")


# In[ ]:




