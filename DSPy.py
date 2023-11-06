#!/usr/bin/env python
# coding: utf-8

# # DS in Py:
#     
# 1. Write a Python program to find a target values in a list using linear search with following steps:
# 
#     a. Initialize the list to store the input elements.
# 
#     b. Initialize found-False.
# 
#     c. Enter the item to be searched (match_item).
# 
#     d. For each element in the list
#         1. if match item = value
#             a. return match item's position.
# 
# 
#     e. If the match item is not in the list. display an error message that the item is not found in the list.

# In[16]:


# Initialize the list
input_list = [10, 20, 30, 40, 50]

# Initialize found
found = False

# Enter the item to be searched
match_item = int(input("Enter the item to be searched: "))

# Linear search
for index, value in enumerate(input_list):
    if match_item == value:
        found = True
        position = index
        break

# Check if the item is found or not
if found:
    print(f"{match_item} found at position {position}")
else:
    print(f"{match_item} is not found in the list.")


# # DS in Py:
#     
# 2. Write a Python program to implement binary search to find the target values from the list:
# 
#     a. Create a separate function to do binary search.
# 
#     b. Get the number of inputs from the user.
# 
#     c. Store the inputs individually in a list.
# 
#     d. In binary search function at first sort the list in order to start the search from middle of the list.
# 
#     e. Compare the middle element to right and left elements to search target element. .
# 
#     f. If greater, move to right of list or else move to another side of the list
# 
#     g. Print the result along with the position of the element.

# In[17]:


def binary_search(arr, target):
    arr.sort()  # Sort the list in ascending order
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid  # Target found, return its position
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # Target not found

# Get the number of inputs from the user
num_inputs = int(input("Enter the number of inputs: "))

# Store the inputs individually in a list
input_list = []
for i in range(num_inputs):
    value = int(input(f"Enter value {i+1}: "))
    input_list.append(value)

# Get the target value from the user
target_value = int(input("Enter the target value to search for: "))

# Perform binary search
result = binary_search(input_list, target_value)

if result != -1:
    print(f"Target value {target_value} found at position {result}.")
else:
    print(f"Target value {target_value} not found in the list.")


# # DS in Py:
#     
# 3. Write a Python program for sorting a list of elements using selection sort algorithm:
# 
#     a. Assume two lists: Sorted list Initially empty and Unsorted List Given input list
# 
#     b. In the first iteration, find the smallest element in the unsorted list and place it in the sorted list
# 
#     c. In the second iteration, find the smallest element in the unsorted list and place it in the correct position by comparing with the element in the sorted list.
# 
#     d. In the third iteration, again find the smallest element in the unsorted list and place it in the correct position by comparing with the elements in the sorted list.
# 
#     e. This process continues till the unsorted list becomes empty.
# 
#     f. Display the sorted list

# In[1]:


def selection_sort(input_list):
    sorted_list = []
    while input_list:
        min_value = min(input_list)  # Find the minimum value in the unsorted list
        sorted_list.append(min_value)  # Append it to the sorted list
        input_list.remove(min_value)  # Remove it from the unsorted list
    return sorted_list

# Input list
unsorted_list = [64, 25, 12, 22, 11]

# Sort the list using selection sort
sorted_list = selection_sort(unsorted_list)

# Display the sorted list
print("Sorted List:", sorted_list)


# # DS in Py:
# 
# 4. Write a Python program for sorting a list of elements using insertion sort algorithm: 
#     
#     a. Assume two lists: Sorted list- Initially empty and Unsorted List-Given input list.
# 
#     b. In the first iteration, take the first element in the unsorted list and insert it in Sorted list.
# 
#     c. In the second iteration, take the second element in the given list and compare with the element in the sorted sub list and place it in the correct position.
# 
#     d. In the third iteration, take the third element in the given list and compare with the elements in the sorted sub list and place the elements in the correct position.
# 
#     e. This process continues until the last element is inserted in the sorted sub list.
# 
#     f. Display the sorted elements.

# In[4]:


def insertion_sort(input_list):
    sorted_list = []
    for element in input_list:
        if not sorted_list:
            sorted_list.append(element)
        else:
            inserted = False
            for i in range(len(sorted_list)):
                if element < sorted_list[i]:
                    sorted_list.insert(i, element)
                    inserted = True
                    break
            if not inserted:
                sorted_list.append(element)
    return sorted_list

# Input list
unsorted_list = [75, 66, 32578, 1547654444, 78956214]

# Sort the list using insertion sort
sorted_list = insertion_sort(unsorted_list)

# Display the sorted list
print("Sorted List:", sorted_list)


# # DS in Py:
#     
# 5.Write a Python program that performs merge sort on a list of numbers:
# 
#     a.Divide: If the given array has zero or one element, return.
# 
#         i. Otherwise
# 
#         ii. Divide the input list in to two halves each containing half of the elements. i.e. left half and right half.
# 
#     b. Conquer: Recursively sort the two lists (left half and right half).
# 
#         a. Call the merge sort on left half.
# 
#         b. Call the merge sort on right half.
# 
#     c. Combine: Combine the elements back in the input list by merging the two sorted lists into a sorted sequence.

# In[6]:


def merge_sort(input_list):
    if len(input_list) > 1:
        mid = len(input_list) // 2  # Find the middle of the list
        left_half = input_list[:mid]  # Divide the list into two halves
        right_half = input_list[mid:]

        # Recursively sort the two halves
        merge_sort(left_half)
        merge_sort(right_half)

        # Merge the sorted halves back into the original list
        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                input_list[k] = left_half[i]
                i += 1
            else:
                input_list[k] = right_half[j]
                j += 1
            k += 1

        # Check if there are any remaining elements in the left and right halves
        while i < len(left_half):
            input_list[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            input_list[k] = right_half[j]
            j += 1
            k += 1

# Input list
unsorted_list = [789, 12345, 2678, 4577, 1100]

# Sort the list using merge sort
merge_sort(unsorted_list)

# Display the sorted list
print("Sorted List:", unsorted_list)


# # DS in Py:
#     
# 6.Write a Python script to perform the following operations on a singly linked list
# 
#     a. Create a list
# 
#     b. Find the smallest element from the list
# 
#     c. Insert an element if it is not a duplicate element
# 
#     d. Display the elements in reverse order

# In[11]:


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def find_smallest(self):
        if not self.head:
            return None

        current = self.head
        smallest = current.data

        while current:
            if current.data < smallest:
                smallest = current.data
            current = current.next

        return smallest

    def insert_unique(self, data):
        if not self.head:
            self.head = Node(data)
            return

        current = self.head
        while current:
            if current.data == data:
                return  # Element already exists, don't insert it again
            if current.next is None:
                current.next = Node(data)
                return
            current = current.next

    def display_reverse(self):
        if not self.head:
            return

        stack = []
        current = self.head

        while current:
            stack.append(current.data)
            current = current.next

        while stack:
            print(stack.pop())


# Create a singly linked list
linked_list = LinkedList()
linked_list.append(1000)
linked_list.append(2087)
linked_list.append(30564)
linked_list.append(20777)
linked_list.append(403321)

# Find the smallest element
smallest = linked_list.find_smallest()
print("Smallest Element:", smallest)

# Insert an element if it's not a duplicate
linked_list.insert_unique(50)
linked_list.insert_unique(20)  # This won't be inserted since 20 is already in the list

# Display elements in reverse order
print("Elements in Reverse Order:")
linked_list.display_reverse()


# # DS in Py:
#     
# 7.Write a python program to implement the various operations for Stack ADT
# 
#     i.) Push
# 
#     ii.) Pop
# 
#     iii.) Display.

# In[7]:


class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            print("Stack is empty. Cannot pop.")
            return None

    def is_empty(self):
        return len(self.items) == 0

    def display(self):
        if not self.is_empty():
            print("Stack:")
            for item in reversed(self.items):
                print(item)
        else:
            print("Stack is empty.")


# Example usage:
stack = Stack()

stack.push(1)
stack.push(2)
stack.push(3)

stack.display()  # Display the stack

popped_item = stack.pop()
if popped_item is not None:
    print("Popped item:", popped_item)

stack.display()  # Display the updated stack


# # DS in Py:
#     
# 8. Write a python script to implement the various operations for Queue ADT
# 
#     i.) Insert ii.) Delete iii.) Display.

# In[14]:


class Queue:
    def __init__(self):
        self.items = []

    # Insert operation (enqueue)
    def enqueue(self, item):
        self.items.append(item)

    # Delete operation (dequeue)
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            return None  # Queue is empty

    # Display operation
    def display(self):
        return self.items

    # Check if the queue is empty
    def is_empty(self):
        return len(self.items) == 0

    # Get the size of the queue
    def size(self):
        return len(self.items)

# Example usage:
if __name__ == "__main__":
    q = Queue()

    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)

    print("Queue:", q.display())

    deleted_item = q.dequeue()
    print("Deleted item:", deleted_item)
    print("Queue after dequeue:", q.display())

    print("Queue size:", q.size())


# # DS in Py:
# 
# 9.Write a program in python to convert the following infix expression to its postfix form using push and pop operations of a Stack
# 
#     a. A/B^C+D E-F G
# 
#     b. (B^2-4*A*C)^(1/2) (100)

# In[15]:


def infix_to_postfix(expression):
    def precedence(operator):
        precedence_map = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
        return precedence_map.get(operator, 0)

    def is_operator(char):
        return char in "+-*/^"

    def higher_precedence(op1, op2):
        return precedence(op1) >= precedence(op2)

    postfix = []
    stack = []

    for token in expression:
        if token.isalnum():
            postfix.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                postfix.append(stack.pop())
            stack.pop()  # Remove the opening parenthesis
        elif is_operator(token):
            while stack and stack[-1] != '(' and higher_precedence(stack[-1], token):
                postfix.append(stack.pop())
            stack.append(token)

    while stack:
        postfix.append(stack.pop())

    return ''.join(postfix)


# Test the program with the given expressions
infix_expression1 = "A/B^C+D E-F G"
infix_expression2 = "(B^2-4*A*C)^(1/2) (100)"

postfix_expression1 = infix_to_postfix(infix_expression1)
postfix_expression2 = infix_to_postfix(infix_expression2)

print("Postfix Expression 1:", postfix_expression1)
print("Postfix Expression 2:", postfix_expression2)


# In[ ]:




