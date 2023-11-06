#!/usr/bin/env python
# coding: utf-8

# # File Handling:
# 1.Write a python function that copies a file reading and writing upto 50 characters at a time.

# In[34]:


def copy_file(source_filename, destination_filename):
    try:
        with open(source_filename, 'r') as source_file, open(destination_filename, 'w') as destination_file:
            while True:
                chunk = source_file.read(50)
                if not chunk:
                    break
                destination_file.write(chunk)
    except FileNotFoundError:
        print("The source file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
source_file = 'source.txt'  # Replace with the name of your source file
destination_file = 'destination.txt'  # Replace with the name of your destination file

copy_file(source_file, destination_file)
print("The file is copied to destination.txt")


# # FH:
# 2. Print all numbers present in the text file and print the number of blank spaces in that file.

# In[35]:


# Define a function to extract numbers from a text file
def extract_numbers_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        numbers = [word for word in content.split() if word.isdigit()]
        return numbers

# Define a function to count blank spaces in a text file
def count_blank_spaces(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        blank_space_count = content.count(' ')
        return blank_space_count

# Replace 'your_file.txt' with the actual path to your text file
file_path = 'prob2.txt'
# Extract numbers from the file and count blank spaces
numbers = extract_numbers_from_file(file_path)
blank_space_count = count_blank_spaces(file_path)

# Print the numbers and the count of blank spaces
print("Numbers in the file:", numbers)
print("Number of blank spaces in the file:", blank_space_count)


# # FH:
# 3. Write a function called sed that takes as arguments a pattern string a replacement string and two filenames it should read the first file and write the contents into the second file creating it if necessary if the pattern string appears anywhere in the file,it should be replaced string. if an error occurs while opening,reading,writing,or closing files your program should catch the exception print an error message and exit.

# In[36]:


def sed(pattern, replacement, source_file, destination_file):
    try:
        # Read the content of the source file
        with open(source_file, 'r') as src_file:
            content = src_file.read()

        # Replace the pattern with the replacement string
        modified_content = content.replace(pattern, replacement)

        # Write the modified content to the destination file
        with open(destination_file, 'w') as dest_file:
            dest_file.write(modified_content)

    except FileNotFoundError:
        print(f"File not found: {source_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage:
pattern = 'old_string'
replacement = 'new_string'
source_file = 'input1.txt'
destination_file = 'output1.txt'

sed(pattern, replacement, source_file, destination_file)
print("The input1.txt file replaced to output1.txt")


# # FH:
# 4. Log File Analysis: You have a log file containing records of user activities on a website. Each line in the file represents a log entry with details like timestamp, user ID, and action performed. Your task is to analyze this log file.
# 
#    a. Write Python code to read the log file and extract specific information, such as the number of unique users or the      most common action.
# 
#    b. How would you handle large log files efficiently without loading the entire file into memory?

# In[50]:


from collections import Counter

log_file_path = "log_file.log"

unique_users = set()
action_counter = Counter()

with open(log_file_path, "r") as log_file:
    for line in log_file:
        # Split the line into its components (assuming a CSV-like format)
        timestamp, user_id, action = line.strip().split(",")

        # Count unique users
        unique_users.add(user_id)

        # Count actions
        action_counter[action] += 1

# Number of unique users
num_unique_users = len(unique_users)
print("Number of Unique Users:", num_unique_users)

# Most common action
most_common_action, most_common_count = action_counter.most_common(1)[0]
print("Most Common Action:", most_common_action)
print("Count:", most_common_count)


# # FH:
# 5. Text File Search and Replace: You have a text file with a large amount of text, and you want to search for specific words or phrases and replace them with new content.
# 
#    a. Write Python code to search for and replace text within a text file.
# 
#    b. How would you handle cases where you need to perform multiple replacements in a single pass?

# In[47]:


def search_and_replace(input_file, output_file, search_term, replacement):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # Replace the search term with the replacement in the current line
                modified_line = line.replace(search_term, replacement)
                outfile.write(modified_line)

        print(f"Text replacement completed. Results saved in {output_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage:
input_file = "Prob5.txt"
output_file = "Prob5replace.txt"
search_term = "old_word"
replacement = "new_word"

search_and_replace(input_file, output_file, search_term, replacement)


def search_and_replace_multiple(input_file, output_file, replacements):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # Apply all replacements in the current line
                for search_term, replacement in replacements:
                    line = line.replace(search_term, replacement)
                outfile.write(line)

        print(f"Text replacement completed. Results saved in {output_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage:
input_file = "Prob5.txt"
output_file = "Prob5replace.txt"
replacements = [("old_word1", "new_word1"), ("old_word2", "new_word2")]

search_and_replace_multiple(input_file, output_file, replacements)


# # FH:
# 6. Write a python script that concatenates the contents of multiple text files into single output files. 
# Allow the user to specify the input files and output file.

# In[41]:


def concatenate_files(input_files, output_file):
    try:
        with open(output_file, 'w') as output:
            for input_file in input_files:
                with open(input_file, 'r') as file:
                    output.write(file.read())
                    output.write('\n')  # Add a newline between concatenated files

        print(f"Concatenation completed. Output written to {output_file}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    input_files = []
    while True:
        file_path = input("Enter the path of an input text file (or type 'done' to finish): ")
        if file_path.lower() == 'done':
            break
        input_files.append(file_path)

    if not input_files:
        print("No input files provided. Exiting.")
    else:
        output_file = input("Enter the path of the output text file: ")
        concatenate_files(input_files, output_file)


# # FH:
# 7.You are given a text file named input.txt containing a list of words, one word per line. Your task is to create a Python program that reads the contents of input.txt, processes the words, and writes the result to an output file named output.txt.
# 
# a. The program should perform the following operations:
# 
# i. Read the words from input.txt.
# 
# ii. For each word in the input file, calculate the length of the word and store it in a dictionary where the word is the key, and the length is the value.
# 
# iii. Write the word-length dictionary to output.txt in the following format:
# iv. Close both input and output files properly.
# 
# v. Write Python code to accomplish this task. Ensure proper error handling for file operations.

# In[31]:


# Function to read words from input.txt and create a word-length dictionary
def create_word_length_dictionary(input_filename):
    word_length_dict = {}
    
    try:
        with open(input_filename, 'r') as input_file:
            for line in input_file:
                word = line.strip()
                length = len(word)
                word_length_dict[word] = length
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return word_length_dict

# Function to write the word-length dictionary to output.txt
def write_word_length_dictionary(output_filename, word_length_dict):
    try:
        with open(output_filename, 'w') as output_file:
            for word, length in word_length_dict.items():
                output_file.write(f"{word}: {length}\n")
    except Exception as e:
        print(f"An error occurred while writing to '{output_filename}': {e}")

# Main program
if __name__ == "__main__":
    input_filename = "input.txt"
    output_filename = "output.txt"
    
    word_length_dict = create_word_length_dictionary(input_filename)
    
    if word_length_dict:
        write_word_length_dictionary(output_filename, word_length_dict)
        print("Word-length dictionary has been written to 'output.txt'.")


# # FH:
# 8. Assume that you are developing a student gradebook system for a school. The system should allow teachers to input student grades for various subjects, store the data in files. and provide students with the ability to view their grades.Design a Python program that accomplishes the following tasks: i. Teachers should be able to input grades for students in different subjects. ii. Store the student grade data in separate text files for each subject. iii. Students should be able to view their grades for each subject iv. Implement error handling for file operations, such as file not found or permission issues.

# In[44]:


import os

def input_grades(subject):
    try:
        filename = f"{subject}_grades.txt"
        with open(filename, 'a') as file:
            student_name = input("Enter student name: ")
            grade = input(f"Enter {subject} grade for {student_name}: ")
            file.write(f"{student_name}: {grade}\n")
        print(f"{subject} grade for {student_name} has been recorded.")

    except PermissionError:
        print(f"Permission error: Cannot write to {filename}.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def view_grades(subject):
    try:
        filename = f"{subject}_grades.txt"
        if not os.path.exists(filename):
            print(f"No {subject} grades found.")
            return

        with open(filename, 'r') as file:
            grades = file.read()
            print(f"{subject} grades:\n{grades}")

    except FileNotFoundError:
        print(f"{subject} grades file not found.")
    except PermissionError:
        print(f"Permission error: Cannot read {filename}.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    while True:
        print("\nGradebook Menu:")
        print("1. Input Grades")
        print("2. View Grades")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            subject = input("Enter the subject name: ")
            input_grades(subject)
        elif choice == '2':
            subject = input("Enter the subject name: ")
            view_grades(subject)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




