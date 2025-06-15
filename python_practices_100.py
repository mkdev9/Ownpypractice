

# Practice 1: How to detect Positive and Negative Numbers
def detect_positive_negative(num):
    if num > 0:
        return "Positive"
    elif num < 0:
        return "Negative"
    else:
        return "Zero"

# Example usage:
# print(detect_positive_negative(5))   # Positive
# print(detect_positive_negative(-3))  # Negative

# Practice 2: How to check for Even and Odd Numbers
def check_even_odd(num):
    return "Even" if num % 2 == 0 else "Odd"

# Example: print(check_even_odd(4))  # Even

# Practice 3: How to check for Greatest of 3 Numbers
def greatest_of_three(a, b, c):
    return max(a, b, c)

# Alternative implementation:
def greatest_of_three_alt(a, b, c):
    if a >= b and a >= c:
        return a
    elif b >= a and b >= c:
        return b
    else:
        return c

# Practice 4: How to check for divisibility of a Number
def check_divisibility(num, divisor):
    return num % divisor == 0

# Example: print(check_divisibility(10, 2))  # True

# Practice 5: How to convert from Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

# Example: print(celsius_to_fahrenheit(25))  # 77.0

# Practice 6: How to convert from Fahrenheit to Celsius
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

# Example: print(fahrenheit_to_celsius(77))  # 25.0

# Practice 7: How to create a simple Thermometer
def simple_thermometer(temp_celsius):
    fahrenheit = celsius_to_fahrenheit(temp_celsius)
    print(f"Temperature: {temp_celsius}°C = {fahrenheit:.1f}°F")
    
    if temp_celsius < 0:
        print("Freezing!")
    elif temp_celsius < 20:
        print("Cold")
    elif temp_celsius < 30:
        print("Comfortable")
    else:
        print("Hot!")

# Practice 8: How to calculate Mass, Density and Volume
def calculate_mass(density, volume):
    return density * volume

def calculate_density(mass, volume):
    return mass / volume if volume != 0 else 0

def calculate_volume(mass, density):
    return mass / density if density != 0 else 0

# Practice 9: How to determine the quadrant of a point
def determine_quadrant(x, y):
    if x > 0 and y > 0:
        return "First Quadrant"
    elif x < 0 and y > 0:
        return "Second Quadrant"
    elif x < 0 and y < 0:
        return "Third Quadrant"
    elif x > 0 and y < 0:
        return "Fourth Quadrant"
    else:
        return "On axis"

# Practice 10: How to determine if a Triangle exists
def triangle_exists(a, b, c):
    return (a + b > c) and (a + c > b) and (b + c > a)

# Practice 11: How to check for Leap year
def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Practice 12: How to check if a point belongs to Circle
def point_in_circle(x, y, center_x, center_y, radius):
    distance_squared = (x - center_x)**2 + (y - center_y)**2
    return distance_squared <= radius**2

# Practice 13: How to create quadratic Equation
import math

def solve_quadratic(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        x1 = (-b + math.sqrt(discriminant)) / (2*a)
        x2 = (-b - math.sqrt(discriminant)) / (2*a)
        return (x1, x2)
    elif discriminant == 0:
        x = -b / (2*a)
        return (x,)
    else:
        return "No real solutions"

# Practice 14: How to make queue of Random numbers
import random

def generate_random_queue(size, min_val=1, max_val=100):
    return [random.randint(min_val, max_val) for _ in range(size)]

# Practice 15: How to print out the ASCII Table
def print_ascii_table(start=32, end=126):
    for i in range(start, end + 1):
        print(f"{i}: {chr(i)}")

# Practice 16: How to create a Multiplication Table using while loop
def multiplication_table_while(num, limit=10):
    i = 1
    while i <= limit:
        print(f"{num} x {i} = {num * i}")
        i += 1

# Practice 17: How to create Multiplication Table using for loop
def multiplication_table_for(num, limit=10):
    for i in range(1, limit + 1):
        print(f"{num} x {i} = {num * i}")

# Practice 18: How to convert from base 2 to 9
def convert_base(number, from_base, to_base):
    # Convert to decimal first
    decimal = int(str(number), from_base)
    # Convert to target base
    if to_base == 10:
        return decimal
    
    result = ""
    while decimal > 0:
        result = str(decimal % to_base) + result
        decimal //= to_base
    return result or "0"

# Practice 19: How to build a simple Calculator
def simple_calculator(a, b, operation):
    operations = {
        '+': a + b,
        '-': a - b,
        '*': a * b,
        '/': a / b if b != 0 else "Division by zero",
        '//': a // b if b != 0 else "Division by zero",
        '%': a % b if b != 0 else "Division by zero",
        '**': a ** b
    }
    return operations.get(operation, "Invalid operation")

# Practice 20: Number of digits in an Integer
def count_digits(num):
    return len(str(abs(num)))

# Practice 21: How to get Sum and Product of digits
def sum_of_digits(num):
    return sum(int(digit) for digit in str(abs(num)))

def product_of_digits(num):
    product = 1
    for digit in str(abs(num)):
        product *= int(digit)
    return product

# Practice 22: How to make a Binary search of number in an array
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Practice 23: How to sum of N series of an element
def sum_n_series(n):
    return n * (n + 1) // 2  # Sum of 1+2+3+...+n

def sum_n_squares(n):
    return n * (n + 1) * (2 * n + 1) // 6  # Sum of 1²+2²+3²+...+n²

# Practice 24: How to get value of Even and Odd digits
def sum_even_odd_digits(num):
    even_sum = odd_sum = 0
    for digit in str(abs(num)):
        d = int(digit)
        if d % 2 == 0:
            even_sum += d
        else:
            odd_sum += d
    return even_sum, odd_sum

# Practice 25: How to get a Factorial using a while loop
def factorial_while(n):
    if n < 0:
        return None
    result = 1
    i = 1
    while i <= n:
        result *= i
        i += 1
    return result

# Practice 26: How to get Factorial using for loop
def factorial_for(n):
    if n < 0:
        return None
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Practice 27: How to create a Fibonacci Sequence
def fibonacci_sequence(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

# Practice 28: How to get the value of Fibonacci Element
def fibonacci_element(n):
    if n <= 0:
        return None
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n):
        a, b = b, a + b
    return b

# Practice 29: How to find the Greatest Common Divisor
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Practice 30: How to get maximum value of a floating point number
import sys

def max_float_value():
    return sys.float_info.max

# Practice 31: How to get Prime and Complex Numbers
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def get_primes_up_to(n):
    return [i for i in range(2, n + 1) if is_prime(i)]

def is_complex_number(num):
    return isinstance(num, complex)

# Practice 32: Quadratic Equation with Solutions at specified Range of Coefficient
def quadratic_in_range(a_range, b_range, c_range):
    solutions = []
    for a in range(a_range[0], a_range[1] + 1):
        if a == 0:
            continue
        for b in range(b_range[0], b_range[1] + 1):
            for c in range(c_range[0], c_range[1] + 1):
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    solutions.append((a, b, c, solve_quadratic(a, b, c)))
    return solutions

# Practice 33: How to Reverse Numbers
def reverse_number(num):
    return int(str(abs(num))[::-1]) * (1 if num >= 0 else -1)

# Practice 34: How to expand Strings of Alphabet
def expand_alphabet_string(s):
    result = []
    i = 0
    while i < len(s):
        if i + 2 < len(s) and s[i + 1] == '-':
            start, end = ord(s[i]), ord(s[i + 2])
            result.extend([chr(j) for j in range(start, end + 1)])
            i += 3
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)

# Practice 35: How to Replace a Substring of a String
def replace_substring(text, old, new):
    return text.replace(old, new)

# Practice 36: How to select Integers from String
import re

def extract_integers(text):
    return [int(match) for match in re.findall(r'-?\d+', text)]

# Practice 37: How to sort words according to their length
def sort_words_by_length(text):
    words = text.split()
    return sorted(words, key=len)

# Practice 38: How to find the longest word in a String
def longest_word(text):
    words = text.split()
    return max(words, key=len) if words else ""

# Practice 39: How to get Percentage of Uppercase and Lowercase
def uppercase_lowercase_percentage(text):
    upper_count = sum(1 for c in text if c.isupper())
    lower_count = sum(1 for c in text if c.islower())
    total_letters = upper_count + lower_count
    
    if total_letters == 0:
        return 0, 0
    
    upper_percent = (upper_count / total_letters) * 100
    lower_percent = (lower_count / total_letters) * 100
    return upper_percent, lower_percent

# Practice 40: How to check for String Palindrome
def is_palindrome(text):
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]

# Practice 41: How to generate Random numbers using Arrays
def generate_random_array(size, min_val=1, max_val=100):
    return [random.randint(min_val, max_val) for _ in range(size)]

# Practice 42: How to get the Maximum Element in an Array
def max_element(arr):
    return max(arr) if arr else None

# Practice 43: How to get the Minimum Element in an Array
def min_element(arr):
    return min(arr) if arr else None

# Practice 44: How to get the Number of Even and Odd numbers
def count_even_odd(arr):
    even_count = sum(1 for num in arr if num % 2 == 0)
    odd_count = len(arr) - even_count
    return even_count, odd_count

# Practice 45: How to get Positive numbers out of Negative Numbers
def extract_positive_numbers(arr):
    return [num for num in arr if num > 0]

# Practice 46: How to get numbers greater than the average of an Array
def numbers_above_average(arr):
    if not arr:
        return []
    avg = sum(arr) / len(arr)
    return [num for num in arr if num > avg]

# Practice 47: How to Replace list-items with -1, 0, 1
def replace_with_signs(arr):
    return [-1 if x < 0 else 0 if x == 0 else 1 for x in arr]

# Practice 48: How to check for File Extension
def get_file_extension(filename):
    return filename.split('.')[-1] if '.' in filename else ""

def has_extension(filename, extension):
    return filename.lower().endswith(f".{extension.lower()}")

# Practice 49: How to remove symbols from Text
def remove_symbols(text):
    return ''.join(c for c in text if c.isalnum() or c.isspace())

# Practice 50: How to get Intersection of list using for loop
def intersection_for_loop(list1, list2):
    result = []
    for item in list1:
        if item in list2 and item not in result:
            result.append(item)
    return result

# Practice 51: Simple Intersection of List
def simple_intersection(list1, list2):
    return list(set(list1) & set(list2))

# Practice 52: Longest ordered sequence in ascending order
def longest_ascending_sequence(arr):
    if not arr:
        return []
    
    max_seq = current_seq = [arr[0]]
    
    for i in range(1, len(arr)):
        if arr[i] > arr[i-1]:
            current_seq.append(arr[i])
        else:
            if len(current_seq) > len(max_seq):
                max_seq = current_seq[:]
            current_seq = [arr[i]]
    
    if len(current_seq) > len(max_seq):
        max_seq = current_seq
    
    return max_seq

# Practice 53: How to get the most occurrence Element
def most_frequent_element(arr):
    from collections import Counter
    if not arr:
        return None
    return Counter(arr).most_common(1)[0][0]

# Practice 54: How to bubble sort elements of an Array
def bubble_sort(arr):
    arr = arr[:]  # Create a copy
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Practice 55: How to sort Array using Selection sorting
def selection_sort(arr):
    arr = arr[:]  # Create a copy
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# Practice 56: How to generate Matrix of Random numbers
def generate_random_matrix(rows, cols, min_val=1, max_val=100):
    return [[random.randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)]

# Practice 57: How to get the rows and columns with maximum sum of elements
def max_sum_row_col(matrix):
    if not matrix:
        return None, None
    
    # Row sums
    row_sums = [sum(row) for row in matrix]
    max_row = row_sums.index(max(row_sums))
    
    # Column sums
    col_sums = [sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix[0]))]
    max_col = col_sums.index(max(col_sums))
    
    return max_row, max_col

# Practice 58: Sum items in rows and columns of elements
def sum_rows_columns(matrix):
    if not matrix:
        return [], []
    
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix[0]))]
    
    return row_sums, col_sums

# Practice 59: How to sum diagonals of a Matrix
def sum_diagonals(matrix):
    if not matrix or len(matrix) != len(matrix[0]):
        return None, None
    
    n = len(matrix)
    main_diagonal = sum(matrix[i][i] for i in range(n))
    anti_diagonal = sum(matrix[i][n-1-i] for i in range(n))
    
    return main_diagonal, anti_diagonal

# Practice 60: How to interchange the principal diagonals of matrix
def interchange_diagonals(matrix):
    if not matrix or len(matrix) != len(matrix[0]):
        return matrix
    
    n = len(matrix)
    result = [row[:] for row in matrix]  # Deep copy
    
    for i in range(n):
        result[i][i], result[i][n-1-i] = result[i][n-1-i], result[i][i]
    
    return result

# Practice 61: How to sort columns of element by sorting the first row
def sort_columns_by_first_row(matrix):
    if not matrix:
        return matrix
    
    # Get column indices sorted by first row values
    cols = len(matrix[0])
    col_indices = sorted(range(cols), key=lambda x: matrix[0][x])
    
    # Rearrange columns
    result = []
    for row in matrix:
        result.append([row[i] for i in col_indices])
    
    return result

# Practice 62: How to check rows and columns that has particular element
def find_element_positions(matrix, element):
    positions = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == element:
                positions.append((i, j))
    return positions

# Practice 63: How to generate beautiful Unicode
def generate_unicode_art():
    symbols = ['★', '♦', '♠', '♣', '♥', '☀', '☁', '☂', '☃', '♪', '♫', '☎', '✉', '✂', '✈']
    return ''.join(random.choice(symbols) for _ in range(20))

# Practice 64: How to get prices of products
def calculate_product_prices(products, tax_rate=0.1):
    result = {}
    for product, base_price in products.items():
        total_price = base_price * (1 + tax_rate)
        result[product] = {
            'base_price': base_price,
            'tax': base_price * tax_rate,
            'total_price': total_price
        }
    return result

# Practice 65: How to make list of dictionary using 2 Lists
def create_dict_from_lists(keys, values):
    return [dict(zip(keys, values[i:i+len(keys)])) for i in range(0, len(values), len(keys))]

# Alternative: Single dictionary
def create_single_dict(keys, values):
    return dict(zip(keys, values))

# Practice 66: How to delete dictionary item
def delete_dict_item(dictionary, key):
    result = dictionary.copy()
    if key in result:
        del result[key]
    return result

# Practice 67: Return value of 2 Arguments using function
def function_two_args(a, b):
    return {
        'sum': a + b,
        'difference': a - b,
        'product': a * b,
        'quotient': a / b if b != 0 else None
    }

# Practice 68: How to fill List
def fill_list(size, value):
    return [value] * size

def fill_list_pattern(size, pattern_func):
    return [pattern_func(i) for i in range(size)]

# Practice 69: How to get the Arithmetic mean of a List
def arithmetic_mean(lst):
    return sum(lst) / len(lst) if lst else 0

# Practice 70: How to generate Fibonacci sequence using Function
def fibonacci_function(n):
    def fib_helper(a, b, count):
        if count == 0:
            return []
        return [a] + fib_helper(b, a + b, count - 1)
    
    return fib_helper(0, 1, n)

# Practice 71: How to get Fibonacci value using recursion
def fibonacci_recursive(n):
    if n <= 0:
        return None
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# Practice 72: How to get Factorial using recursion
def factorial_recursive(n):
    if n < 0:
        return None
    elif n <= 1:
        return 1
    else:
        return n * factorial_recursive(n - 1)

# Practice 73: How to get the LCM
def lcm(a, b):
    return abs(a * b) // gcd(a, b) if a and b else 0

def lcm_multiple(numbers):
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = lcm(result, numbers[i])
    return result

# Practice 74: How to Reverse Word Sequence
def reverse_word_sequence(text):
    return ' '.join(text.split()[::-1])

# Practice 75: How to search for Binary numbers
def is_binary_string(s):
    return all(c in '01' for c in s)

def find_binary_numbers(text_list):
    return [text for text in text_list if is_binary_string(text)]

# Practice 76: How to make a ring shift or recycle items of a list
def ring_shift_left(lst, positions):
    if not lst:
        return lst
    positions = positions % len(lst)
    return lst[positions:] + lst[:positions]

def ring_shift_right(lst, positions):
    return ring_shift_left(lst, -positions)

# Practice 77: How to Read Text
def read_text_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"Error reading file: {e}"

# Practice 78: How to use Read Method
def read_method_example(filename, size=-1):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read(size)
    except Exception as e:
        return f"Error: {e}"

# Practice 79: How to use ReadLine Method
def readline_method_example(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.readline().strip()
    except Exception as e:
        return f"Error: {e}"

# Practice 80: How to use ReadLines Method
def readlines_method_example(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file.readlines()]
    except Exception as e:
        return f"Error: {e}"

# Practice 81: How to Write to File
def write_to_file(filename, content, mode='w'):
    try:
        with open(filename, mode, encoding='utf-8') as file:
            file.write(content)
        return "File written successfully"
    except Exception as e:
        return f"Error writing file: {e}"

# Practice 82: How to Read Text from File to Dictionary
def text_to_dictionary(filename, delimiter='='):
    try:
        result = {}
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if delimiter in line:
                    key, value = line.split(delimiter, 1)
                    result[key.strip()] = value.strip()
        return result
    except Exception as e:
        return f"Error: {e}"

# Practice 83: How to count Number of Lines, Words and Letters in a text file
def count_file_stats(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            lines = len(content.split('\n'))
            words = len(content.split())
            letters = len(content)
            return {
                'lines': lines,
                'words': words,
                'letters': letters
            }
    except Exception as e:
        return f"Error: {e}"

# Practice 84: How to capture String Errors
def safe_string_operation(text, operation):
    try:
        if operation == 'upper':
            return text.upper()
        elif operation == 'lower':
            return text.lower()
        elif operation == 'reverse':
            return text[::-1]
        else:
            raise ValueError("Invalid operation")
    except AttributeError:
        return "Error: Input is not a string"
    except Exception as e:
        return f"Error: {e}"

# Practice 85: How to check for non existence of number
def safe_number_check(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

# Practice 86: How to display Error message for non existed files
def check_file_exists(filename):
    try:
        with open(filename, 'r') as file:
            return "File exists and is readable"
    except FileNotFoundError:
        return f"Error: File '{filename}' does not exist"
    except PermissionError:
        return f"Error: Permission denied to read '{filename}'"
    except Exception as e:
        return f"Error: {e}"

# Practice 87: How to get Division by Zero error
def safe_division(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed"
    except Exception as e:
        return f"Error: {e}"

# Practice 88: How to get Index out of Exception
def safe_list_access(lst, index):
    try:
        return lst[index]
    except IndexError:
        return f"Error: Index {index} is out of range for list of length {len(lst)}"
    except Exception as e:
        return f"Error: {e}"

# Practice 89: How to Raise Exceptions
def validate_age(age):
    if not isinstance(age, int):
        raise TypeError("Age must be an integer")
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age seems unrealistic")
    return "Valid age"

# Practice 90: How to use classes and constructor
class Person:
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email
    
    def introduce(self):
        return f"Hi, I'm {self.name}, {self.age} years old"
    
    def update_email(self, new_email):
        self.email = new_email
    
    def __str__(self):
        return f"Person(name='{self.name}', age={self.age}, email='{self.email}')"

# Practice 91: How to fill a list with natural Numbers
def natural_numbers_list(n):
    return list(range(1, n + 1))

# Practice 92: How to fill a list with Random Numbers
def random_numbers_list(size, min_val=1, max_val=100):
    return [random.randint(min_val, max_val) for _ in range(size)]

# Practice 93: How to select Even Numbers from list
def select_even_numbers(lst):
    return [num for num in lst if num % 2 == 0]

# Practice 94: How to create List from Dictionary
def dict_to_list(dictionary, mode='keys'):
    if mode == 'keys':
        return list(dictionary.keys())
    elif mode == 'values':
        return list(dictionary.values())
    elif mode == 'items':
        return list(dictionary.items())
    else:
        return list(dictionary.keys())

# Practice 95: How to unpack Matrix into one level list
def flatten_matrix(matrix):
    return [item for row in matrix for item in row]

# Alternative using itertools
def flatten_matrix_itertools(matrix):
    import itertools
    return list(itertools.chain.from_iterable(matrix))