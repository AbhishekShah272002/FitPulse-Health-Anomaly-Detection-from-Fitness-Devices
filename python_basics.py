# List in Python having multiple values in a single variable. It is ordered, changeable, and allows duplicate values.
# List unbuld function are append(), insert(), remove(), pop(), clear(), sort(), reverse() and copy().
'''number = [10, 20, 30, 40, 50]
number.append(20)
print(number)

number.insert(2, 100)
print(number)'''

# Tuple in Python is a collection which is ordered and unchangeable. It allows duplicate values. 
# Tuple unbuld function are count() and index(). 
"""data = (10, 20, 30, 40, 20, 50)
print(data)
print(data[2])
data[1]= 25"""

# set in Python is a collection which is unordered and unindexed. It is written with curly brackets. It does not allow duplicate values. We can change the items, but we cannot change the order of the items. why it is called unodered because the items in a set do not have a defined order. When we print a set, the items may appear in a different order than they were added.
# Set unbuld function are add(), remove(), discard(), pop(), clear() and update().
"""values = {10, 20, 30, 40, 20, 50}
print(values) 
values.add(60)
print(values) """

# Dictionary in Python is a collection which is ordered, changeable and do not allow duplicates. It is written with curly brackets, and it has keys and values. Each key is separated from its value by a colon (:), the items are separated by commas, and the whole thing is enclosed in curly braces. 
# Dictionary unbuld function are clear(), copy(), fromkeys(), get(), items(), keys(), pop(), popitem(), setdefault() and update().
"""students= {
    "name": "ram",
    "name": "python",
    "age": 20, "course": "python basics" }

print(students) 
print(students["name"])
students["age"] = 21
print(students)"""

# CONTROL FLOW STATEMENT: It is used to control the flow of the program. It is used to execute a block of code repeatedly as long as a certain condition is true. 

#For loop
"""for i in range(5):
    print(i) # output: 0, 1, 2, 3, 4"""

"""for i in range(2, 10):
    print(i) # output: 2, 3, 4, 5, 6, 7, 8, 9"""

"""numbers = [10, 20, 30, 40]
for nom in numbers:
    print(nom) # output: 10, 20, 30, 40"""

"""words = "python"
for char in words:
    print(char) # output: p, y, t, h, o, n"""
#Type 2
"""num = [10, 20, 30, 40]
for i in num:
    print(i) # output: 10, 20, 30, 40"""

#While loop
"""i = 1
while i <= 5:
    print(i)
    i += 1 # output: 1, 2, 3, 4, 5"""

#Break statement
"""for i in range(1, 10):
    if i == 5:
        break
    print(i) # output: 1, 2, 3, 4"""
#Continue statement
"""for i in range(1, 10):
    if i == 8:
        continue
    print(i) # output: 1, 2, 3, 4, 6, 7, 9"""

'''for i in range(3, 7):
    if i == 6:
        break
    print(i) # output: 3, 4, 5'''


# Conditional statement 
# If statement 

'''age = 20
if age >= 18:
    print("You are eligible to vote.") # output: You are eligible to vote.
else:
    print("You are not eligible to vote.")'''

'''marks = 85
if marks >= 90:
    print("Grade: A")
elif marks >= 80:
    print("Grade: B")
elif marks >= 70:
    print("Grade: C")
else:
    print("Grade: D") # output: Grade: B'''


'''num = 10
if num > 0:
    print("Positive number") # output: Positive number'''

# File handling in Python
# Creating a file and writing to it with w 
"""file = open("data.txt", "w")
file.write("Hello, World!\n")
file.write("Welcome to Python programming.")
file.close()"""

# Reading from a file with r
"""file = open("data.txt", "r")
content = file.read()
print(content)
file.close()"""

# Appending to a file with a
"""file = open("data.txt", "a")
file.write("This line is appended to the file.\n   ")
file.close()"""