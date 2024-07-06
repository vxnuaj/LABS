nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

sq_num = [n*n for n in nums]

# --- 

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

nums = [n for n in nums if n % 2 == 0]

# --- 

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

nums = [n ** 3 for n in nums if n % 3 == 0]

# ---

nums = [(x, y) for x in range(1, 4) for y in range(1, 4)]

# --- 

letters = ['A', 'B']
nums = [1, 2]

pairs = [(x, y) for x in letters for y in nums] 

# ---

nums = [1, 2, 3, 4, 5, 6]

prime_nums = [n for n in nums if n % 2 == 1]

# ---

strings = ['abc', 'abcd', 'abcde', 'abcdef']

filtered_strings = [string for string in strings if len(string) <= 4]

# ---

dict = {'A': 1, 'B': 2, 'C': 3, 'D': 3}

filtered_dict = [key for key in dict if key in ['A', 'B']]

print(filtered_dict)