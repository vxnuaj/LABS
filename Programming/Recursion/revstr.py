"""
Write a function to reverse a string using recursion.
Given a string s, the function should return a new
string that is the reverse of s.

For example, if s is "hello", the function should return "olleh".


"""

def rev_str(s:str):
    if len(s) == 0:
        return ''
    return s[-1] + rev_str(s[:-1])

print(rev_str("hello"))
