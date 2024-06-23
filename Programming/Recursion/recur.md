1. **Factorial Calculation:**
   Write a recursive function to calculate the factorial of a non-negative integer n. The factorial of n (n!) is the product of all positive integers less than or equal to n. 
   Example: factorial(5) should return 120.

2. **Reverse a String:**
   Write a recursive function to reverse a string. The function should take a string as input and return the string in reverse order.
   Example: reverse("hello") should return "olleh".

3. **Sum of Digits:**
   Write a recursive function to find the sum of digits of a positive integer. The function should take a positive integer and return the sum of its digits.
   Example: sumOfDigits(123) should return 6.

4. **Power Function:**
   Write a recursive function to calculate the value of x raised to the power of y (x^y) without using the built-in power operator (**). The function should take two integers, x and y, and return the result of x raised to the power y. Recursion is used to break down the problem into smaller subproblems.
   Example: power(2, 3) should return 8.

   **Hint:** The recursive definition can be:
   - Base case: If y is 0, return 1 (since any number raised to the power of 0 is 1).
   - Recursive case: If y is positive, return x multiplied by the power of x raised to y-1.


5. **Fibonacci Sequence:**
   Write a recursive function to calculate the nth Fibonacci number. The Fibonacci sequence is defined as F(0) = 0, F(1) = 1, and F(n) = F(n-1) + F(n-2) for n > 1.
   Example: fibonacci(5) should return 5.

6. **Count Digits:**
   Write a recursive function to count the number of digits in a positive integer. The function should take a positive integer as input and return the count of its digits.
   Example: countDigits(12345) should return 5.

7. **Sum of Array Elements:**
   Write a recursive function to find the sum of elements in an integer array. The function should take an array and return the sum of its elements.
   Example: sumArray([1, 2, 3, 4]) should return 10.

8. **Check Element in Array:**
   Write a recursive function to check if a given element is present in an integer array. The function should take an array and an element as input and return true if the element is found, otherwise false.
   Example: checkElement([1, 2, 3, 4], 3) should return true.

9. **Calculate Exponential:**
   Write a recursive function to calculate the exponential (e^x) of a given number x using the series expansion. The series expansion for e^x is:

   $e^x = 1 + (x/1!) + (x^2/2!) + (x^3/3!) + ... + (x^n/n!)$

   The function should take a number x and a non-negative integer n (the number of terms in the series) and return the approximation of e^x using the first n terms of the series.

   **Hint:** The recursive definition can be:
   - Base case: If n is 0, return 1.
   - Recursive case: Return (x^n / n!) + calculate the exponential of x using n-1 terms.

   Example: exponential(1, 10) should return approximately 2.71828 (using the first 10 terms of the series).


10.  **Reverse Array:**
    Write a recursive function to reverse the elements of an integer array. The function should take an array and return the array with its elements in reverse order.
    Example: reverseArray([1, 2, 3, 4]) should return [4, 3, 2, 1].
