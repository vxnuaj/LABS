# Book Management System

Building a `Book` class to manage a collection of books in a library. The `Book` class should include the following features:


### Class Variables
- A class variable `total_books` to keep track of the total number of books created.
- A class variable `library_name` to store the name of the library. This should be the same for all books.

### Instance Variables
- Instance variables `title`, `author`, and `year` to store the title, author, and publication year of each book.

### Class Methods
- A class method `set_library_name(cls, name)` to set the name of the library.
- A class method `get_total_books(cls)` to return the total number of books created.

### Static Methods
- A static method `is_valid_year(year)` to check if the year is a valid year (greater than 0).

### Instance Methods
- An instance method `display_info(self)` to print the book's title, author, year, and the library name.