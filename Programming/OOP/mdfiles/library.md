### Problem Set: Library Management System

Create a Python system to manage books and patrons in a library. Your system should include:

1. A class for managing books.
2. A class for managing patrons.
3. A class for the library that ties books and patrons together.

### Requirements:

#### Book Class:
1. Create a `Book` class with the following attributes:
    - `title`: The title of the book.
    - `author`: The author of the book.
    - `isbn`: The ISBN of the book.
    - `available`: A boolean indicating if the book is available or not.
    
2. Implement methods for the `Book` class:
    - `borrow(self)`: Marks the book as borrowed.
    - `return_book(self)`: Marks the book as returned.
    - `display_info(self)`: Displays the book's information.

#### Patron Class:
1. Create a `Patron` class with the following attributes:
    - `name`: The name of the patron.
    - `patron_id`: The unique ID for the patron.
    - `borrowed_books`: A list to store the ISBNs of borrowed books.

2. Implement methods for the `Patron` class:
    - `borrow_book(self, isbn)`: Adds the book's ISBN to the list of borrowed books.
    - `return_book(self, isbn)`: Removes the book's ISBN from the list of borrowed books.
    - `display_info(self)`: Displays the patron's information and the list of borrowed books.

#### Library Class:
1. Create a `Library` class with the following attributes:
    - `catalog`: A dictionary to store books, where the key is the ISBN and the value is a `Book` object.
    - `patrons`: A dictionary to store patrons, where the key is the patron's ID and the value is a `Patron` object.

2. Implement methods for the `Library` class:
    - `add_book(self, book)`: Adds a book to the catalog.
    - `remove_book(self, isbn)`: Removes a book from the catalog by its ISBN.
    - `add_patron(self, patron)`: Adds a patron to the library.
    - `remove_patron(self, patron_id)`: Removes a patron from the library by their ID.
    - `borrow_book(self, patron_id, isbn)`: Allows a patron to borrow a book if it is available.
    - `return_book(self, patron_id, isbn)`: Allows a patron to return a borrowed book.
    - `display_catalog(self)`: Displays all the books in the catalog.
    - `display_patrons(self)`: Displays all the patrons and their borrowed books.

### Additional Requirements:

- Implement error handling to ensure that books cannot be borrowed if they are not available, and that patrons cannot borrow books if they are not registered.
- Ensure that books are marked as available/unavailable correctly when borrowed or returned.
