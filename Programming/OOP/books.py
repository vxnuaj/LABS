'''Problem: Online Bookstore

Create a class Book that represents a book with attributes such as title, author, ISBN, and price. 
Include methods to get and set these attributes.

Next, create a class Bookstore that manages a collection of books. 
It should have methods to add a new book to the collection, remove a book by ISBN, and search for books by title or author.

Finally, create a class Customer that represents a customer of the bookstore. 
The customer should have attributes for their name, email, and a list of books they have purchased. 
Include methods for the customer to purchase a book (given the ISBN), return a book (given the ISBN), and view their purchased books.

Your program should demonstrate the use of these classes by creating a bookstore with some initial books, adding new books, 
allowing a customer to purchase books, and displaying the customer's purchased books.

These are the 5 books:

Title: "Deep Learning"
Author: "Ian Goodfellow, Yoshua Bengio, Aaron Courville"
ISBN: "978-0262035613"
Price: $72.00

Title: "Python Machine Learning"
Author: "Sebastian Raschka, Vahid Mirjalili"
ISBN: "978-1789955750"
Price: $49.99

Title: "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"
Author: "Aurélien Géron"
ISBN: "978-1492032649"
Price: $53.99

Title: "Pattern Recognition and Machine Learning"
Author: "Christopher M. Bishop"
ISBN: "978-0387310732"
Price: $77.24

Title: "Machine Learning Yearning"
Author: "Andrew Ng"
ISBN: "N/A"
Price: $0.00 (Free PDF download)

This problem will require you to design the class structures, implement methods for interacting with the classes, 
and create a functional program that utilizes these classes.'''

class Book:
    def __init__(self, title, author, isbn, price):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.price = price

class Bookstore:
    def __init__(self):
        self.books = [
            {'Title': 'Deep Learning', 'Author':"Ian Goodfellow, Yoshua Bengio, Aaron Courville", 'ISBN': "978-0262035613", 'Price': '$72.00'},
            {'Title': 'Python Machine Learning', 'Author':'Sebastian Raschka, Vahid Mirjalili', 'ISBN':'978-1789955750', 'Price':'$49.99'},
            {'Title': "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow", 'Author': "Aurélien Géron", 'ISBN':"978-1492032649", 'Price': '$53.99'},
            {'Title': "Pattern Recognition and Machine Learning", 'Author': "Christopher M. Bishop", 'ISBN':"978-0387310732", 'Price': '$77.24'},
            {'Title': "Machine Learning Yearning", 'Author': "Andrew Ng", 'ISBN':"N/A", 'Price': '$0.00'},
        ]

    def add_book(self, title, author, isbn, price):
        self.book = self.books.append({'Title':title, 'Author':author, 'ISBN':isbn, 'Price':price})
        return self.book

    def remove_book(self, isbn):
        for i in self.books:
            if i['ISBN'] == isbn:
                self.books.remove(i)
                return self.books
        print("Book does not exist!")

    def search_book(self, title=None, author=None):
        if title == None and author == None:
            print("Please enter a title or author!")
            return
        elif isinstance(title, str):
            for i in self.books:
                if i['Title'] == title:
                    print(f"Title: {i['Title']} \nAuthor(s): {i['Author']} \nPrice: {i['Price']}")
                    return
        elif isinstance(author, str):
            books = []
            for i in self.books:
                if i['Author'] == author:
                    books.append(i['Title'])
            print(f"Available Books by Author: {', '.join(books)}")
            return
        elif isinstance(title, str) == False or isinstance(author, str) == False:
            print("Invalid Input!")
            return
        else:
            print("Book does not exist!")
            
class Customer(Bookstore):
    def __init__(self, name, email):
        super().__init__()
        self.name = name
        self.email = email
        self.purchased_books = [ ] 

    def purchase_book(self, isbn):
        for i in self.books:
            if i['ISBN'] == isbn:
                self.purchased_books.append(i)
                return self.purchased_books
        return "Book Does not Exist"

    def view_purchased(self):
        return self.purchased_books

    def return_book(self, isbn):
        for i in self.purchased_books:
            if i['ISBN'] == isbn:
                self.purchased_books.remove(i)
                return self.purchased_books

'''

TO TEST:

Instantiate the Bookstore: Create an instance of the Bookstore class to represent your bookstore.

Add Books: Use the add_book method of the Bookstore class to add some books to your bookstore.

Instantiate a Customer: Create an instance of the Customer class to represent a customer.

Purchase Books: Use the purchase_book method of the Customer class to simulate the customer purchasing some books from the bookstore.

View Purchased Books: Use the view_purchased_books method of the Customer class to view the books the customer has purchased.

Return Books (Optional): Use the return_book method of the Customer class to simulate the customer returning some books to the bookstore.

Remove Books (Optional): Use the remove_book method of the Bookstore class to remove books from the bookstore's collection.

Search for Books (Optional): Use the search_books_by_title or search_books_by_author methods of the Bookstore class to search for books by title or author.

Test Edge Cases (Optional): Test edge cases such as trying to purchase a book that is not in the bookstore, trying to return a book that the customer has not purchased, etc.

Display Results: Print relevant information to the console to verify that the bookstore and customer objects are behaving as expected.

'''