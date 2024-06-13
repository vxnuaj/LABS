import sys
from utils import Utils as ut

class Book:
    def __init__(self, title:str, author:str, isbn:int):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.libraries = []
        self.available = 'Not Available'

    def _borrow(self):
        self.available = 'Not Available'
        return
        
    def return_book(self):
        self.available = 'Available'

    def add_to_library(self, library_name):
        self.libraries.append(library_name)
        self.available = 'Available'
        
    def display_info(self):
        print(f"Title: {self.title}")
        print(f"Author: {self.author}")
        print(f"ISBN: {self.isbn}")
        print(f"Availability: {self.available}")
        print(f"Available @: {ut.list_to_string(self.libraries)}\n")


class Patron:
    def __init__(self, name:str, patron_id:int, ):
        self.name = name.title()
        self.patron_id = patron_id
        self.borrowed_books = {}

    def borrow_book(self, book:Book):
        self.borrowed_books[book.isbn] = book

    def return_book(self, isbn):
        self.borrowed_books.pop(isbn)

    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Patron ID: {self.patron_id}")
        print(f"Borrowed Books: {self.borrowed_books}\n")
    

class Library:
    def __init__(self, library_name:str):
        self.library_name = library_name
        self.catalog = {}
        self.patrons = {}

    def add_book(self, book: Book):
        if isinstance(book, Book):
            self.catalog[book.isbn] = book
            book.add_to_library(self.library_name)
        else:
            sys.exit("Invalid book!")

    def remove_book(self, isbn:int):
        if isinstance(isbn,  int):
            try:
                self.catalog.pop(isbn)
            except KeyError:
                sys.exit('Book not in Library!')
        else:
            sys.exit("Invalid ISBN!")

    def add_patron(self, patron: Patron):
        if isinstance(patron, Patron):
            self.patrons[patron.patron_id] = patron
        else:
            sys.exit("Invalid patron!")

    def remove_patron(self, patron_id:int):
        if isinstance(patron_id, int):
            try:
                self.patrons.pop(patron_id)
            except KeyError:
                raise KeyError("Patron not in library!")

    def borrow_book(self, patron_id:int, isbn:int):
        if patron_id in self.patrons:
            patron = self.patrons[patron_id]
            if isbn in self.catalog:
                book = self.catalog[isbn] # Gets the specific book instance
                if book.available == 'Available':
                    book.available = 'Not Available'
                    patron.borrow_book(book) # Adds book class to Patron's borrowed book list
                    print(f"{patron.name} successfuly borrowed {book.title}!")
                else:
                    print(f"{patron.name}! {book.title} is already borrowed!")
            else:
                raise sys.exit(f"{patron.name}! {book.isbn} is an invalid ISBN!")
        else:
            sys.exit("Invalid Patron ID!")

    def return_book(self, patron_id:int, isbn:int):
        if patron_id in self.patrons:
            patron = self.patrons[patron_id]
            if isbn in patron.borrowed_books:
                book = patron.borrowed_books[isbn]
                book.return_book()
                patron.return_book(isbn)
                print(f"{patron.name}! You succesfully returned {book.title}!")
            else:
                raise ValueError("Invalid ISBN!")
        else:
            raise KeyError("Invalid Patron ID!")
        
    def display_catalog(self):
        for book in self.catalog:
            book_l = self.catalog[book]
            print(f"Title: {book_l.title}")
            print(f"Author: {book_l.author}")
            print(f"ISBN: {book_l.isbn}")
            print(f"Availability: {book_l.available}\n")
    
    def display_patron(self):
        for patron in self.patrons:
            patron_l = self.patrons[patron]
            print(f"Patron: {patron_l.name}")
            print(f"Patron ID: {patron_l.patron_id}\n")
            for book in patron_l.borrowed_books:
                print(f"{book.title},", end = None)

if __name__ == "__main__":

    Book1 = Book(title = 'The Little Book of Deep Learning', author = 'François Fleuret', isbn = 9789732346495 )
    Book2 = Book(title = "Daniels' RUNNING Formula", author = "Jack Daniels", isbn = 9781718203662)

    ''' Book1.display_info()
    Book2.display_info()'''

    Patron1 = Patron('Juan', 419293)
    Patron2 = Patron('Xeno', 123456)

    ''' Patron1.display_info()
    Patron2.display_info()'''

    Library1 = Library('MC')
    Library1.add_book(Book1)
    Library1.add_book(Book2)

    Library1.add_patron(Patron1)
    Library1.add_patron(Patron2)
    Library1.display_patron()
    Library1.display_catalog()

    Library1.borrow_book(419293, 9789732346495)
    Library1.borrow_book(419293, 9781718203662)

    Library1.borrow_book(123456, 9789732346495)
    Library1.return_book(419293, 9789732346495)

    Library1.borrow_book(123456, 9789732346495)



