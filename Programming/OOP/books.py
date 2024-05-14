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
    def __init__(self, bookstore, name, email):
        self.bookstore = bookstore
        self.name = name
        self.email = email
        self.purchased_books = []

    def purchase_book(self, isbn):
        super().__init__()
        for i in self.bookstore.books:
            if i['ISBN'] == isbn:
                self.purchased_books.append(i)
                return self.purchased_books
        print("Book does not exist!")

    def view_purchased(self):
        print(f"Hi {self.name}!\n\nYou previously purchased:\n")
        for i in self.purchased_books:
            print(f"{i['Title']} by {i['Author']}")

    def return_book(self, isbn):
        for i in self.purchased_books:
            if i['ISBN'] == isbn:
                self.purchased_books.remove(i)
                return self.purchased_books

if __name__ == "__main__":
    bookstore = Bookstore()

    bookstore.add_book('The Beginning of Infinity', 'David Deutsch', '123', '$10')
    bookstore.add_book('The Fabric of Reality', 'David Deutsch', '134', '$10')

    customer = Customer(bookstore, 'vxnuaj', 'vxnuaj@gmail.com')

    customer.purchase_book('154')