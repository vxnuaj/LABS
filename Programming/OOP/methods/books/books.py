import string

class Book:
    total_books = 0
    library_name = None
    
    def __init__(self, title:str, author:str, year:int):
        self.title = title
        self.author = author
        self.year = year
        Book.total_books += 1
        
    def display_info(self):
        print(f"Title: {self.title}")
        print(f"Author: {self.author}")
        print(f"Year Published: {self.year}")
        print(f"Library: {Book.library_name}")
        
    @classmethod
    def set_library_name(cls, name):
        if not isinstance(name, str):
            raise ValueError("name must be type str!")
        name = string.capwords(name)
        cls.library_name = name
        
    @classmethod
    def get_total_books(cls):
        return cls.total_books
    
    @staticmethod
    def is_valid_year(year):
        if not isinstance(year, int):
            raise ValueError("year must be type int!")
        elif year < 0:
            raise ValueError("year must be greater than 0!")
        return True
    
    @property
    def title(self):
        return self._title
    
    @title.setter
    def title(self, title):
        if not isinstance(title, str):
            raise ValueError("title must be type str!")
        self._title = title
    
    @property
    def author(self):
        return self._author
    
    @author.setter
    def author(self, author):
        if not isinstance(author, str):
            raise ValueError("author must be type str!")
        self._author = author
    
    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, year):
        if not isinstance(year, int):
            raise ValueError("year must be type int!")
        self._year = year

if __name__ == "__main__":

    Book.set_library_name('White Oak')

    book1 = Book('Superintelligence', 'Nick B.', 2022)

    book1.display_info()