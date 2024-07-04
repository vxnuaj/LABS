class Person:
    def __init__(self, name, age, address):
        self.name = name
        self._age = age
        self.__address = address

    def update_age(self, age):
        self._age = age

    def _change_address(self, address):
        self.__address = address

    def __display_info(self):
        print(self.name)
        print(self._age)
        print(self.__address)

    def print_info(self):
        self.__display_info()

    @property
    def name(self): return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise ValueError("Name must be type str!")
        self._name = name

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, age):
        if not isinstance(age, int):
            raise ValueError("Age must be type int!")
        self._age = age

    @property
    def address(self):
        return self.__address

    @address.setter
    def address(self, address):
        if not isinstance(address, str):
            raise ValueError("Address must be type str!")
        self._address = address

class BankAccount:
    def __init__(self, balance, account_number):
        self.balance = balance
        self.__account_number = account_number

    def deposit_funds(self, amount):
        self.balance += amount

    def withdraw_funds(self, amount):
        self.balance -= amount 

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, balance):
        if not isinstance(balance, (float, int)):
            raise ValueError("Balance must be type float or int!")

    @property
    def account_number(self):
        return self.__account_number

    @account_number.setter
    def account_number(self, account_number):
        if not isinstance(account_number, int):
            raise ValueError("Account number muyst be type int!")
        self.__account_number = account_number

class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.__model = model
        self._year = year

    def get_make_model_year(self):
        print(f"Make: {self.make}")
        print(f"Model: {self.__model}")
        print(f"Year: {self.year}")

    @property
    def make(self):
        return self._make

    @make.setter
    def make(self, make):
        if not isinstance(make, str):
            raise ValueError("Make must be type str!")
        self._make = make

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, year):
        if not isinstance(year, int):
            raise ValueError("Year must be type int!")
        self._year = year

if __name__ == "__main__":
    
    vxnuaj = Person("juan", 18, "SF")
    vxnuaj._age = 19
    print(vxnuaj.age)

    vxnuaj._change_address('hayes valley')
    print(vxnuaj.address, '\n')

    #vxnuaj.__display_info() # Private !

    # vxnuaj.print_info() # Works if used in class method!

    vxnuajAccount = BankAccount(100, 419293)
    vxnuajAccount.account_number = 100420
    print(vxnuajAccount.account_number)

    ferrari = Car('ferrari', '812 GTS', 2024)
    ferrari.get_make_model_year()
