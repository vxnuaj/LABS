'''

Create a Python class representing a bank account with the following features:

- Class variable interest_rate that stores the annual interest rate for all accounts.

- Instance variables name and balance to store the account holder's name and balance, respectively.

- Class method set_interest_rate(cls, rate) to set the annual interest rate for all accounts.

- Instance method deposit(self, amount) to deposit a specified amount into the account.

- Instance method withdraw(self, amount) to withdraw a specified amount from the account, ensuring the balance does not go below zero.

- Instance method add_interest(self) to add interest to the account balance based on the annual interest rate.

'''


class BankAccount:
    interest_rate = 0.02

    def __init__(self, name, balance=0):
        self.name = name
        self.balance = balance
        pass

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        self.balance -= amount

    def add_interest(self):
        interest = self.balance * BankAccount.interest_rate 
        self.balance += interest


    @classmethod
    def set_interest_rate(cls, rate):
        cls.interest_rate = rate




account1 = BankAccount("Alice")
account1.deposit(1000)
account1.add_interest()
print(account1.balance)

'''BankAccount.set_interest_rate(0.03)
account2 = BankAccount("Bob", 500)
account2.add_interest()
print(account2.balance)'''