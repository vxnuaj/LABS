class BankAccount:
    def __init__(self, account_holder, account_number, balance):
        self.account_holder = account_holder
        self.account_number = account_number
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if self.balance >= amount:
            self.balance -= amount
        else:
            raise ValueError

    @staticmethod
    def is_valid_account_number(account_number: str):
        if len(account_number) == 10:
            return True
        else:
            return False
        
    @classmethod
    def from_account_details(cls, account_holder, account_number, balance):
        validity = cls.is_valid_account_number(account_number)
        if validity == False:
            raise ValueError
        else:
            return cls(account_holder, account_number, balance)
        
account = BankAccount('Juan', 1004201234, 10)

print(f"Initial Account:\n")
print(account.account_holder)
print(account.account_number)
print(account.balance)

account.deposit(10)

print(f"\nNew Deposit Balance: {account.balance}")

account.withdraw(5)

print(f"\nNew Withdrawal balance: {account.balance}")

# account.withdraw(100) ''' raised value error '''

print(f"\nValid account number? \n{BankAccount.is_valid_account_number('1234567890')}")
print(f"\nValid account number? \n{BankAccount.is_valid_account_number('12345678901')}")