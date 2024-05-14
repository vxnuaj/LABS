class BankAccount:
    def __init__(self, account_number, balance):
        self.account_number = account_number
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return self.balance
    
    def withdraw(self, amount):
        self.balance -= amount
        if self.balance < 0:
            print("WARNING. NEGATIVE BALANCE!")            
        return self.balance
    
    def get_balance(self):
        return self.balance
    
myaccount = BankAccount(123456, 1000)

print(myaccount.deposit(500))
print(myaccount.withdraw(200))
print(myaccount.balance)