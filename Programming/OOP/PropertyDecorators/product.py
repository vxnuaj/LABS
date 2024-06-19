class Product:
    def __init__(self, name:str, description:str, price: float, discount:float, is_available: bool ):
        self._name = name
        self.description = description
        self._price = price
        self._discount = discount
        self.is_availble = is_available

    @property
    def discount(self):
        return f"The discount for {self.name} is {self._discount}%!"
    
    @discount.setter
    def discount(self, discount): 
        if discount >= 0 and discount <= 100:
            self._discount = discount
        else:
            raise ValueError('Discount must be between 0 and 100!')
    
    @property
    def price(self):
        return f"The price of {self.name} is ${self._price}"
    
    @price.setter
    def price(self, price):
        if price < 0:
            raise ValueError("The price can't be a negative value!")
        elif isinstance(price, int):
            price = float(price)
        self._price = price

    @property
    def description(self):
        return self._description
    
    @description.deleter
    def description(self):
        self.description = "No description available"

    @property
    def name(self):
        return self._name
    
    @property
    def final_price(self):
        discount = self._discount / 100
        discount = self._price * discount
        return self._price - discount

    
prod = Product('shizzle', 'a shizzle', 10, 20, True)

print(prod.final_price)