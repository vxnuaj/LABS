class Jar:
    def __init__(self, capacity=12):
        self.capacity = capacity
        self.size = 0
        ...

    def __str__(self):
        return "🍪" * self.size
        ...

    def deposit(self, n):
        self.size += n
        return self.size
        ...

    def withdraw(self, n):
        self.size -= n
        return self.size
        ...

    @property
    def capacity(self):
        return self._capacity
    
    @capacity.setter
    def capacity(self, capacity):
        if capacity < 0:
            raise ValueError('Impossible capacity!')
        self._capacity = capacity
        return self._capacity
        ...

    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, size):
        if size < 0:
            raise ValueError("Cookies cannot be below 0!")
        elif size > self.capacity:
            raise ValueError("Too many cookies!")
        self._size = size
        return self._size
        ...