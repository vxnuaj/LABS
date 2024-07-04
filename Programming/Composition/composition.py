class Engine:
    def __init__(self, horse_power):
        self.horse_power = horse_power
       
    def __str__(self):
        return f"Horse Power: {self.horse_power}"  
        
class Wheel:
    def __init__(self, size):
        self.size = size
   
    def __str__(self):
        return f"Size - {self.size}" 
     
class Car:
    def __init__(self, make, model, horse_power, wheel_size:tuple):
        self.make = make
        self.model = model
        self.engine = Engine(horse_power)
        self.wheel = [Wheel(wheel) for wheel in wheel_size] 
        
    def __str__(self):
        wheel_str = '\n'.join(str(wheel)for wheel in self.wheel)
        return f"Make: {self.make}\nModel: {self.model}\n{self.engine}\nWheels:\n{wheel_str}"
        
         
if __name__ == "__main__":
    ferrari = Car('Ferrari', 'SF90 Stradale', 769, ('255/35r20', '255/35r20', '315/30R20', '315/30R20'))
    print(ferrari) 
    