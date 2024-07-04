import time

class Clock:

    def __init__(self, hours, minutes, seconds):
        self.hours = hours 
        self.minutes = minutes
        self.seconds = seconds

    def start(self, limit = None, verbose = False):
        self.running = True
        elapsed = 0
        while self.running:
            self.tick() 
            elapsed += 1 
            time.sleep(1)
            if verbose:
                print(self.get_time())
            if elapsed == limit:
                print("\nClock reached the time limit!")
                self.stop()
   
    def stop(self):
        self.running = False
    
    def tick(self):
        self.seconds += 1
        if self.seconds == 60:
            self.seconds = 0
            self.minutes += 1
        if self.minutes == 60:
            self.minutes = 0
            self.hours += 1
        if self.hours == 24:
            self.hours = 0

         
    def get_time(self):
        hour = f'0{self.hours}' if len(str(self.hours)) == 1 else str(self.hours)
        minutes = f'0{self.minutes}' if len(str(self.minutes)) == 1 else str(self.minutes)
        seconds = f'0{self.seconds}' if len(str(self.seconds)) == 1 else str(self.seconds) 
        
        return f"{hour}:{minutes}:{seconds}"

class Display:
    def __init__(self, hours, minutes, seconds):
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.clock = Clock(self.hours, self.minutes, self.seconds)

    def show(self):
        print(self.clock.get_time())  
       
    def start(self, limit = None, verbose = False):
        self.clock.start(limit, verbose)
        
    @property
    def hours(self):
        return self._hours
    
    @hours.setter
    def hours(self, hours):
        if hours > 23:
            raise ValueError("There can only be 24 completed hours in a day!")
        self._hours = hours 
        
    @property
    def minutes(self):
        return self._minutes
    
    @minutes.setter
    def minutes(self, minutes):
        if minutes > 59: raise ValueError("There can only be 60 completed mintues in an hour!")
        self._minutes = minutes 

    @property
    def seconds(self):
        return self._seconds

    @seconds.setter
    def seconds(self, seconds):
        if seconds > 59:
            raise ValueError("There can only be 60 completed seconds in a minute!")
        self._seconds = seconds
 
if __name__ == "__main__":
    
    display = Display(10, 59, 55) 
    
    display.show()
    
    display.start(10, True)
    