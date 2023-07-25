import time
from matplotlib.pyplot import scatter, show

class tools:
    def __init__(self):
        self.tick = 0
        self.timelist = []
        self.ticklist = []
    def timer(self,function):
        def wrapper(*args,**kwargs):
            self.tick += 1
            start = time.time()
            rv = function(*args,**kwargs)
            print(f'duration {function.__name__}', time.time()-start)
            self.timelist.append(time.time()-start)
            self.ticklist.append(self.tick)
            return rv
        return wrapper
    def showtimegraph(self):
        scatter(self.ticklist, self.timelist)
        show()