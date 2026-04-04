# EMA smoother for gaze coordinates

#ALPHA VALUE - change to change smoothness
ALPHA = 0.5

class EMAsmoother:
    def __init__(self, ALPHA):
        self.alpha = ALPHA
        self.last_x = None
        self.last_y = None

    def update(self, x, y):
        #initial measurement
        if self.last_x is None:
            self.last_x = x
            self.last_y = y
            return x, y
        
        self.last_x = int(self.alpha * x + (1 - self.alpha) * self.last_x)
        self.last_y = int(self.alpha * y + (1 - self.alpha) * self.last_y)

        return self.last_x, self.last_y


    def reset(self):
        # Call this if tracking is lost for a while
        # so it doesn't drag from a stale position
        self.last_x = None
        self.last_y = None