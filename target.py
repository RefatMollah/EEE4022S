import numpy as np

class target:
    
    def __init__(self, coordinates, velocity, rcs):
        self.coordinates = coordinates
        self.velocity = velocity
        self.rcs = rcs
    
    def updatePosition(self, timeStep):
        self.coordinates = np.add(self.coordinates,self.velocity*timeStep)