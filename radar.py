import numpy as np
from geometry import *


class Transmitter:
    """_summary_
    """
    def __init__(self,coordinates,gain):
        self.coordinates = coordinates
        self.gain = gain
    
class Receiver:
    """_summary_
    """
    def __init__(self,coordinates,gain):
        self.coordinates = coordinates
        self.gain = gain

class monostatic:
    """_summary_
    """
    
    def __init__(self, coordinates, gain, peakPower):
        self.coordinates = coordinates
        self.gain = gain
        self.peakPower = peakPower

    def calculate_delay(self, c, target):
        distance = target.coordinates - self.coordinates
        r = np.linalg.norm(distance)
        print(r)
        return 2 * r / c
        
        