import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
from numba import jit


def rect(x,duration):
    """rect function implementation

    Args:
        x (_type_): _description_
        duration (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.where ((abs(x)<=duration/2),1,0)

def rect2(x):
    """rect function implementation

    Args:
        x (_type_): _description_
        duration (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.where ((abs(x)<=0.5),1,0)



def matchedFiltering(Tx,Rx):
    """_summary_

    Args:
        Tx (_type_): _description_
        Rx (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    N = Rx.size- Tx.size
    h = np.pad(Tx, (0,N), 'constant')
    h = np.conj(np.flip(h))

    H = sfft.fft(h)
    Y = sfft.fft(Rx)
    output = H * Y
    out = sfft.fftshift(sfft.ifft(output))
    return out

@jit
def matchedFilteringJIT(Tx,Rx):
    """_summary_

    Args:
        Tx (_type_): _description_
        Rx (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    N = Rx.size- Tx.size
    h = np.concatenate((Tx, np.zeros(N)))
    h = np.conj(np.flip(h))

    H = np.fft.fft(h)
    Y = np.fft.fft(Rx)
    output = H * Y
    out = np.fft.ifft(output)
    return out



    

class pulse:
    """Class to generate Linear frequency modulated chirp pulses
    """
    
    def __init__(self, fc, duration, bandwidth):
        """_summary_

        Args:
            fc (_type_): _description_
            duration (_type_): _description_
            bandwidth (_type_): _description_
        """
        self.fc = fc
        self.duration = duration
        self.bandwidth = bandwidth
        #self.samplingRate = samplingRate
    
    def baseband_chirpPulse(self,t,delay, phase):
        """_summary_

        Args:
            t (_type_): _description_
            delay (_type_): _description_

        Returns:
            _type_: _description_
        """
        fc = self.fc
        B = self.bandwidth
        T = self.duration
        
        K = 2*np.pi*1*B/T
        
        #Tx = np.exp(0.5j*K*((t - T/2 - delay)**2))* rect((t - T/2-delay)/T,T)
        Tx = np.exp(1j* (0.5* K*((t - T/2 - delay)**2) + phase) ) * rect2((t - T/2-delay)/T)
        return Tx
        
    def modulated_chirpPulse(self,t,delay, phase):
        """_summary_

        Args:
            t (_type_): _description_
            delay (_type_): _description_

        Returns:
            _type_: _description_
        """
        fc = self.fc
        B = self.bandwidth
        T = self.duration
        
        K = 2*np.pi*1*B/T
        
        #Tx = np.exp(0.5j*K*((t - T/2 - delay)**2))* rect((t - T/2-delay)/T,T)
        Tx = np.exp(1j* (0.5* K*((t - T/2 - delay)**2))) * np.exp(1j*phase) * rect2((t - T/2-delay)/T)
        return Tx
        
        
        
    def baseband_chirpTrain(self,t,PRI,numPulses):
        """_summary_

        Args:
            t (_type_): _description_
            PRI (_type_): _description_
            numPulses (_type_): _description_

        Returns:
            _type_: _description_
        """
        Tx = np.zeros(len(t))
        for i in range(numPulses):
            Tx = Tx + self.baseband_chirpPulse(t,i*PRI,0)
        return Tx
    
    
    def real_chirpPulse(self, t, delay):
        """_summary_

        Args:
            t (np array): time vector
        """
        B = self.bandwidth
        T = self.duration
        K = 2 * np.pi * B/T
        fc = self.fc
        
        Tx = np.cos(2*np.pi*(fc*(t-delay)) + 0.5*K*(t-delay)**2) * rect2(t-T/2-delay)
        return Tx
    
    def real_chirpTrain(self, t, PRI, numPulses):
        """_summary_

        Args:
            t (_type_): _description_
            PRI (_type_): _description_
            numPulses (_type_): _description_

        Returns:
            _type_: _description_
        """
        Tx = np.zeros(len(t))
        for i in range(numPulses):
            Tx = Tx + self.real_chirpPulse(t,i*PRI)
        return Tx
    