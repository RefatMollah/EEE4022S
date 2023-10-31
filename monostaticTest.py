from signals import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc
from target import *
from radar import *
from geometry import *
from signals import *

c = 340
T = 1   #[s] Transmit time
PRI = 5 #[s] PRI time
echoTime = PRI-T 
PRF = 1/PRI #[Hz]
fc = 1
fs = 44100
B = 1000
K= 2 * np.pi * B/T
numPulses = 1
t1 = np.linspace(0,T,T*fs)
t2 = np.linspace(0,PRI*numPulses,fs*PRI*numPulses)

#adding coordinates
target1 = target(np.array([10,0,0]),np.array([0,0,0]),1)
radar = monostatic([0,0,0],1,1)


channel = pulse(fc,T,B)

Tx = channel.baseband_chirpTrain(t2,PRI,numPulses)
tx2 = channel.baseband_chirpPulse(t1,0)
#Tx3 = np.cos(2 * np.pi * fc * (t1) + 0.5 * K * (t1)**2)

#simulate response
td = 0
Rx = np.zeros(len(t2))

for i in range(numPulses):
    td = radar.calculate_delay(c, target1) + i * PRI
    print(td)
    Rx = Rx + channel.baseband_chirpPulse(t2,0)

rangeLine = matchedFiltering(tx2, Rx)
rangeMatrix = rangeLine.real.reshape(numPulses, PRI*fs)
rangeDoppler = np.fft.fftshift(rangeMatrix, axis=0)

plt.figure()

plt.plot(t2, Rx.real)
plt.title("Linear Chirp")
plt.xlabel('t (sec)')
plt.grid()

plt.show()   

plt.plot(t2, rangeLine.real)
plt.title("Linear Chirp")
plt.xlabel('t (sec)')
plt.grid()


plt.show()    


plt.figure()
plt.imshow(abs(rangeMatrix.real), cmap='plasma', aspect='auto', interpolation='none')
plt.colorbar(label='Amplitude')
plt.title('Range Map')
plt.xlabel('Time')
plt.ylabel('Range Bins')

plt.figure()
plt.imshow(abs(rangeDoppler.real), cmap='plasma', aspect='auto', interpolation='none')
plt.colorbar(label='Amplitude')
plt.title('Range Map')
plt.xlabel('Time')
plt.ylabel('Range Bins')


plt.show()



