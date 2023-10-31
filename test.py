from signals import *
import matplotlib.pyplot as plt
import numpy as np
from target import *
from radar import *
from geometry import *

#######################################################################################################################
c = 340
T = 1   #[s] Transmit time
PRI = 2 #[s] PRI time
echoTime = PRI-T 
PRF = 1/PRI #[Hz]
fc = 1
fs = 4410
B = 1000
numPulses = 32

Lambda = c/fc
Vel_target = -0.6e-2
Fd = 2*Vel_target/Lambda

t1 = np.linspace(0,T,T*fs)  #
t2 = np.linspace(0,PRI*numPulses,fs*PRI*numPulses)

#t1 = np.arange(start=0, step=T*fs,)
#t2 = np.arange()

#adding coordinates 
target1 = target(np.array([100,40,0]),np.array([-0.006,0,0]),1)
print(target1.coordinates.shape)
transmitter1 = Transmitter([0,0,0],1)
receiver1 = Receiver([300,0,0],1)
######################################################################################################################


signal1 = pulse(fc,T,B)

Tx = signal1.baseband_chirpPulse(t1,0,0)
Tx2 = signal1.baseband_chirpTrain(t2,PRI,numPulses)

#simulate response
td = 0
Rx = np.zeros(len(t2))

for i in range(numPulses):
    td = bistatic_delay(c,receiver1,transmitter1,target1) + i * PRI
    fd = calculate_DopplerShift(transmitter1.coordinates, receiver1.coordinates, c=c, fc=fc, Target=target1)
    print(fd)
    phaseShift = 2*np.pi*Fd*PRI*i
    target1.updatePosition(PRI)
    Rx = Rx + signal1.baseband_chirpPulse(t2,td,90)
    td = td + PRI

#Pulse compression
out = matchedFiltering(Tx,Rx)

#range lines
rxMatrix = np.reshape( out ,(numPulses ,fs*PRI))
#endIndex = int((fs*PRI/2)-1)
#outdoppler = np.fft.fft2(rxMatrix[:endIndex,:])

#dopplerShifted = np.fft.fftshift(outdoppler)

dopplerRange = np.fft.fftshift(np.fft.fft(rxMatrix,axis=0))

print(rxMatrix.shape)

plt.figure()

plt.plot(t1, Tx)
plt.title("Linear Chirp")
plt.xlabel('t (sec)')
plt.grid()

plt.figure()

plt.plot(t2, out)
plt.title("Linear Chirp")
plt.xlabel('t (sec)')
plt.grid()


plt.show()

plt.figure()
plt.imshow(abs(rxMatrix.real), cmap='viridis', aspect='auto')
plt.colorbar(label='Amplitude')
plt.title('Range Map')
plt.xlabel('Time')
plt.ylabel('Range Bins')

plt.figure()
plt.imshow(abs(dopplerRange.real), cmap='viridis', aspect='auto')
plt.colorbar(label='Amplitude')
plt.title('Range Map')
plt.xlabel('Time')
plt.ylabel('Range Bins')

plt.show()