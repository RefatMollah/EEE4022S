from signals import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc
import math
from target import *
from radar import *
from geometry import *
from signals import *
from scipy.constants import speed_of_light
from scipy.signal import find_peaks
from scipy import signal


plt.rcParams.update({
    "font.family": "serif",      # Use serif/main font for text elements
    "text.usetex": True,          # Use inline math for ticks
    "pgf.rcfonts": False,         # Don't setup fonts from rc parameters
    "axes.labelsize": 16,         # Adjust label size
    "axes.titlesize": 18,        # Adjust title size
    "xtick.labelsize": 14,       # Adjust x-axis tick label size
    "ytick.labelsize": 14,       # Adjust y-axis tick label size
    "legend.fontsize": 14,       # Adjust legend font size
    "axes.grid": True            # Enable grid
})


c = speed_of_light  # Speed of light in m/s
T = 10e-5  # Pulse width in seconds
PRI = 1e-3  # PRI in seconds (pulse repetition interval for 10 kHz PRF)
echoTime = PRI - T
PRF = 1/PRI  # Pulse repetition frequency in Hz
fc = 3e9  # Carrier frequency in Hz
fs = 1e6  # Sampling frequency in Hz
B = 100e3  # Sweep bandwidth in Hz
K = B / T  # Chirp rate (slope of the linear FM waveform)
numPulses = 1
t1 = np.arange(0, T, 1/fs)
t2 = np.arange(0, PRI*numPulses, 1/fs)

t2 = t2[:int(PRI*numPulses*fs)]
t1 = t1[:int(T*fs)]

print(PRI)
print(len(t2))
# Adding coordinates
target1 = target(np.array([5000, 5000, 0]), np.array([0, 0, 0]), 3)

targets = []
targets.append(target1)

radar = monostatic([0, 0, 0], 1, 1000)

# Create linear FM waveform
channel = pulse(fc, T, B)

Tx = channel.baseband_chirpTrain(t2, PRI, numPulses) 
Tx =  np.sqrt(radar.peakPower) * Tx
tx2 = channel.baseband_chirpPulse(t1, 0)

# Simulate response
td = 0
Rx = np.zeros(len(t2))

for i in range(numPulses):
    for tgt in targets:
        #td = radar.calculate_delay(c, tgt) + i * PRI
        td = bistatic_delay(c, radar, radar, tgt) + i * PRI
        rd = calculate_TargetRange(radar, tgt)
        receivedPower =  calculate_RadarRange(radar.peakPower, 1, 1, c/fc, target1.rcs, rd, rd)
        Rx = Rx + channel.baseband_chirpPulse(t2, td)
        Rx = Rx * np.sqrt(receivedPower)
        tgt.updatePosition(PRI)

rangeLine = matchedFiltering(tx2, Rx)
peak_index = np.argmax(rangeLine)
print("delay time is:")
print(peak_index/fs)
rangeMatrix = rangeLine.real.reshape(numPulses, math.ceil(PRI*fs))
rangeDoppler = np.fft.fft(rangeMatrix, axis=0)

print(c)

plt.figure()

plt.plot(t2, Tx.real)
#plt.title("Linear Chirp")
plt.ylabel('Amplitude')
plt.xlabel('t (sec)')
plt.grid(True)

plt.show()


plt.figure()

plt.plot(t2, Rx.real)
plt.ylabel('Amplitude')
plt.xlabel('t (sec)')
plt.grid(True)

plt.show()   


plt.plot(t2, abs(rangeLine))
plt.ylabel('Amplitude')
plt.xlabel('t (sec)')
plt.grid(True)



plt.show()   

plt.figure()
plt.imshow(abs(rangeMatrix), cmap='plasma', aspect='auto', interpolation='none')
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

