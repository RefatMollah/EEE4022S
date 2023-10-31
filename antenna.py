import skrf as rf
import matplotlib.pyplot as plt

# Create a Network object representing the transmission line (e.g., coaxial cable)
tl = rf.Network('example.s2p')

# Define the antenna properties
frequency = rf.Frequency(start=1, stop=10, npoints=101, unit='GHz')
antenna = rf.Antenna(frequency=frequency, z0=50)

# Set up the transmission line and antenna network
network = tl ** antenna

# Perform the simulation
result = network.s_parameters

# Plot the results
result.plot_s_db()

# Show the plot
plt.show()

