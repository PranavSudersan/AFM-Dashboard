import numpy as np

#Lorentzian function
# y0 = white noise offset, f0 = resonance freq, w = Full width at half maximum, A = area
def lorentzian(f, y0,f0, w, A):
    return y0 + ((2*A/np.pi) * (w / ( w**2 + 4*( f - f0 )**2)))

#Amplitude function for a harmonic oscillator (frequency response)
def harmonic_amplitude(f, f0, gamma, F0, m):
    """
    Calculate the amplitude A(f) of a driven harmonic oscillator.

    Parameters:
    - f: Frequency at which to calculate the amplitude
    - f0: Resonant frequency of the oscillator
    - gamma: Width parameter related to the damping coefficient
    - F0: Amplitude of the driving force
    - m: Mass of the oscillator

    Returns:
    - A(f): Amplitude of the driven harmonic oscillator at frequency f
    """
    A = (F0/(m*2*np.pi)) / np.sqrt((f0**2 - f**2)**2 + (2 * f * gamma)**2)
    return A

#general logistic function
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)