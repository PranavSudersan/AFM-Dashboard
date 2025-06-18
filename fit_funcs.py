import numpy as np
import scipy.special as sc

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

#analytical expression for frequency shift due to vdW interaction
#reference: Giessibl and  Bielefeldt Phys. Rev. B 61, 9968 (https://doi.org/10.1103/PhysRevB.61.9968)
def vdw_freqshift(z, C, z0, f0, k, A):
    n = 2
    d = z - z0
    f_shift = (f0*C*np.sqrt(A)/(k*(A**(3/2))*(d**n)))*(sc.hyp2f1(n, 0.5, 1, -2*A/d)-sc.hyp2f1(n, 1.5, 2, -2*A/d))
    # f_shift = (f0/(2*k))*(n*C/d**(n+1))
    return f_shift