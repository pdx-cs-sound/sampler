import numpy as np
from scipy.io import wavfile

import sampler

def wavwrite(f, s):
    wavfile.write(f, 44100, (32767 * s).astype(np.int16))

refa4 = 0.5 * np.sin(np.linspace(0, 2*np.pi*440, 44100))
wavwrite('refa4.wav', refa4)
loop = sampler.Loop(refa4)
shifta5 = loop.sample(880, 44100)
wavwrite('shifta5.wav', shifta5)
