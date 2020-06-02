#!/usr/bin/python3
# Copyright (c) 2019 Bart Massey
# [This program is licensed under the "MIT License"]
# Please see the file LICENSE in the source
# distribution of this software for license terms.

import numpy as np
import numpy.fft as fft

class Wave(object):
    """Wavetable VCO"""
    def __init__(self, wavetable, f0, f):
        self.step = f / f0
        self.wavetable = wavetable
        self.nwavetable = len(wavetable)

    def sample(self, t, tv = None):
        """Return the next sample from this generator."""
        assert tv is None
        # XXX Should antialias
        t0 = (self.step * t) % self.nwavetable
        i = int(t0)
        frac = t0 % 1.0
        x0 = self.wavetable[i]
        x1 = self.wavetable[(i + 1) % self.nwavetable]
        return x0 * frac + x1 * (1.0 - frac)

class Loop(object):
    """Wavetable VCO factory."""
    def __init__(self, psignal, rate=44100):
        """Make a new wave generator generator."""
        # Find fundamental frequency with DFT.
        npsignal = len(psignal)
        ndft = 2
        minsamples = int(0.1 * rate)
        if npsignal < minsamples:
            raise Exception("sample too short")
        while ndft < 16 * 1024 and 2 * ndft <= npsignal:
            ndft *= 2
        if npsignal > ndft:
            start = (npsignal - ndft) // 2
            fsignal = psignal[start:start + ndft]
        else:
            fsignal = psignal
        assert len(fsignal) == ndft
        window = np.blackman(ndft)
        dft = fft.fft(fsignal * window)
        maxbin = np.argmax(np.abs(dft))
        maxf = abs(fft.fftfreq(ndft, 1 / rate)[maxbin])
        if maxf < 100 or maxf > 4000:
            raise Exception(f"sample frequency {maxf} out of range")
        # XXX Should auto-truncate the signal to sustain (hard).
        # Loop the sample properly (if sufficient samples).
        # Heuristic considers first 8 vs last 16 periods' worth of
        # samples.
        p = rate / maxf
        w = int(p * 8)
        if npsignal >= 2 * w:
            ssignal = psignal[-2 * w:]
            tsignal = psignal[:w]
            corrs = np.correlate(ssignal, tsignal, mode='valid')
            maxc = np.argmax(corrs)
            trunc = w - maxc
            psignal = psignal[:-trunc]
            # Smooth the transition over a period.
            t = int(p)
            sweight = np.linspace(0, 1, t)
            ssignal = psignal[:t]
            tweight = np.linspace(1, 0, t)
            tsignal = psignal[-t:]
            smoothed = sweight * ssignal + tweight * tsignal
            psignal = np.append(psignal[:-t], smoothed)
            # write_wave("trunc.wav", psignal)
        # Save analysis results for Wave creation.
        self.f0 = maxf
        self.wavetable = psignal

    def sample(self, f, nsamples):
        wave = self(f)
        return np.array([wave.sample(t) for t in range(nsamples)])

    def __call__(self, f):
        return Wave(self.wavetable, self.f0, f)
