#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Unit tests for Octave """
import numpy as np
from numpy.testing import assert_allclose
from oct2py import octave
import pytest
import scipy

from pystoi.stoi import FS, N_FRAME, NFFT
from pystoi.utils import resample_oct

ATOL = 1e-5

def test_hanning():
    """ Compare scipy and Matlab hanning window.

        Matlab returns a N+2 size window without first and last samples.
        A custom Octave function has been written to mimic this
        behavior."""
    hanning = scipy.hanning(N_FRAME+2)[1:-1]
    hanning_m = np.squeeze(octave.feval('octave/ml_hanning.m', N_FRAME))
    assert_allclose(hanning, hanning_m, atol=ATOL)


def test_fft():
    """ Compare FFT to Octave. """
    x = np.random.randn(NFFT)
    fft_m = np.squeeze(octave.fft(x))
    fft_m = fft_m[:NFFT//2+1]
    fft = np.fft.rfft(x, n=NFFT)
    assert_allclose(fft, fft_m, atol=ATOL)


def test_resample():
    """ Compare Octave and SciPy resampling.
    Both packages use polyphase resampling with a Kaiser window. We use
    the window designed by Octave in the SciPy resampler."""
    RTOL = 1e-4
    for fs in [8000, 11025, 16000, 22050, 32000, 44100, 48000]:
        x = np.random.randn(2 * fs)
        octave.eval('pkg load signal')
        x_m, h = octave.resample(x, float(FS), float(fs), nout=2)
        h = np.squeeze(h)
        x_m = np.squeeze(x_m)
        x_r = resample_oct(x, FS, fs)
        assert_allclose(x_r, x_m, atol=ATOL, rtol=RTOL)


if __name__ == '__main__':
    pytest.main([__file__])
