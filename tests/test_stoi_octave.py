#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_allclose
from oct2py import octave
import pytest

from pystoi.stoi import FS, stoi

RTOL = 1e-6
ATOL = 1e-6

def test_stoi_good_fs():
    """ Test STOI at sampling frequency of 10kHz. """
    x = np.random.randn(2 * FS)
    y = np.random.randn(2 * FS)
    stoi_out = stoi(x, y, FS)
    stoi_out_m = octave.feval('octave/stoi.m', x, y, float(FS))
    assert_allclose(stoi_out, stoi_out_m, atol=ATOL, rtol=RTOL)


def test_estoi_good_fs():
    """ Test extended STOI at sampling frequency of 10kHz. """
    x = np.random.randn(2 * FS)
    y = np.random.randn(2 * FS)
    estoi_out = stoi(x, y, FS, extended=True)
    estoi_out_m = octave.feval('octave/estoi.m', x, y, float(FS))
    assert_allclose(estoi_out, estoi_out_m, atol=ATOL, rtol=RTOL)


def test_stoi_downsample():
    """ Test STOI at sampling frequency below 10 kHz. """
    for fs in [11025, 16000, 22050, 32000, 44100, 48000]:
        x = np.random.randn(2 * fs)
        y = np.random.randn(2 * fs)
        octave.eval('pkg load signal')
        stoi_out = stoi(x, y, fs)
        stoi_out_m = octave.feval('octave/stoi.m', x, y, float(fs))
        assert_allclose(stoi_out, stoi_out_m, atol=ATOL, rtol=RTOL)


def test_stoi_upsample():
    """ Test STOI at sampling frequency above 10 kHz. """
    for fs in [8000]:
        x = np.random.randn(2 * fs)
        y = np.random.randn(2 * fs)
        octave.eval('pkg load signal')
        stoi_out = stoi(x, y, fs)
        stoi_out_m = octave.feval('octave/stoi.m', x, y, float(fs))
        assert_allclose(stoi_out, stoi_out_m, atol=ATOL, rtol=RTOL)


if __name__ == '__main__':
    pytest.main([__file__])
