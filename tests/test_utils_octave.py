#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test utilities based on Octave"""
import numpy as np
from numpy.testing import assert_allclose
from oct2py import octave
import pytest

from pystoi.stoi import DYN_RANGE, FS, MINFREQ, N_FRAME, NFFT, NUMBAND, OBM
from pystoi.utils import remove_silent_frames, stft, thirdoct

ATOL = 1e-5


def test_thirdoct():
    """Test thirdoct by comparing to Octave"""
    obm_m, cf_m = octave.feval('octave/thirdoct.m',
                               float(FS), float(NFFT), float(NUMBAND),
                               float(MINFREQ), nout=2)
    obm, cf = thirdoct(FS, NFFT, NUMBAND, MINFREQ)
    obm_m = np.array(obm_m)
    cf_m = np.array(cf_m).transpose().squeeze()
    assert_allclose(obm, obm_m, atol=ATOL)
    assert_allclose(cf, cf_m, atol=ATOL)


def test_stdft():
    """Test stdft by comparing to Octave"""
    x = np.random.randn(2 * FS)
    spec_m = octave.feval('octave/stdft.m',
                          x, float(N_FRAME), float(N_FRAME/2), float(NFFT))
    spec_m = spec_m[:, 0:(NFFT // 2 + 1)].transpose()
    spec = stft(x, N_FRAME, NFFT, overlap=2).transpose()
    assert_allclose(spec, spec_m, atol=ATOL)


def test_removesf():
    """Test remove_silent_frames by comparing to Octave"""
    # Initialize
    x = np.random.randn(2 * FS)
    y = np.random.randn(2 * FS)
    # Add silence segment
    silence = np.zeros(3 * NFFT, )
    x = np.concatenate([x[:FS], silence, x[FS:]])
    y = np.concatenate([y[:FS], silence, y[FS:]])
    xs, ys = remove_silent_frames(x, y, DYN_RANGE, N_FRAME, N_FRAME // 2)
    xs_m, ys_m = octave.feval('octave/removeSilentFrames.m',
                              x, y, float(DYN_RANGE),
                              float(N_FRAME),
                              float(N_FRAME / 2),
                              nout=2)
    xs_m = np.squeeze(xs_m)
    ys_m = np.squeeze(ys_m)
    assert_allclose(xs, xs_m, atol=ATOL)
    assert_allclose(ys, ys_m, atol=ATOL)


def test_apply_OBM():
    """Test apply_OBM by comparing to Octave"""
    obm_m, _ = octave.feval('octave/thirdoct.m',
                            float(FS), float(NFFT), float(NUMBAND),
                            float(MINFREQ), nout=2)
    x = np.random.randn(2 * FS)
    x_tob_m = octave.feval('octave/applyOBM',
                           x, obm_m, float(N_FRAME), float(NFFT),
                           float(NUMBAND))
    x_tob_m = np.array(x_tob_m)
    x_spec = stft(x, N_FRAME, NFFT, overlap=2).transpose()
    x_tob = np.zeros((NUMBAND, x_spec.shape[1]))
    for i in range(x_tob.shape[1]):
        x_tob[:, i] = np.sqrt(np.matmul(OBM, np.square(np.abs(x_spec[:, i]))))
    assert_allclose(x_tob, x_tob_m, atol=ATOL)


if __name__ == '__main__':
    pytest.main([__file__])
