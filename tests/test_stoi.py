import pytest
import matlab.engine
import numpy as np
import scipy
from numpy.testing import assert_allclose
from pystoi.stoi import stoi
from pystoi.stoi import FS, N_FRAME, NFFT, NUMBAND, MINFREQ, N, BETA, DYN_RANGE

RTOL = 1e-5
ATOL = 1e-4

eng = matlab.engine.start_matlab()
eng.cd('matlab/')


def test_stoi_good_fs():
    """
    Note : Here we don't only use absolute tolerance because the scale is unstable
    and the source of difference between the original and this STOI is the
    resampling method. So we use a relative tolerance of 0.01%
    """
    x = np.random.randn(2*FS, )
    y = np.random.randn(2*FS, )
    stoi_out = stoi(x, y, FS)
    x_m = matlab.double(list(x))
    y_m = matlab.double(list(y))
    stoi_out_m = eng.stoi(x_m, y_m, float(FS))
    assert_allclose(stoi_out, stoi_out_m, atol=ATOL, rtol=RTOL)


def test_stoi_downsample():
    """
    Note : Here we don't only use absolute tolerance because the scale is unstable
    and the source of difference between the original and this STOI is the
    resampling method. So we use a relative tolerance of 0.01%
    """
    for fs in [11025, 16000, 22050, 32000, 44100, 48000]:
        x = np.random.randn(2*fs, )
        y = np.random.randn(2*fs, )
        stoi_out = stoi(x, y, fs)
        x_m = matlab.double(list(x))
        y_m = matlab.double(list(y))
        stoi_out_m = eng.stoi(x_m, y_m, float(fs))
        assert_allclose(stoi_out, stoi_out_m, atol=ATOL, rtol=RTOL)


def test_stoi_upsample():
    """
    Note : Here we don't only use absolute tolerance because the scale is unstable
    and the source of difference between the original and this STOI is the
    resampling method. So we use a relative tolerance of 0.01%
    FAILING
    """
    for fs in [8000]:
        x = np.random.randn(2*fs, )
        y = np.random.randn(2*fs, )
        stoi_out = stoi(x, y, fs)
        x_m = matlab.double(list(x))
        y_m = matlab.double(list(y))
        stoi_out_m = eng.stoi(x_m, y_m, float(fs))
        assert_allclose(stoi_out, stoi_out_m, atol=ATOL, rtol=RTOL)


if __name__ == '__main__':
    pytest.main([__file__])
