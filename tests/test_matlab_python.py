import pytest
import matlab.engine
import numpy as np
import scipy
from numpy.testing import assert_allclose
from pystoi.stoi import N_FRAME, NFFT, FS

ATOL = 1e-5

eng = matlab.engine.start_matlab()
eng.cd('matlab/')


def test_hanning():
    """ Compare scipy and Matlab hanning window.
        Matlab returns a N+2 size window without first and last samples"""
    hanning = scipy.hanning(N_FRAME+2)[1:-1]
    hanning_m = eng.hanning(float(N_FRAME))
    hanning_m = np.array(hanning_m._data)
    assert_allclose(hanning, hanning_m, atol=ATOL)


def test_fft():
    x = np.random.randn(N_FRAME, )
    x_m = matlab.double(list(x))
    fft_m = eng.fft(x_m, NFFT)
    fft_m = np.array(fft_m).transpose()
    fft_m = fft_m[0:NFFT//2+1, 0]
    fft = np.fft.rfft(x, n=NFFT)
    assert_allclose(fft, fft_m, atol=ATOL)


def test_resampy():
    """ Compare matlab and librosa resample : FAILING """
    from resampy import resample
    from pystoi.stoi import FS
    import matlab_wrapper
    matlab = matlab_wrapper.MatlabSession()
    matlab.put('FS', float(FS))
    RTOL = 1e-4

    for fs in [8000, 11025, 16000, 22050, 32000, 44100, 48000]:
        x = np.random.randn(2*fs,)
        x_r = resample(x, fs, FS)
        matlab.put('x', x)
        matlab.put('fs', float(fs))
        matlab.eval('x_r = resample(x, FS, fs)')
        assert_allclose(x_r, matlab.get('x_r'), atol=ATOL, rtol=RTOL)


def test_nnresample():
    """ Compare matlab and nnresample resample : FAILING """
    from nnresample import resample
    from pystoi.stoi import FS
    import matlab_wrapper
    matlab = matlab_wrapper.MatlabSession()
    matlab.put('FS', float(FS))
    RTOL = 1e-4

    for fs in [8000, 11025, 16000, 22050, 32000, 44100, 48000]:
        x = np.random.randn(2*fs,)
        x_r = resample(x, FS, fs)
        matlab.put('x', x)
        matlab.put('fs', float(fs))
        matlab.eval('x_r = resample(x, FS, fs)')
        assert_allclose(x_r, matlab.get('x_r'), atol=ATOL, rtol=RTOL)


if __name__ == '__main__':
    pytest.main([__file__])
