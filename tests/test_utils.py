import pytest
import matlab.engine
import numpy as np
import scipy
from numpy.testing import assert_allclose
from pystoi.utils import thirdoct, stft, remove_silent_frames
from pystoi.stoi import FS, N_FRAME, NFFT, NUMBAND, MINFREQ, N, BETA, DYN_RANGE, OBM

ATOL = 1e-5

eng = matlab.engine.start_matlab()
eng.cd('matlab/')


def test_thirdoct():
    obm_m, cf_m = eng.thirdoct(float(FS), float(NFFT), float(NUMBAND),
                               float(MINFREQ), nargout=2)
    obm, cf = thirdoct(FS, NFFT, NUMBAND, MINFREQ)
    obm_m = np.array(obm_m)
    cf_m = np.array(cf_m).transpose().squeeze()
    assert_allclose(obm, obm_m, atol=ATOL)
    assert_allclose(cf, cf_m, atol=ATOL)


def test_stdft():
    x = np.random.randn(2*FS, )
    x_m = matlab.double(list(x))
    spec_m = eng.stdft(x_m, float(N_FRAME), float(N_FRAME/2), float(NFFT))
    spec_m = np.array(spec_m)
    spec_m = spec_m[:, 0:(NFFT/2+1)].transpose()
    spec = stft(x, N_FRAME, NFFT, overlap=2).transpose()
    assert_allclose(spec, spec_m, atol=ATOL)


def test_removesf():
    # Initialize
    x = np.random.randn(2*FS, )
    y = np.random.randn(2*FS, )
    # Add silence segment
    silence = np.zeros(3*NFFT, )
    x = np.concatenate([x[:FS], silence, x[FS:]])
    y = np.concatenate([y[:FS], silence, y[FS:]])
    x_m = matlab.double(list(x))
    y_m = matlab.double(list(y))
    xs, ys = remove_silent_frames(x, y, DYN_RANGE, N_FRAME, N_FRAME/2)
    xs_m, ys_m = eng.removeSilentFrames(x_m, y_m, float(DYN_RANGE),
                                        float(N_FRAME), float(N_FRAME/2),
                                        nargout=2)
    xs_m, ys_m = np.array(xs_m._data), np.array(ys_m._data)
    assert_allclose(xs, xs_m, atol=ATOL)
    assert_allclose(ys, ys_m, atol=ATOL)


def test_apply_OBM():
    obm_m, cf_m = eng.thirdoct(float(FS), float(NFFT), float(NUMBAND),
                               float(MINFREQ), nargout=2)
    x = np.random.randn(2*FS, )
    x_m = matlab.double(list(x))
    x_tob_m = eng.applyOBM(x_m, obm_m, float(N_FRAME), float(NFFT), float(NUMBAND))
    x_tob_m = np.array(x_tob_m)
    x_spec = stft(x, N_FRAME, NFFT, overlap=2).transpose()
    x_tob = np.zeros((NUMBAND, x_spec.shape[1]))
    for i in range(x_tob.shape[1]):
        x_tob[:, i] = np.sqrt(np.matmul(OBM, np.square(np.abs(x_spec[:, i]))))
    assert_allclose(x_tob, x_tob_m, atol=ATOL)


if __name__ == '__main__':
    pytest.main([__file__])
