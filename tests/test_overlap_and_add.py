import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystoi.stoi import N_FRAME, stoi, FS
from pystoi.utils import _overlap_and_add


def test_OLA_vectorisation():
    """test the vectorised overlap_and_add comparing to the old one"""

    def old_overlap_and_app(x_frames, hop):
        num_frames, framelen = x_frames.shape
        x_sil = np.zeros((num_frames - 1) * hop + framelen)
        for i in range(num_frames):
            x_sil[range(i * hop, i * hop + framelen)] += x_frames[i, :]
        return x_sil

    batch_size = 4
    # Initialize
    x = np.random.randn(batch_size, 1000 * N_FRAME)
    # Add silence segment
    silence = np.zeros((batch_size, 10 * N_FRAME))
    x = np.concatenate([x[:, : 500 * N_FRAME], silence, x[:, 500 * N_FRAME :]], axis=1)
    x = x.reshape([batch_size, -1, N_FRAME])
    xs = [old_overlap_and_app(xi, N_FRAME // 2) for xi in x]
    xs_vectorise = _overlap_and_add(x, N_FRAME // 2)
    assert_allclose(xs, xs_vectorise)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("fs", [10000, 16000])
@pytest.mark.parametrize("extended", [True, False])
def test_pystoi_run(batch_size, fs, extended):
    N = fs * 4  # 4 seconds of random audio
    x = np.random.randn(batch_size, N)
    res = stoi(x, x, fs, extended)
    print(batch_size, fs, extended, res)
    assert res.shape == x.shape[:-1]


@pytest.mark.parametrize("extended", [True, False])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_pystoi_complete_silence(batch_size, extended):
    fs = 16000
    N = fs * 4  # 4 seconds of random audio
    x = np.zeros((batch_size, N))
    res = stoi(x, x, fs, extended)
    print(batch_size, fs, extended, res)
    assert res.shape == x.shape[:-1]


@pytest.mark.parametrize("extended", [True, False])
def test_pystoi_silence(extended):
    rng = np.random.default_rng(seed=0)
    batch_size = 4
    fs = 16000
    N = fs * 4  # 4 seconds of random audio
    x = np.random.randn(batch_size, N)
    silence = np.random.randn(int(N / 7))
    audio = []
    for i in range(batch_size):
        t = int(rng.random() * N)
        audio.append(np.concatenate([x[i, :t], silence, x[i, t:]]))
    audio = np.array(audio)
    res = stoi(audio, audio, fs, extended)
    print(batch_size, fs, extended, res)
    assert res.shape == x.shape[:-1]


def test_vectorisation():
    # Initialize batch of data
    batch_size = 4
    x = np.random.random((batch_size, 100 * N_FRAME))
    y = np.random.random((batch_size, 100 * N_FRAME))
    res = np.array([stoi(xi, yi, FS) for xi, yi in zip(x, y)])
    res_vec = stoi(x, y, FS)
    assert res_vec.shape == x.shape[:-1]
    assert np.allclose(res, res_vec)
