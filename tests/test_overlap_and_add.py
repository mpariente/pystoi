import numpy as np
from numpy.testing import assert_allclose

from pystoi.stoi import N_FRAME
from pystoi.utils import _overlap_and_add


def test_OLA_vectorisation():
    """test the vectorised overlap_and_add comparing to the old one"""

    def old_overlap_and_app(x_frames, framelen, hop):
        # init zero arrays to hold x, y with silent frames removed
        n_sil = (len(x_frames) - 1) * hop + framelen
        x_sil = np.zeros(n_sil)
        for i in range(x_frames.shape[0]):
            x_sil[range(i * hop, i * hop + framelen)] += x_frames[i, :]
        return x_sil

    # Initialize
    x = np.random.randn(1000 * N_FRAME)
    # Add silence segment
    silence = np.zeros(10 * N_FRAME)
    x = np.concatenate([x[: 500 * N_FRAME], silence, x[500 * N_FRAME :]])
    x = x.reshape([-1, N_FRAME])
    xs = old_overlap_and_app(x, N_FRAME, N_FRAME // 2)
    xs_vectorise = _overlap_and_add(x, N_FRAME, N_FRAME // 2)
    assert_allclose(xs, xs_vectorise)
