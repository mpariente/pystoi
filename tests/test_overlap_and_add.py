import numpy as np
from numpy.testing import assert_allclose

from pystoi.stoi import N_FRAME
from pystoi.utils import _overlap_and_add


def test_OLA_vectorisation():
    """test the vectorised overlap_and_add comparing to the old one"""

    def old_overlap_and_app(x_frames, hop):
        num_frames, framelen = x_frames.shape
        x_sil = np.zeros((num_frames - 1) * hop + framelen)
        for i in range(num_frames):
            x_sil[range(i * hop, i * hop + framelen)] += x_frames[i, :]
        return x_sil

    # Initialize
    x = np.random.randn(1000 * N_FRAME)
    # Add silence segment
    silence = np.zeros(10 * N_FRAME)
    x = np.concatenate([x[: 500 * N_FRAME], silence, x[500 * N_FRAME :]])
    x = x.reshape([-1, N_FRAME])
    xs = old_overlap_and_app(x, N_FRAME // 2)
    xs_vectorise = _overlap_and_add(x, N_FRAME // 2)
    assert_allclose(xs, xs_vectorise)
