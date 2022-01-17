import functools

import numpy as np
from scipy.signal import resample_poly

EPS = np.finfo("float").eps


def _resample_window_oct(p, q):
    """Port of Octave code to Python"""

    gcd = np.gcd(p, q)
    if gcd > 1:
        p /= gcd
        q /= gcd

    # Properties of the antialiasing filter
    log10_rejection = -3.0
    stopband_cutoff_f = 1.0 / (2 * max(p, q))
    roll_off_width = stopband_cutoff_f / 10

    # Determine filter length
    rejection_dB = -20 * log10_rejection
    L = np.ceil((rejection_dB - 8) / (28.714 * roll_off_width))

    # Ideal sinc filter
    t = np.arange(-L, L + 1)
    ideal_filter = 2 * p * stopband_cutoff_f \
        * np.sinc(2 * stopband_cutoff_f * t)

    # Determine parameter of Kaiser window
    if (rejection_dB >= 21) and (rejection_dB <= 50):
        beta = 0.5842 * (rejection_dB - 21)**0.4 \
            + 0.07886 * (rejection_dB - 21)
    elif rejection_dB > 50:
        beta = 0.1102 * (rejection_dB - 8.7)
    else:
        beta = 0.0

    # Apodize ideal filter response
    h = np.kaiser(2 * L + 1, beta) * ideal_filter

    return h


def resample_oct(x, p, q):
    """Resampler that is compatible with Octave"""
    h = _resample_window_oct(p, q)
    window = h / np.sum(h)
    return resample_poly(x, p, q, window=window)


@functools.lru_cache(maxsize=None)
def thirdoct(fs, nfft, num_bands, min_freq):
    """ Returns the 1/3 octave band matrix and its center frequencies
    # Arguments :
        fs : sampling rate
        nfft : FFT size
        num_bands : number of 1/3 octave bands
        min_freq : center frequency of the lowest 1/3 octave band
    # Returns :
        obm : Octave Band Matrix
        cf : center frequencies
    """
    f = np.linspace(0, fs, nfft + 1)
    f = f[: int(nfft / 2) + 1]
    k = np.array(range(num_bands)).astype(float)
    cf = np.power(2.0 ** (1.0 / 3), k) * min_freq
    freq_low = min_freq * np.power(2.0, (2 * k - 1) / 6)
    freq_high = min_freq * np.power(2.0, (2 * k + 1) / 6)
    obm = np.zeros((num_bands, len(f)))  # a verifier

    for i in range(len(cf)):
        # Match 1/3 oct band freq with fft frequency bin
        f_bin = np.argmin(np.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = np.argmin(np.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        # Assign to the octave band matrix
        obm[i, fl_ii:fh_ii] = 1
    return obm, cf


def stft(x, win_size, fft_size, overlap=4):
    """ Short-time Fourier transform for real 1-D inputs
    # Arguments
        x : 1D array, the waveform
        win_size : integer, the size of the window and the signal frames
        fft_size : integer, the size of the fft in samples (zero-padding or not)
        overlap: integer, number of steps to make in fftsize
    # Returns
        stft_out : 2D complex array, the STFT of x.
    """
    hop = int(win_size / overlap)
    w = np.hanning(win_size + 2)[1: -1]  # = matlab.hanning(win_size)
    stft_out = np.array([np.fft.rfft(w * x[i:i + win_size], n=fft_size)
                        for i in range(0, len(x) - win_size, hop)])
    return stft_out


def _overlap_and_add(x_frames, hop):
    num_frames, framelen = x_frames.shape
    # Compute the number of segments, per frame.
    segments = -(-framelen // hop)  # Divide and round up.

    # Pad the framelen dimension to segments * hop and add n=segments frames
    signal = np.pad(x_frames, ((0, segments), (0, segments * hop - framelen)))

    # Reshape to a 3D tensor, splitting the framelen dimension in two
    signal = signal.reshape((num_frames + segments, segments, hop))
    # Transpose dimensions so that signal.shape = (segments, frame+segments, hop)
    signal = np.transpose(signal, [1, 0, 2])
    # Reshape so that signal.shape = (segments * (frame+segments), hop)
    signal = signal.reshape((-1, hop))

    # Now behold the magic!! Remove the last n=segments elements from the first axis
    signal = signal[:-segments]
    # Reshape to (segments, frame+segments-1, hop)
    signal = signal.reshape((segments, num_frames + segments - 1, hop))
    # This has introduced a shift by one in all rows

    # Now, reduce over the columns and flatten the array to achieve the result
    signal = np.sum(signal, axis=0)
    end = (len(x_frames) - 1) * hop + framelen
    signal = signal.reshape(-1)[:end]
    return signal


def remove_silent_frames(x, y, dyn_range, framelen, hop):
    """ Remove silent frames of x and y based on x
    A frame is excluded if its energy is lower than max(energy) - dyn_range
    The frame exclusion is based solely on x, the clean speech signal
    # Arguments :
        x : array, original speech wav file
        y : array, denoised speech wav file
        dyn_range : Energy range to determine which frame is silent
        framelen : Window size for energy evaluation
        hop : Hop size for energy evaluation
    # Returns :
        x without the silent frames
        y without the silent frames (aligned to x)
    """
    # Compute Mask
    w = np.hanning(framelen + 2)[1:-1]

    x_frames = np.array(
        [w * x[i:i + framelen] for i in range(0, len(x) - framelen, hop)])
    y_frames = np.array(
        [w * y[i:i + framelen] for i in range(0, len(x) - framelen, hop)])

    # Compute energies in dB
    x_energies = 20 * np.log10(np.linalg.norm(x_frames, axis=1) + EPS)

    # Find boolean mask of energies lower than dynamic_range dB
    # with respect to maximum clean speech energy frame
    mask = (np.max(x_energies) - dyn_range - x_energies) < 0

    # Remove silent frames by masking
    x_frames = x_frames[mask]
    y_frames = y_frames[mask]

    x_sil = _overlap_and_add(x_frames, hop)
    y_sil = _overlap_and_add(y_frames, hop)

    return x_sil, y_sil


def vect_two_norm(x, axis=-1):
    """ Returns an array of vectors of norms of the rows of matrices from 3D array """
    return np.sum(np.square(x), axis=axis, keepdims=True)


def row_col_normalize(x):
    """ Row and column mean and variance normalize an array of 2D segments """
    # Row mean and variance normalization
    x_normed = x + EPS * np.random.standard_normal(x.shape)
    x_normed -= np.mean(x_normed, axis=-1, keepdims=True)
    x_inv = 1. / np.sqrt(vect_two_norm(x_normed))
    x_diags = np.array(
        [np.diag(x_inv[i].reshape(-1)) for i in range(x_inv.shape[0])])
    x_normed = np.matmul(x_diags, x_normed)
    # Column mean and variance normalization
    x_normed += + EPS * np.random.standard_normal(x_normed.shape)
    x_normed -= np.mean(x_normed, axis=1, keepdims=True)
    x_inv = 1. / np.sqrt(vect_two_norm(x_normed, axis=1))
    x_diags = np.array(
        [np.diag(x_inv[i].reshape(-1)) for i in range(x_inv.shape[0])])
    x_normed = np.matmul(x_normed, x_diags)
    return x_normed
