import numpy as np
import scipy
from resampy import resample
import utils

# Constant definition
FS = 10000                          # Sampling frequency
N_FRAME = 256                       # Window support
NFFT = 512                          # FFT Size
NUMBAND = 15                        # Number of 13 octave band
MINFREQ = 150                       # Center frequency of 1st octave band (Hz)
OBM, CF = utils.thirdoct(FS, NFFT, NUMBAND, MINFREQ)  # Get 1/3 octave band matrix
N = 30                              # N. frames for intermediate intelligibility
BETA = -15.                         # Lower SDR bound
DYN_RANGE = 40                      # Speech dynamic range


def stoi(x, y, fs_sig, extended=False):
    """ Short term objective intelligibility
    Computes the STOI (See [1][2]) of a denoised signal compared to a
    clean signal, The output is expected to have a monotonic
    relation with the subjective speech-intelligibility, where a higher d
    denotes better speech intelligibility

    # Arguments
        x : clean original speech
        y : denoised speech
        fs_sig : sampling rate of x and y
        extended : Boolean, whether to use the extended STOI described in [3]
    # Returns
        Short time objective intelligibility measure between clean and denoised
        speech
    # Raises
        AssertionError : if x and y have different lengths
    # Reference
        [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
            Objective Intelligibility Measure for Time-Frequency Weighted Noisy
            Speech', ICASSP 2010, Texas, Dallas.
        [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
            Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
            IEEE Transactions on Audio, Speech, and Language Processing, 2011.
        [3] Jesper Jensen and Cees H. Taal, 'An Algorithm for Predicting the
            Intelligibility of Speech Masked by Modulated Noise Maskers',
            IEEE Transactions on Audio, Speech and Language Processing, 2016.
    """
    if x.shape != y.shape:
        raise Exception('x and y should have the same length,' +
                        'found {} and {}'.format(len(x), len(y)))
    # Resample is fs_sig is different than fs
    if fs_sig != FS:
        x = resample(x, fs_sig, FS)
        y = resample(y, fs_sig, FS)
    # Remove silent frames
    x, y = utils.remove_silent_frames(x, y, DYN_RANGE, N_FRAME, int(N_FRAME/2))
    # Take STFT
    x_spec = utils.stft(x, N_FRAME, NFFT, overlap=2).transpose()
    y_spec = utils.stft(y, N_FRAME, NFFT, overlap=2).transpose()
    # Init third octave band TF representation
    x_tob = np.zeros((NUMBAND, x_spec.shape[1]))
    y_tob = np.zeros((NUMBAND, y_spec.shape[1]))
    # Apply OB matrix to the spectrograms as in Eq. (1)
    for i in range(x_tob.shape[1]):
        x_tob[:, i] = np.sqrt(np.matmul(OBM, np.square(np.abs(x_spec[:, i]))))
        y_tob[:, i] = np.sqrt(np.matmul(OBM, np.square(np.abs(y_spec[:, i]))))

    if extended:
        interm_meas = np.zeros((x_tob.shape[1] - N + 1, ))
    else:
        interm_meas = np.zeros((NUMBAND, x_tob.shape[1] - N + 1))
        clip = 10**(-BETA/20)

    for m in range(N, x_tob.shape[1] + 1):
        x_seg = x_tob[:, m-N:m]
        y_seg = y_tob[:, m-N:m]
        if extended:
            x_n = utils.row_col_normalize(x_seg)
            y_n = utils.row_col_normalize(y_seg)
            interm_meas[m-N] = np.sum(x_n * y_n / N)
        else:
            alpha = np.sqrt(np.sum(np.square(x_seg), 1) / np.sum(np.square(y_seg), 1))
            a_y_seg = np.multiply(y_seg.transpose(), alpha).transpose()
            for j in range(NUMBAND):
                Y_prime = np.minimum(a_y_seg[j, :], x_seg[j, :] * (1 + clip))
                interm_meas[j, m-N] = utils.corr(x_seg[j, :], Y_prime[:])
    d = np.mean(interm_meas)
    return d
