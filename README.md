# Python implementation of STOI

Implementation of the classical and extended Short Term Objective Intelligibility measures

Intelligibility measure which is highly correlated with the intelligibility of degraded speech signals, e.g., due to additive noise, single/multi-channel noise reduction, binary masking and vocoded speech as in CI simulations. The STOI-measure is intrusive, i.e., a function of the clean and degraded speech signals. STOI may be a good alternative to the speech intelligibility index (SII) or the speech transmission index (STI), when you are interested in the effect of nonlinear processing to noisy speech, e.g., noise reduction, binary masking algorithms, on speech intelligibility.   
Description taken from [Cees Taal's website](http://www.ceestaal.nl/code/)


### Install

`pip install pystoi` or
`pip3 install pystoi`

### Usage
```
from scipy.io.wavfile import read
from pystoi.stoi import stoi

fs, clean = read('path/to/clean/audio')
fs, den = read('path/to/denoised/audio')

# Clean and den should have the same length, and be 1D
d = stoi(clean, den, fs, extended=False)
```

### Matlab code & Testing

All the Matlab code in this repo is taken from or adapted from the code available [here](http://www.ceestaal.nl/code/) (STOI – Short-Time Objective Intelligibility Measure – ) written by Cees Taal.

Thanks to Cees Taal who open-sourced his Matlab implementation and enabled thorough testing of this python code.

If you want to run the tests, you will need Matlab, `matlab.engine` (install instructions [here](https://fr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)) and `matlab_wrapper` (install with `pip install matlab_wrapper`).
The tests can only be ran under Python 2.7 as `matlab.engine` and `matlab_wrapper` are only compatible with Python2.7

### Contribute

Any contribution are welcome, specially to improve the execution speed of the code :
* Vectorize `utils.interm_si_measure` with a matrix correlation
* Vectorize `utils.remove_silent_frames`
* Vectorize OBM matrix multiplication in `stoi.stoi`
* Improve the resampling method to match Matlab's resampling in `tests/`
* Write tests for Python 3 (with [`transplant`](https://github.com/bastibe/transplant) for example)

### Limits

The method is based on audio signal sampled at 10kHz (this is not the problem), so any audio file sampled at a different sampling rate will be resampled to 10kHz. However there is no equivalent of Matlab's resampling in Python, so :

* The tests on an initial sampling rate different than 10kHz are failing.
* The tests on resampling (both with `resampy` and `nnresample`) are failing when compared to Matlab

**Key message** : All the variability in the estimation of the STOI by this package (compared to the original Matlab function) is due to the resampling method. This is a fully tested behavior.

### References
* [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
  Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech',
  ICASSP 2010, Texas, Dallas.
* [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
  Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
  IEEE Transactions on Audio, Speech, and Language Processing, 2011.
* [3] J. Jensen and C. H. Taal, 'An Algorithm for Predicting the
  Intelligibility of Speech Masked by Modulated Noise Maskers',
  IEEE Transactions on Audio, Speech and Language Processing, 2016.
