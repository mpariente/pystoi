# Python implementation of STOI

Implementation of the Short Term Objective Intelligibility measure

Intelligibility measure which is highly correlated with the intelligibility of degraded speech signals, e.g., due to additive noise, single/multi-channel noise reduction, binary masking and vocoded speech as in CI simulations. The STOI-measure is intrusive, i.e., a function of the clean and degraded speech signals. STOI may be a good alternative to the speech intelligibility index (SII) or the speech transmission index (STI), when you are interested in the effect of nonlinear processing to noisy speech, e.g., noise reduction, binary masking algorithms, on speech intelligibility.
Description taken from [Cees Taal's website](http://www.ceestaal.nl/code/)


### Install

To install this package, clone the repo and install with setup.py :
 ```
git clone https://github.com/mpariente/pystoi
cd pystoi/
python setup.py install
 ```
Note : This is a Python 2.7 package as `matlab_wrapper` (used for testing) is only compatible with Python 2.7


### Usage
```
from scipy.io.wavfile import read
from pystoi.stoi import stoi

fs, clean = read('path/to/clean/audio')
fs, den = read('path/to/denoised/audio')
d = stoi(clean, den, fs)
```

### Matlab code & Testing

All the Matlab code in this repo is taken from or adapted from the code available [here](http://www.ceestaal.nl/code/) (STOI – Short-Time Objective Intelligibility Measure – ) written by Cees Taal.

Thanks to Cees Taal who open-sourced his Matlab implementation and enabled thorough testing of this python code.

If you want to run the tests, you will need Matlab, `matlab.engine` (install instructions [here](https://fr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)) and `matlab_wrapper` (install with `pip install matlab_wrapper`)

### Contribute

Any contribution are welcome, specially to improve the execution speed of the code :
* Vectorize `utils.interm_si_measure` with a matrix correlation
* Vectorize `utils.remove_silent_frames`
* Vectorize OBM matrix multiplication in `stoi.stoi`


### References
* [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
  Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech', ICASSP 2010, Texas, Dallas.
* [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
  Intelligibility Prediction of Time-Frequency Weighted Noisy Speech', IEEE Transactions on Audio, Speech, and Language Processing, 2011.
