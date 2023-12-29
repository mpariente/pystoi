# Python implementation of STOI

Implementation of the classical and extended Short Term Objective Intelligibility measures

Intelligibility measure which is highly correlated with the intelligibility of degraded speech signals, e.g., due to additive noise, single/multi-channel noise reduction, binary masking and vocoded speech as in CI simulations. The STOI-measure is intrusive, i.e., a function of the clean and degraded speech signals. STOI may be a good alternative to the speech intelligibility index (SII) or the speech transmission index (STI), when you are interested in the effect of nonlinear processing to noisy speech, e.g., noise reduction, binary masking algorithms, on speech intelligibility.   
Description taken from [Cees Taal's website](http://www.ceestaal.nl/code/)


### Install

`pip install pystoi` or
`pip3 install pystoi`

### Usage
```
import soundfile as sf
from pystoi import stoi

clean, fs = sf.read('path/to/clean/audio')
denoised, fs = sf.read('path/to/denoised/audio')

# Clean and den should have the same length, and be 1D
d = stoi(clean, denoised, fs, extended=False)
```

### Running the Octave tests 

```bash 
sudo apt update 
sudo apt install octave octave-signal 
pip install oct2py
```

```bash
python -m pytest tests/test_python_octave.py
python -m pytest tests/test_stoi_octave.py
```

### Matlab code & Testing

All the Matlab code in this repo is taken from or adapted from the code available [here](http://www.ceestaal.nl/code/) (STOI – Short-Time Objective Intelligibility Measure – ) written by Cees Taal.

Thanks to Cees Taal who open-sourced his Matlab implementation and enabled thorough testing of this python code.

If you want to run the tests, you will need Matlab, `matlab.engine` (install instructions [here](https://fr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)) and `matlab_wrapper` (install with `pip install matlab_wrapper`).
The tests can only be ran under Python 2.7 as `matlab.engine` and `matlab_wrapper` are only compatible with Python2.7
Tests are passing at relative and absolute tolerance of `1e-3`, which is enough for the considered application (all the variability is coming from the resampling method when signals are not natively sampled at 10kHz).

Very big thanks to @gauss256 who translated all the matlab scripts to Octave, and wrote all the tests for it!

### Contribute

Any contribution are welcome~, specially to improve the execution speed of the code~ (thank you Przemek Pobrotyn for a 4x speed-up!) :

* ~Improve the resampling method to match Matlab's resampling in `tests/`.~ This can be considered a solved issue thanks to @gauss256 !
* Write tests for Python 3 (with [`transplant`](https://github.com/bastibe/transplant) for example)


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
