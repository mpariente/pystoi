function x_stdft = stdft(x, N, K, N_fft)
%   X_STDFT = X_STDFT(X, N, K, N_FFT) returns the short-time
%	hanning-windowed dft of X with frame-size N, overlap K and DFT size
%   N_FFT. The columns and rows of X_STDFT denote the frame-index and
%   dft-bin index, respectively.

frames      = 1:K:(length(x)-N);
x_stdft     = zeros(length(frames), N_fft);

w           = hanning(N);
x           = x(:);

for i = 1:length(frames)
    ii              = frames(i):(frames(i)+N-1);
	x_stdft(i, :) 	= fft(x(ii).*w, N_fft);
end
