function d = estoi(x, y, fs_signal)
%   d = estoi(x, y, fs_signal) returns the output of the extended short-time
%   objective intelligibility (ESTOI) predictor.
%  
% Implementation of the Extended Short-Time Objective
% Intelligibility (ESTOI) predictor, described in Jesper Jensen and
% Cees H. Taal, "An Algorithm for Predicting the Intelligibility of
% Speech Masked by Modulated Noise Maskers," IEEE Transactions on
% Audio, Speech and Language Processing, 2016.
%
% Input:
%        x:         clean reference time domain signal
%        y:         noisy/processed time domain signal
%        fs_signal: sampling rate [Hz]
%
% Output:
%        d: intelligibility index
%
%
% Copyright 2016: Aalborg University, Section for Signal and Information Processing. 
% The software is free for non-commercial use. 
% The software comes WITHOUT ANY WARRANTY.


if length(x)~=length(y)
  error('x and y should have the same length');
end

% initialization
x               = x(:);                   % clean speech column vector
y               = y(:);                   % processed speech column vector

fs              = 10000;                  % sample rate of proposed intelligibility measure
N_frame         = 256;                    % window support
K               = 512;                    % FFT size
J               = 15;                     % Number of 1/3 octave bands
mn              = 150;                    % Center frequency of first 1/3 octave band in Hz.
[H,fc_thirdoct] = thirdoct(fs, K, J, mn); % Get 1/3 octave band matrix
N               = 30;                     % Number of frames for intermediate intelligibility measure
dyn_range       = 40;                     % speech dynamic range

% resample signals if other samplerate is used than fs
if fs_signal ~= fs
  x	= resample(x, fs, fs_signal);
  y 	= resample(y, fs, fs_signal);
end

% remove silent frames
[x y] = removeSilentFrames(x, y, dyn_range, N_frame, N_frame/2);

% apply 1/3 octave band TF-decomposition
x_hat     	= stdft(x, N_frame, N_frame/2, K); % apply short-time DFT to clean speech
y_hat     	= stdft(y, N_frame, N_frame/2, K); % apply short-time DFT to processed speech


x_hat       = x_hat(:, 1:(K/2+1)).'; % take clean single-sided spectrum
y_hat       = y_hat(:, 1:(K/2+1)).'; % take processed single-sided spectrum

X           = zeros(J, size(x_hat, 2)); % init memory for clean speech 1/3 octave band TF-representation
Y           = zeros(J, size(y_hat, 2)); % init memory for processed speech 1/3 octave band TF-representation

for i = 1:size(x_hat, 2)
  X(:, i)	= sqrt(H*abs(x_hat(:, i)).^2); % apply 1/3 octave band filtering
  Y(:, i)	= sqrt(H*abs(y_hat(:, i)).^2);
end

% loop all segments of length N and obtain intermediate intelligibility measure for each
d1 = zeros(length(N:size(X, 2)),1); % init memory for intermediate intelligibility measure
for m=N:size(X,2)
    X_seg  	= X(:, (m-N+1):m); % region of length N with clean TF-units for all j
    Y_seg  	= Y(:, (m-N+1):m); % region of length N with processed TF-units for all j
    X_seg = X_seg + eps*randn(size(X_seg)); % to avoid divide by zero
    Y_seg = Y_seg + eps*randn(size(Y_seg)); % to avoid divide by zero
    
    %% first normalize rows (to give \bar{S}_m)
    XX = X_seg - mean(X_seg.').'*ones(1,N); % normalize rows to zero mean
    YY = Y_seg - mean(Y_seg.').'*ones(1,N); % normalize rows to zero mean
    
    YY = diag(1./sqrt(diag(YY*YY')))*YY; % normalize rows to unit length
    XX = diag(1./sqrt(diag(XX*XX')))*XX; % normalize rows to unit length

    XX = XX + eps*randn(size(XX)); % to avoid corr.div.by.0
    YY = YY + eps*randn(size(YY)); % to avoid corr.div.by.0

    %% then normalize columns (to give \check{S}_m)
    YYY = YY - ones(J,1)*mean(YY); % normalize cols to zero mean
    XXX = XX - ones(J,1)*mean(XX); % normalize cols to zero mean

    YYY = YYY*diag(1./sqrt(diag(YYY'*YYY))); % normalize cols to unit length
    XXX = XXX*diag(1./sqrt(diag(XXX'*XXX))); % normalize cols to unit length

    %compute average of col.correlations (by stacking cols)
    d1(m-N+1) = 1/N*XXX(:).'*YYY(:);
end
d = mean(d1);


%%
function  [A cf] = thirdoct(fs, N_fft, numBands, mn)
%   [A CF] = THIRDOCT(FS, N_FFT, NUMBANDS, MN) returns 1/3 octave band matrix
%   inputs:
%       FS:         samplerate
%       N_FFT:      FFT size
%       NUMBANDS:   number of bands
%       MN:         center frequency of first 1/3 octave band
%   outputs:
%       A:          octave band matrix
%       CF:         center frequencies

f               = linspace(0, fs, N_fft+1);
f               = f(1:(N_fft/2+1));
k               = 0:(numBands-1);
cf              = 2.^(k/3)*mn;
fl              = sqrt((2.^(k/3)*mn).*2.^((k-1)/3)*mn);
fr              = sqrt((2.^(k/3)*mn).*2.^((k+1)/3)*mn);
A               = zeros(numBands, length(f));

for i = 1:(length(cf))
  [a b]                   = min((f-fl(i)).^2);
  fl(i)                   = f(b);
  fl_ii                   = b;
  
  [a b]                   = min((f-fr(i)).^2);
  fr(i)                   = f(b);
  fr_ii                   = b;
  A(i,fl_ii:(fr_ii-1))	= 1;
end

rnk         = sum(A, 2);
numBands  	= find((rnk(2:end)>=rnk(1:(end-1))) & (rnk(2:end)~=0)~=0, 1, 'last' )+1;
A           = A(1:numBands, :);
cf          = cf(1:numBands);

%%
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

%%
function [x_sil y_sil] = removeSilentFrames(x, y, range, N, K)
%   [X_SIL Y_SIL] = REMOVESILENTFRAMES(X, Y, RANGE, N, K) X and Y
%   are segmented with frame-length N and overlap K, where the maximum energy
%   of all frames of X is determined, say X_MAX. X_SIL and Y_SIL are the
%   reconstructed signals, excluding the frames, where the energy of a frame
%   of X is smaller than X_MAX-RANGE

x       = x(:);
y       = y(:);

frames  = 1:K:(length(x)-N);
w       = hanning(N);
msk     = zeros(size(frames));

for j = 1:length(frames)
  jj      = frames(j):(frames(j)+N-1);
  msk(j) 	= 20*log10(norm(x(jj).*w)./sqrt(N));
end

msk     = (msk-max(msk)+range)>0;
count   = 1;

x_sil   = zeros(size(x));
y_sil   = zeros(size(y));

for j = 1:length(frames)
  if msk(j)
    jj_i            = frames(j):(frames(j)+N-1);
    jj_o            = frames(count):(frames(count)+N-1);
    x_sil(jj_o)     = x_sil(jj_o) + x(jj_i).*w;
    y_sil(jj_o)  	= y_sil(jj_o) + y(jj_i).*w;
    count           = count+1;
  end
end

x_sil = x_sil(1:jj_o(end));
y_sil = y_sil(1:jj_o(end));


