function x_stdft = stdft(x, N, K, N_fft)

frames      = 1:K:(length(x)-N);
x_stdft     = zeros(length(frames), N_fft);

w           = ml_hanning(N);
x           = x(:);

for i = 1:length(frames)
    ii              = frames(i):(frames(i)+N-1);
	x_stdft(i, :) 	= fft(x(ii).*w, N_fft);
end
