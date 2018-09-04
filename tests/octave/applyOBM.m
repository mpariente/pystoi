function X = applyOBM(x, OBM, N_frame, NFFT, NUMBAND)

x_hat = stdft(x, N_frame, N_frame/2, NFFT);
x_hat = x_hat(:, 1:(NFFT/2+1)).';

X = zeros(NUMBAND, size(x_hat, 2));
for i = 1:size(x_hat, 2)
    X(:, i)	= sqrt(OBM*abs(x_hat(:, i)).^2);
end
