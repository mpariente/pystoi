function  [A cf] = thirdoct(fs, N_fft, numBands, mn)

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
