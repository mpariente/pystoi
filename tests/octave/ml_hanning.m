function w = ml_hanning(M)
    %   Compute a Hann window compatible with the MATLAB `hanning` function
    w = .5 * (1 - cos(2 * pi * (1:M)'/double(M + 1)));
end

