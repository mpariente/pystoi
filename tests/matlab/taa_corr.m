function rho = taa_corr(x, y)
%   RHO = TAA_CORR(X, Y) Returns correlation coeffecient between column
%   vectors x and y. Gives same results as 'corr' from statistics toolbox.
xn    	= x-mean(x);
xn  	= xn/sqrt(sum(xn.^2));
yn   	= y-mean(y);
yn    	= yn/sqrt(sum(yn.^2));
rho   	= sum(xn.*yn);
