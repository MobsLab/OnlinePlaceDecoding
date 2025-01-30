%elementary_kernel_1D - Build one elementary epanechnikov kernel in 1 dimension
%
%----INPUT----
% bin_size			size of the bins
% Bandwidth			bandwidth of the kernel
%
%----OUTPUT----
% result 			1d array containing the elementary kernel
%
% used in every 'build_kernel'-like function, see in particular build_kernel_proba_matrix.m and build_kernel_position_matrix.m
% used in computing high-dimension kernels, see elementary_kernel_5D.m and the general version elementary_kernel.m
%
%
% Assumptions : 
% 	the kernel is an epanechnikov one, for it is suited for a grid-computation
%	Since it is a grid computation, the actual output is an integration between samplig points of the Kernel function. This is to keep it as a meaningful probability density.
%

% Copyright (C) 2017 by Thibault Balenbois
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.

function result=elementary_kernel_1D(bin_size,Bandwidth)

	n_points=floor(2*Bandwidth/bin_size)+1;

	%--- First we separate the x axis with n_points+2 points, ranging from -1 to 1
	X=(((0:n_points+1)*2*Bandwidth/(n_points+1))-Bandwidth)/Bandwidth;

	%--- Then we take the integral of the Kernel function between each of these points (in order to keep this a probability density)
	result=0.75*(X(2:end)-X(1:end-1)-(1/3)*(X(2:end).^3-X(1:end-1).^3));