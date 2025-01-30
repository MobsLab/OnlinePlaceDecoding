%elementary_kernel_5D - Build one elementary epanechnikov kernel in 5 dimensions
%
%-----INPUT-----
% bin_sizes			array containing the sizes of the bins along every dimension
% Bandwidths		array of tha same size containing the bandwidths of the kernel in every direction
%
%-----OUTPUT-----
% result 			5D matrix containing the elementary kernel
%
% used in every 'build_kernel'-like function, see in particular build_kernel_proba_matrix.m and build_kernel_position_matrix.m
% uses elementary_kernel_1D.m, which is a simple one-dimensional epanechnikov grid-computation
%
%
% Assumptions : 
% 	the kernel is an epanechnikov one, for it is suited for a grid-computation
% 	the bandwidths matrix is completely diagonal, the input is then a list of the diagonal coefficients
% 	the function is not dimension-robust yet, it is only suited for 5D, e.g. a tetrode with position


% Copyright (C) 2017 by Thibault Balenbois
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.


function result=elementary_kernel_5D(bin_sizes,Bandwidths)

	N_dimensions=size(bin_sizes,2);

	%--- Build all 1-dimension kernels
	kernel_a1=elementary_kernel_1D(bin_sizes(1),Bandwidths(1));
	kernel_a2=elementary_kernel_1D(bin_sizes(2),Bandwidths(2));
	kernel_a3=elementary_kernel_1D(bin_sizes(3),Bandwidths(3));
	kernel_a4=elementary_kernel_1D(bin_sizes(4),Bandwidths(4));
	kernel_x=elementary_kernel_1D(bin_sizes(5),Bandwidths(5));
	
	%--- Build 2-dimension kernel
	result=(kernel_a1.'*kernel_a2);

	%--- Build 3-dimension kernel
	result_temp=result;
	result=kernel_a3(1)*result;
	for i = 2:size(kernel_a3,2)
		result=cat(3, result, kernel_a3(i)*result_temp);
	end

	%--- Build 4-dimension kernel
	result_temp=result;
	result=kernel_a4(1)*result;
	for i = 2:size(kernel_a4,2)
		result=cat(4, result, kernel_a4(i)*result_temp);
	end

	%--- Build 5-dimension kernel
	result_temp=result;
	result=kernel_x(1)*result;
	for i = 2:size(kernel_x,2)
		result=cat(5, result, kernel_x(i)*result_temp);
	end