%elementary_kernel - Build one elementary epanechnikov kernel in several dimensions
%
%-----INPUT-----
% bin_sizes			array containing the sizes of the bins along every dimension
% Bandwidths		array of tha same size containing the bandwidths of the kernel in every direction
%
%-----OUTPUT-----
% result 			multiD matrix containing the elementary kernel
%
% used in every 'build_kernel'-like function, see in particular build_kernel_proba_matrix.m and build_kernel_position_matrix.m
% uses elementary_kernel_1D.m, which is a simple one-dimensional epanechnikov grid-computation
%
%
% Assumptions : 
% 	the kernel is an epanechnikov one, for it is suited for a grid-computation
% 	the bandwidths matrix is completely diagonal, the input is then a list of the diagonal coefficients


% Copyright (C) 2017 by Thibault Balenbois
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.


function result=elementary_kernel(bin_sizes,Bandwidths)

	ndim=size(bin_sizes,1);
	if ndim~=size(Bandwidths,2) && ndim~=size(Bandwidths,1)
		error('inconsistent dimensions');
	end

	local_kernel = elementary_kernel_1D(bin_sizes(1),Bandwidths(1));

	if ndim==1
		result = local_kernel;
	elseif ndim==2
		result = (local_kernel.'*elementary_kernel_1D(bin_sizes(2),Bandwidths(2)));
	else
		result = (local_kernel.'*elementary_kernel_1D(bin_sizes(2),Bandwidths(2)));
		for i = 3:ndim
			local_kernel = elementary_kernel_1D(bin_sizes(i),Bandwidths(i));
			result_temp = result;
			result = local_kernel(1)*result;
			for j = 2:size(local_kernel,2)
				result = cat(i, result, local_kernel(j)*result_temp);
			end
		end
	end
