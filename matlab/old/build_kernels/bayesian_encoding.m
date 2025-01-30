%bayesian_encoding - 
%	Build the rate function P(a,x) and the marginalized rate function P(x) 
%	Basically encode what is needed for a bayesian decoding of a polytrode
%
%----INPUT----
% list_events			2D array containing all spike events recorded along several dimensions (signal features + position)
% N_bins				array containing the number of bins along each dimension
% Bandwidths			array of tha same size containing the bandwidths of the kernel in every direction
% time					length of the complete recording in time (units ??????)
%
%----OUTPUT----
% result = [M1,M2,occupation] 		
% M1					multiD array containing the rate function P(a,x).
% M2					2-D array containing the marginalized rate function P(x).
% occupation			2-D array containing the occupation matrix, regardless of spike activity.
%
%----EXAMPLE----
% used in a working script called 'encoding_script_04_2017', which uses fake data
% also used in 'reading_raw_data_testing', which is a basis for a future decoding script
% uses build_kernel_proba_matrix, build_kernel_position_matrix
%
%
% Assumptions : 
% 	the elementary kernel is an epanechnikov one, for it is suited for a grid-computation
% 	the bandwidths matrix is completely diagonal, the input is then a list of the diagonal coefficients
%
% Warning 
%	The result contains a multiD matrix with supposedly a certain number of bins in each dimension.
% 	CPU time may explode if one is not careful.

% Copyright (C) 2017 by Thibault Balenbois
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.

function [M1,M2,occupation]=bayesian_encoding(list_events,places,N_bins,Bandwidths,time)

	if size(list_events,1)~=size(N_bins,2)
		error('The list of events and number of bins array must have the same dimension');
	elseif size(list_events,1)~=size(Bandwidths,2)
		error('The Bandwidths array must have the same number of elements as the number of dimensions of the list of events');
	end

	
	M1=build_kernel_proba_matrix(list_events,N_bins,Bandwidths)*size(list_events,2)/time;
	disp('Proba matrix build.')
	M2=build_kernel_position_matrix(list_events(end-1:end,:),N_bins(end-1:end),Bandwidths(end-1:end))*size(list_events,2)/time;
	disp('Position matrix build.')
	occupation=build_kernel_position_matrix(places,N_bins(end-1:end),Bandwidths(end-1:end));
	disp('Occupation matrix build.')


	%--- The resulting matrices must be normalized by occupation rate (for now it is done somwhere else for robustness reasons)
	% M2(:,:)=M2(:,:)./1;
	% feature_target={};
	% target_size=[];
	% for dim = 1:size(list_events,1)-2
	% 	feature_target=[feature_target 1:size(M1,dim)];
	% 	target_size=[target_size size(M1,dim)];
	% end
	% for x=1:N_bins(end-1)
	% 	for y=1:N_bins(end)
	% 		target=[feature_target x:x y:y];
	% 		M1(target{1,1:size(target,2)})=M1(target{1,1:size(target,2)})/1;
	% 	end
	% end

	M1=clean_matrix(M1,0);
	M2=clean_matrix(M2,0);

	
	
