%build_kernel_proba_matrix - 
%	Build the probablity matrix of the data P(a,x). 
%	The kernel density matrix will be L(a,x)=N*P(a,x)/T/Occupation(x)
%
%----INPUT----
% list_events			2D array containing all spike events recorded along n+1 dimensions : n features and the mouse position
% N_bins				array containing the number of bins along each dimension
% Bandwidths			array of tha same size containing the bandwidths of the kernel in every direction
%
%----OUTPUT----
% result 				multiD array containing the probability function P(a,x).
%
% used in bayesian_encoding.m script
% uses elementary_kernel.m, which is a multi-dimensional epanechnikov kernel grid-computation
%
%
% Assumptions : 
% 	the kernel is an epanechnikov one, for it is suited for a grid-computation
% 	the bandwidths matrix is completely diagonal, the input is then a list of the diagonal coefficients
%
% Warning 
%	The result is a multiD matrix with supposedly a certain number of bins in each dimension.
% 	CPU time may explode if one is not careful.

% Copyright (C) 2017 by Thibault Balenbois
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.



function result=build_kernel_proba_matrix(list_events,N_bins,Bandwidths)

	if size(list_events,1)~=size(N_bins,2)
		error('The number of dimensions is not the same between the list of events and the number of bins in each feature dimension');
	elseif size(N_bins,2)~=size(Bandwidths,2)
		error('The number of dimensions between the number of bins and the bandwidths is not the same');
	end

	M=zeros(N_bins);

	two_first_bins=compute_two_first_bins(list_events,N_bins);


	%--- Building elementary Kernel
	Kernel=elementary_kernel(two_first_bins(:,2)-two_first_bins(:,1),Bandwidths);
	size(Kernel)

	for event_id = 1:size(list_events,2)
		%--- Selection of the event to treat
		current_event=list_events(:,event_id);
		
		%--- Next paragraph computes targets to hit
		target={};
		kernel_target={};
		for dim = 1:size(list_events,1)
			%--- Position of the event in feature space and real space
			position=min(floor((current_event(dim)-two_first_bins(dim,1))/(two_first_bins(dim,2)-two_first_bins(dim,1)))+1,size(M,dim));
			%--- Beginning and end of the target, for the proba matrix and for the kernel
			position_start=min(max(position-floor(size(Kernel,dim)/2)+mod(size(Kernel,dim)+1,2),1),size(M,dim));
			position_stop=max(min(position+floor(size(Kernel,dim)/2),size(M,dim)),1);
			%--- Saving these
			target=[target position_start:position_stop];
			kernel_target=[kernel_target floor(size(Kernel,dim)/2)+mod(size(Kernel,dim),2)+position_start-position:floor(size(Kernel,dim)/2)+mod(size(Kernel,dim),2)+position_stop-position];
		end

		%--- Hitting target on the kernel proba matrix
		%--- Longest step. All others are negligeable compared to this one
		M(target{1,1:size(target,2)}) = M(target{1,1:size(target,2)}) + Kernel(kernel_target{1,1:size(kernel_target,2)});

		if mod(event_id,floor(size(list_events,2)/10))==0
			disp(['Building proba matrix, ', num2str(100*event_id/size(list_events,2)), '% achieved']);
		end
		
	end


	result=M/size(list_events,2);