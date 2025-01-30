%event_likelihood - 
%	computes the likelihood of an event for every location P(a|x). This should be computed for each bin.
%	Finding P(x|a) can be computed with Bayes' rule after that.
%
%----INPUT----
% spike_events						2D array containing all spike events' features in a bin
% time_bin							array containing the number of bins along each dimension
% Rate_function_matrix 				rate function matrix that can be computed in build_kernel_proba_matrix.m and bayesian_encoding.m
% Marginal_rate_function 			arry containing the marginal rate function, computed from build_kernel_position_matrix.m and bayesian_encoding.m
% Occupation						array containing the occupation rate of the mouse, regardless of spiking activity, computed from build_kernel_position_matrix.m and bayesian_encoding.m
% two_first_bins					2D array (2,n) containing the origin value of the first two bins of Rate_function_matrix in each dimension
%
%----OUTPUT----
% event_likelihood 					array containing the likelihood of the event as a function of the assumed position P(a|x).
%
% used in bayesian_encoding.m script
% uses
%
%
% Assumptions : 
% 	the binning is assumed to be sufficiently small so that there 0, 1, or 2 spike events per bin
% 	the position of the Rate_function_matrix must be in the last dimension
%
% Warning 
%	No complex calculation is needed, but adressing of a multiD matrix. This could be long.

% Copyright (C) 2017 by Thibault Balenbois
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.


function event_likelihood=event_likelihood(spike_events, time_bin, Rate_function_matrix, Marginal_rate_function_matrix, Occupation, two_first_bins)


	n_spike=size(spike_events,2);

	%--- Poisson's law to compute the absence of spike probability
	%--- We use a 'mask' to get rid of bins where the mouse has spent too little time, as those won't be statistically relevent
	mask(:,:)=(Occupation(:,:)>max(max(Occupation))/15);
	occupation_inverse=clean_matrix(ones(size(Occupation))./Occupation).*mask;
	no_other_spike_likelihood(:,:)=((time_bin)^n_spike)*exp(-(time_bin)*Marginal_rate_function_matrix(:,:).*occupation_inverse);
	% no_other_spike_likelihood(:,:)=(ones(size(Marginal_rate_function_matrix,2)));

	event_likelihood(:,:)=normalise_deg1(no_other_spike_likelihood(:,:)).*mask(:,:);

	if n_spike~=0
		if size(spike_events,1)~=ndims(Rate_function_matrix)-2
			error('number of dimensions not coherent between the rate function and the extracted features');
		end
		for i = 1:n_spike
			current_event(:)=spike_events(:,i);

			% Position of event in feature space
			position={};
			for j =1:size(spike_events,1)
				position=[position min(max(floor((current_event(j)-two_first_bins(j,1))/(two_first_bins(j,2)-two_first_bins(j,1)))+1,1),size(Rate_function_matrix,j))];
			end
			
			spike_likelihood(:,:)=Rate_function_matrix(position{1,1:size(position,2)},:,:);
			spike_likelihood(:,:)=spike_likelihood(:,:).*occupation_inverse(:,:);
			if sum(sum(spike_likelihood))==0
				spike_likelihood=normalise_deg1(ones(size(Marginal_rate_function_matrix)));
			else
				spike_likelihood(:,:)=normalise_deg1(sum(sum(spike_likelihood))/1000+spike_likelihood(:,:));
			end
			event_likelihood(:,:)=event_likelihood(:,:).*spike_likelihood(:,:);
		end
	end
