%position_density_proba - 
%	computes the likelihood of every location P(x|a) in a bin of data. This should be computed for each bin.
%	Essentially applies Bayes rule to previously computed arrays.
%
%----INPUT----
% event_likelihood						likelihood of observation P(a|x) for every location
% prior_estimate						prior estimation of position P(x), which could be constant if non_informative.
%
%----OUTPUT----
% position_density_proba 				probability of finding the mouse in all positions, knowing th data P(x|a)
%
% used
% uses
%
%
% Assumptions : 
% 	The third term of the Bayes rule P(a) is ignored, due to the fact that it is constant over x, so it's "eaten" in the normalization function.
% 	

% Copyright (C) 2017 by Thibault Balenbois
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.


function position_density_proba=position_density_proba(event_likelihood, prior_estimate)

	if size(event_likelihood)~=size(prior_estimate)
		error('dimension mismatch between prior_estimate vector and the event_likelihood vector')
	end
	position_density_proba(:,:)=event_likelihood(:,:).*prior_estimate(:,:);
	% position_density_proba(:)=position_density_proba(:)+0.01;
	if sum(sum(position_density_proba))==0
		position_density_proba=ones(size(event_likelihood));
	end	
	position_density_proba=normalise_deg1(position_density_proba);