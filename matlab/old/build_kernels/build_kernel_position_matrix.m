%build_kernel_position_matrix - 
%	Build the position matrix of the data P(x). 
%	The marginal kernel density matrix will be L(x)=N*P(x)/T/Occupation(x)
%
%----INPUT----
% list_events			array containing the position of all spike events recorded
% N_bins				number of bins (2-D array when working in two dimensions)
% Bandwidths			bandwidths of the kernel (2-D array when working in two dimensions)
%
%----OUTPUT----
% result 				array containing the probability function P(x) in one or two dimensions.
%
% used in bayesian_encoding.m script
% uses elementary_kernel_1D.m, which is a 1-dimensional epanechnikov kernel grid-computation
%
%
% Assumptions : 
% 	the kernel is an epanechnikov one, for it is suited for a grid-computation
%	the position must have one or two dimensions, any other will be rejected
%


% Copyright (C) 2017 by Thibault Balenbois
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.



function result=build_kernel_position_matrix(list_events,N_bins,Bandwidth)

	if size(list_events,1)==1

		M=zeros(N_bins);
		M=M(1,:);
		
		bins_x=[min(list_events):(max(list_events)-min(list_events))/N_bins:max(list_events)];
		bin_size=bins_x(2)-bins_x(1);


		%--- Building elementary Kernel
		Kernel=elementary_kernel_1D(bin_size,Bandwidth);

		for event_id = 1:size(list_events,2)
			%--- Selection of the event to treat
			current_event=list_events(event_id);

			%--- Position of the event
			i5=floor((current_event-bins_x(1))/(bins_x(2)-bins_x(1)))+1;

			%--- Beginning and end of the kernel target
			ib5=max(i5-floor(size(Kernel,2)/2),1);		ie5=min(ib5+size(Kernel,2)-1,size(M,2));

			%--- Hitting target on the kernel proba matrix
			M(ib5:ie5) = ...
			M(ib5:ie5) + ...
			Kernel(1:ie5-ib5+1);

		end

	elseif size(list_events,1)==2

		M=zeros(N_bins);
		
		bins_x=[min(list_events(1,:)):(max(list_events(1,:))-min(list_events(1,:)))/N_bins(1):max(list_events(1,:))];
		bins_y=[min(list_events(2,:)):(max(list_events(2,:))-min(list_events(2,:)))/N_bins(2):max(list_events(2,:))];
		bin_sizes=[bins_x(2)-bins_x(1);bins_y(2)-bins_y(1)];


		%--- Building elementary Kernel
		Kernel=elementary_kernel(bin_sizes,Bandwidth);

		for event_id = 1:size(list_events,2)
			%--- Selection of the event to treat
			current_event=list_events(:,event_id);

			%--- Position of the event
			i5=floor((current_event(1)-bins_x(1))/(bins_x(2)-bins_x(1)))+1;
			i6=floor((current_event(2)-bins_y(1))/(bins_y(2)-bins_y(1)))+1;


			%--- Beginning and end of the kernel target
			ib5=max(i5-floor(size(Kernel,1)/2)+mod(size(Kernel,1)+1,2),1);		ie5=min(i5+floor(size(Kernel,1)/2),size(M,1));
			ib6=max(i6-floor(size(Kernel,2)/2)+mod(size(Kernel,2)+1,2),1);		ie6=min(i6+floor(size(Kernel,2)/2),size(M,2));

			%--- Hitting target on the kernel proba matrix
			M(ib5:ie5, ib6:ie6) = M(ib5:ie5, ib6:ie6) + ...
			Kernel(floor(size(Kernel,1)/2)+mod(size(Kernel,1),2)+ib5-i5:floor(size(Kernel,1)/2)+mod(size(Kernel,1),2)+ie5-i5, ... 
				floor(size(Kernel,2)/2)+mod(size(Kernel,2),2)+ib6-i6:floor(size(Kernel,2)/2)+mod(size(Kernel,2),2)+ie6-i6);
			
			if mod(event_id,floor(size(list_events,2)/10))==0
				disp(['Building position matrix, ', num2str(100*event_id/size(list_events,2)), '% achieved']);
			end
		end

	else
		error('dimension is neither 1 nor 2 for position -> in what space are we?')
	end


	result=M/size(list_events,2);