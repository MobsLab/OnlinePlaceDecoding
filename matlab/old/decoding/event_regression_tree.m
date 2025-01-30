%event_regression_tree - 
%	computes the best guess for the position using regression trees. This should be computed for each bin.
%	
%
%----INPUT----
% spike_events						2D array containing all spike events' features in a bin
% TreeX					 			regression tree for one tetrode for dimension X
% TreeY								regression tree for one tetrode for dimension Y
%
%----OUTPUT----
% positions		 					2D array with the guesses od X and Y for each spike (e.g. 3 spikes mean a 2x3 array)
%
% used in encoding.m script in the regression_tree branch.
% uses
%
%
% Assumptions : 
% 	the binning is assumed to be sufficiently small so that there 0, 1, or 2 spike events per bin
% 	
%
% Warning 
%	It's all based on the 'predict' function of a MATLAB toolbox : kink of obscure what it does ...

% Copyright (C) 2017 by Thibault Balenbois
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.


function positions=event_regression_tree(spike_events, TreeX, TreeY)


	n_spike=size(spike_events,2);

	positions=[];
	if n_spike~=0
		for i = 1:n_spike
			current_event(:)=spike_events(:,i)';

			X=predict(TreeX, current_event);
			Y=predict(TreeY, current_event);

			x_prediction=X{1,1};
			y_prediction=Y{1,1};
			x_prediction=x_prediction(1);
			y_prediction=y_prediction(1);

			try
				positions=[positions [x_prediction;y_prediction]];
			catch
				error('something went wrong');
			end
		end
	end
