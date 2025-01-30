function encoding(varargin)
%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%
Behavior=importdata('behavResources.mat');
try isstr(DATA);
catch 
	try
		DATA=load('DATA_reduced.mat','DATA');
		DATA=DATA.DATA;
		nb_clusters=load('DATA_reduced.mat','nb_clusters');
		nb_clusters=nb_clusters.nb_clusters;
	catch
		try
			DATA=load('DATA.mat','DATA');
			DATA=DATA.DATA;
			nb_clusters=load('DATA.mat','nb_clusters');
			nb_clusters=nb_clusters.nb_clusters;
		catch
			error('Impossible to load DATA.mat, have you used the reading script ?');
		end
	end
end





%%%%%%%%%%%--- LOAD BASIC INFOS ---%%%%%%%%%%%
disp('Starting encoding.');disp(sprintf('\n'));
n_polytrode=size(nb_clusters,2);

max_time=0;
for polytrode=1:n_polytrode
	try
		local_time=DATA(polytrode).spikes(end);
	catch
		local_time=0;
	end
	if local_time>max_time
		max_time=local_time;
	end
end
min_time=max_time;
for polytrode=1:n_polytrode
	try
		local_min=DATA(polytrode).spikes(1);
	catch
		local_min=max_time;
	end
	if local_min<min_time
		min_time=local_min;
	end
end
% learning_time=(max_time-min_time)/2;
% start_time=0;
start_time = 1180;
learning_time = 2000;






%% Initialization of variables
if nargin==0
	Ha=150;
	Hx=15;
elseif nargin==2
	Hx=varargin{1};
	Ha=varargin{2};
else
	Ha=150; Hx=15;
	warning('Didn''t understand the number of arguments (neither 2 nor 0), bandwidths set to default values');
end
Bandwidths=[Ha Hx];
Nbin_a=25;
Nbin_x=30;
N_bins=[Nbin_a Nbin_x];


for polytrode=1:n_polytrode
	disp(['Starting electrode ',num2str(polytrode),'.']);
	n_events=size(DATA(polytrode).events,2);

	%%%%%%%%%%%--- ENCODING EVENTS ---%%%%%%%%%%%

	if n_events~=0
		n_events_learning=1; n_events_start=1;
		%%-- Determines the number of events to learn from.
		while (DATA(polytrode).spikes(n_events_start))<start_time
			if n_events_start==n_events
				break
			end
			n_events_start=n_events_start+1;
		end
		n_events_learning = n_events_start;
		while (DATA(polytrode).spikes(n_events_learning))<start_time+learning_time
			if n_events_learning==n_events
				break
			end
			n_events_learning=n_events_learning+1;
		end

		%%-- Making useful variables for computing
		T=DATA(polytrode).spikes(n_events_learning)-DATA(polytrode).spikes(n_events_start);%/19531;
		N_bins=[N_bins(1) N_bins(end)]; Bandwidths=[Bandwidths(1) Bandwidths(end)];
		for adim=1:size(DATA(polytrode).events,1)-3
			Bandwidths=[Bandwidths(1) Bandwidths];
			N_bins=[N_bins(1) N_bins];
		end
		Bandwidths=[Bandwidths Bandwidths(end)];
		N_bins=[N_bins N_bins(end)];

		
		clearvars Nbin_a Ha Hx posX posY channel data_id event led_id lfp_id place_id res_id spike spikes spk_id time_stamp
		
		%--- Computation of our different Kernel matrices, this is the actual encoding, right there.
		[Rate_function_matrix, Marginal_rate_function_matrix, Occupation_matrix] = bayesian_encoding(DATA(polytrode).events(:,n_events_start:n_events_learning), DATA(polytrode).cleaned_positions(:,1:end/2), N_bins, Bandwidths,T);
		two_first_bins=compute_two_first_bins(DATA(polytrode).events(:,n_events_start:n_events_learning), N_bins);

		%--- Stocking our results.
		ENCODED_DATA(polytrode).Rate_Function=Rate_function_matrix;
		ENCODED_DATA(polytrode).Marginal_Rate_Function=Marginal_rate_function_matrix;
		ENCODED_DATA(polytrode).Occupation=Occupation_matrix;
		ENCODED_DATA(polytrode).twofirstbins=two_first_bins;
		ENCODED_DATA(polytrode).n_events_learning=n_events_learning;
		ENCODED_DATA(polytrode).learning_time=learning_time;
		ENCODED_DATA(polytrode).start_time=start_time;
		disp(['Electrode ',num2str(polytrode),' encoded.']);disp(sprintf('\n'));

	else
		ENCODED_DATA(polytrode).Rate_Function=ones(N_bins);
		ENCODED_DATA(polytrode).Marginal_Rate_Function=normalise_deg1(ones(N_bins(end-1:end)));
		ENCODED_DATA(polytrode).Occupation=normalise_deg1(ones(N_bins(end-1:end)));
		ENCODED_DATA(polytrode).twofirstbins=[zeros(size(N_bins,2),1) ones(size(N_bins,2),1)];
		ENCODED_DATA(polytrode).n_events_learning=0;
		ENCODED_DATA(polytrode).learning_time=learning_time;
		ENCODED_DATA(polytrode).start_time=start_time;
		disp(['Electrode ',num2str(polytrode),' found empty']);disp(sprintf('\n'));
	end
		

	clearvars -except Reading_result n_events DATA ENCODED_DATA n_events_learning learning_time start_time nb_clusters N_bins Bandwidths Nbin_x Behavior

end

clearvars -except ENCODED_DATA DATA nb_clusters

disp('Encoding completed. Saving ...')

save('EncodedData.mat','ENCODED_DATA','-v7.3');

disp('Workspace was saved.')
whos
