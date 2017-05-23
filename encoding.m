
%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%
try isstr(DATA);
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
learning_time=max_time/2;






%% Initialization of variables
Ha=100;
Hx=30;
Bandwidths=[Ha Ha Ha Ha Hx Hx];
Nbin_a=25;
Nbin_x=30;
N_bins=[Nbin_a Nbin_a Nbin_a Nbin_a Nbin_x Nbin_x];

Encoded_data={};
for polytrode=1:n_polytrode
	disp(['Starting electrode ',num2str(polytrode),'.']);
	n_events=size(DATA(polytrode).events,2);

	%%%%%%%%%%%--- ENCODING EVENTS ---%%%%%%%%%%%

	if n_events~=0
		n_events_learning=1;
		%%-- Determines the number of events to learn from.
		while DATA(polytrode).spikes(n_events_learning)<learning_time
			if n_events_learning==n_events
				break
			end
			n_events_learning=n_events_learning+1;
		end

		T=DATA(polytrode).spikes(n_events_learning)/19531;

		
		clearvars Nbin_a Nbin_x Ha Hx posX posY channel data_id event led_id lfp_id place_id res_id spike spikes spk_id time_stamp

		%--- Computation of our different Kernel matrices, this is the actual encoding, right there.
		[Rate_function_matrix, Marginal_rate_function_matrix, Occupation_matrix] = bayesian_encoding(DATA(polytrode).events(:,1:n_events_learning), DATA(polytrode).cleaned_positions, N_bins, Bandwidths,T);
		two_first_bins=compute_two_first_bins(DATA(polytrode).events(:,1:n_events_learning), N_bins);

		%--- Stocking our results.
		ENCODED_DATA(polytrode).Rate_Function=Rate_function_matrix;
		ENCODED_DATA(polytrode).Marginal_Rate_Function=Marginal_rate_function_matrix;
		ENCODED_DATA(polytrode).Occupation=Occupation_matrix;
		ENCODED_DATA(polytrode).twofirstbins=two_first_bins;
		ENCODED_DATA(polytrode).n_events_learning=n_events_learning;
		ENCODED_DATA(polytrode).learning_time=learning_time;

	else
		ENCODED_DATA(polytrode).Rate_Function=ones(N_bins);
		ENCODED_DATA(polytrode).Marginal_Rate_Function=normalise_deg1(ones(N_bins(end-1:end)));
		ENCODED_DATA(polytrode).Occupation=normalise_deg1(ones(N_bins(end-1:end)));
		ENCODED_DATA(polytrode).twofirstbins=[zeros(size(N_bins,2),1) ones(size(N_bins,2),1)];
		ENCODED_DATA(polytrode).n_events_learning=0;
		ENCODED_DATA(polytrode).learning_time=learning_time;
		disp(['Electrode ',num2str(polytrode),' found empty']);
	end
		

	disp(['Electrode ',num2str(polytrode),' encoded.']);disp(sprintf('\n'));
	clearvars -except Reading_result Encoded_data n_events DATA ENCODED_DATA n_events_learning learning_time nb_clusters N_bins Bandwidths

end

clearvars -except ENCODED_DATA DATA nb_clusters

disp('Encoding completed. Saving ...')

save('EncodedData.mat','ENCODED_DATA','-v7.3');

disp('Workspace was saved.')
whos
