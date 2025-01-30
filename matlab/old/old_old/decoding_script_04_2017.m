N=2400000; %240 samples is 10ms
T=N/24;

bin_size=1280;
time_bin=bin_size/24;
threshold=0.0508;


mesure=importdata(['/home/mobsyoda/Documents/database_onlineplacedecoding/simulation/simulation_1.mat']);
RAW=mesure(1:N);

%--- Computing Neo signal
NEO=RAW(1:N-2);
for i = 1:N-2
    NEO(i)=RAW(i+1)*RAW(i+1) - RAW(i+2)*RAW(i);
end

disp('data file loaded');




%%-------------------------    Bin cutting    -------------------------%%

signal=NEO;

N_bins=floor(size(signal,2)/bin_size);
bins=[];
for i=1:N_bins
	bins=[bins; signal((i-1)*bin_size+1:i*bin_size)];
end

disp('Bins_cutted');

clearvars NEO RAW






%%-------------------------    Bin decoding    -------------------------%%
position_estimates=[];
interesting_bins=[];

for j=1:size(bins,1)

	signal=bins(j,:);

	spike_len=[];
	spike_max=[];
	spike_cen=[];
	spike_val=[];
	spike_bool=0;
	for i = 1:size(signal,2)
	    if (signal(i)>threshold) && (spike_bool==0)
	        spike_bool=1;
	        interesting_bins=[interesting_bins j];
	        spike_pos=[spike_pos i];
	        spike_len=[spike_len 1];
	    elseif signal(i)>threshold
	    	spike_len(end)=spike_len(end)+1;
	    elseif spike_bool==1
	    	spike_bool=0;
	        [val,cen]=max(signal(spike_pos(end):end));
	        spike_max=[spike_max val];
	    	end_valley=min(size(signal,2),spike_pos(end)+2*spike_len(end));
	        spike_val=[spike_val spike_max(end)-min(signal(spike_pos(end):end_valley))];
	    end
	end

	spike_events=[spike_max; spike_val; spike_max; spike_val];




	prior_estimate=(ones(size(Marginal_rate_function_matrix,2))).';
	current_event_likelihood=event_likelihood(spike_events, time_bin, Rate_function_matrix, Marginal_rate_function_matrix, two_first_bins);
	position_estimates=[position_estimates; position_density_proba(current_event_likelihood, prior_estimate)];

end

clearvars val cen med stand_dev i j spike_bool end_valley signal
clearvars current_event_likelihood prior_estimate spike_cen spike_events spike_len spike_max spike_val
