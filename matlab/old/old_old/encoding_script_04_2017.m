

N=2400000; %240 samples is 10ms
T=N/24;


mesure=importdata(['/home/mobshamilton/Documents/online_place_decoding/simulation/simulation_1.mat']);
RAW=mesure(1:N);

%--- Computing Neo signal
NEO=RAW(1:N-2);
for i = 1:N-2
    NEO(i)=RAW(i+1)*RAW(i+1) - RAW(i+2)*RAW(i);
end


clearvars mesure N
disp([T,' seconds from the datafile loaded']);

%%-------------------------    Spike detection    -------------------------%%

signal=NEO;

%--- Threshold could be computed automatically from the standard deviation of noise
med=median(abs(signal));
stand_dev=med/0.6745;
threshold=25*stand_dev;
% threshold=0.0508;

spike_pos=[];
spike_len=[];
spike_max=[];
spike_cen=[];
spike_val=[];
spike_bool=0;
for i = 1:size(signal,2)
    if (signal(i)>threshold) && (spike_bool==0)
        spike_bool=1;
        spike_pos=[spike_pos i];
        spike_len=[spike_len 1];
    elseif signal(i)>threshold
    	spike_len(end)=spike_len(end)+1;
    elseif spike_bool==1
    	spike_bool=0;
        [val,cen]=max(signal(spike_pos(end):spike_pos(end)+spike_len(end)));
        spike_max=[spike_max val];
        spike_cen=[spike_cen cen];
        spike_val=[spike_val spike_max(end)-min(signal(spike_pos(end):spike_pos(end)+2*spike_len(end)))];
    end
end

clearvars val cen med stand_dev i spike_bool

disp([size(spike_max,2),' spikes detected in the signal']);
clearvars RAW NEO signal


%%-------------------------    Abstract Building    -------------------------%%


% 'building kernel density function'
% Ha=1;
% Hx=1;
% f_gauss=build_all_gauss_kernels(spike_max,spike_val,spike_len,spike_cen,spike_pos,Ha,Hx);
% 'building epanechnikov kernels'
% f_epanech=build_all_epanech_kernels(spike_max,spike_val,spike_len,spike_cen,spike_pos,Ha,Hx);

% This part is currently not converging.
% Maybe we should try again without a recursive method ?






%%-------------------------    Grid Building    -------------------------%%

%--- Initialization of variables
Ha=threshold*3;
Hx=spike_pos(end)/15;
Bandwidths=[Ha Ha Ha Ha Hx];
Nbin_a=50;
Nbin_x=100;
N_bins=[Nbin_a Nbin_a Nbin_a Nbin_a Nbin_x];
list_events=[spike_max; spike_val; spike_max; spike_val; spike_pos];
N_events=size(list_events,2);

clearvars Nbin_a Nbin_x Ha Hx
%--- Computation of our different Kernel matrices
[Rate_function_matrix, Marginal_rate_function_matrix] = bayesian_encoding(list_events, N_bins, Bandwidths,T);
Rate_function_matrix=Rate_function_matrix+0.0001;
two_first_bins=compute_two_first_bins(list_events, N_bins);

disp(['rate function matrices built on a ',N_bins(1),'x',N_bins(2),'x',N_bins(3),'x',N_bins(4),'x',N_bins(5),' grid']);