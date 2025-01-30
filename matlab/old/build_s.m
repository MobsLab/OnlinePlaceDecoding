%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%
% Behavior=importdata('behavResources.mat');
SpikeData=importdata('SpikeData.mat');
% Waveforms=importdata('Waveforms.mat');
% disp('Data Loaded.')







disp('Building s variable.')

%%%%%%%%%%%--- READ CLUSTERS INFO ---%%%%%%%%%%%
% cellnames=Waveforms.cellnames;
cellnames=SpikeData.cellnames;
n_tetrode=1;
nb_clusters(1)=0;
clusters{1}=[];
for cluster=1:size(cellnames,2)
	info = sscanf(cellnames{1,cluster},strcat('TT','%d','c','%d'));
	if info(1)==n_tetrode
		nb_clusters(n_tetrode)=nb_clusters(n_tetrode)+1;
		clusters{n_tetrode}=[clusters{n_tetrode} info(2)];
	else
		n_tetrode=info(1);
		nb_clusters(n_tetrode)=1;
		clusters{n_tetrode}=[];
		clusters{n_tetrode}=[clusters{n_tetrode} info(2)];
	end
end



s=[];
cluster_cursor = 0;
for tetrode = 1:size(clusters, 2)
	tetrode_s = [];

	for loc_cluster = 1:size(clusters{tetrode},2)
		cluster_cursor = cluster_cursor + 1;
		spike_time = Range(SpikeData.S{1,cluster_cursor});
		local_s = [spike_time./10000 ones(size(spike_time,1),1).*tetrode ones(size(spike_time,1),1).*loc_cluster];

		tetrode_s = [tetrode_s; local_s];
		disp(['cluster ',num2str(loc_cluster),' done']);
	end


	s = [s; sortrows(tetrode_s)];
	disp(['tetrode ',num2str(tetrode),' done'])
end

SpikeData.s = s;