%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%
try 
	Behavior=importdata('behavResources.mat');
catch
	Behavior=importdata('behavResourcesOPD.mat');
end
SpikeData=importdata('SpikeData.mat');
Waveforms=importdata('Waveforms.mat');
disp('Data Loaded.')



%%%%%%%%%%%--- BUILD nnBeahavior.mat ---%%%%%%%%%%%
X = Data(Behavior.Xtsd);
Y = Data(Behavior.Ytsd);
behavior.positions     = [X Y];
behavior.positions     = behavior.positions(1:end-1,:);
behavior.speed         = Data(Behavior.Vtsd);
behavior.position_time = Range(Behavior.Xtsd)/10000;
behavior.position_time = behavior.position_time(1:end-1);

Behavior.Pos           = [behavior.position_time behavior.positions];
Behavior.Speed         = behavior.speed;

save('nnBehavior.mat','behavior','-v7.3');
clearvars -except Behavior SpikeData Waveforms
disp('Behavior data extracted and saved.');





%%%%%%%%%%%--- READ CLUSTERS INFO ---%%%%%%%%%%%
cellnames=Waveforms.cellnames;
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

try isstr(s);
catch
	try
		build_s;
	catch
		error('Missing s table.');
	end
end

% pause;

n_tetrodes = size(nb_clusters, 2);
cursor = 1;
for polytrode = 1:size(nb_clusters, 2)
	if nb_clusters(polytrode)~=0

		spike_cursor = 1;
		clusters_cursors=ones(clusters{polytrode}(end),1);
		tetrode = Waveforms.W(1,sum(nb_clusters(1,1:(polytrode-1)))+1:sum(nb_clusters(1,1:polytrode)));

		clearvars spikes
		spikes(:,:,:)=tetrode{1,1};
		for i =2:size(tetrode,2)
			spikes=cat(1,spikes,tetrode{1,i});
		end

		labels=zeros(size(spikes,1),size(tetrode,2));
		% j=1;
		% for i = 1:size(tetrode,2)
		% 	for k = 1:size(tetrode{1,i},1)
		% 		labels(j,i) = 1;
		% 		j = j+1;
		% 	end
		% end

		spike_time = zeros(size(spikes,1),1);
		positions = zeros(size(spikes,1),2);
		speed = zeros(size(spikes,1),1);
		while (SpikeData.s(cursor,2) == polytrode)
			if SpikeData.s(cursor,3) ~= 0
				spikes(spike_cursor,:,:) = tetrode{1,SpikeData.s(cursor,3)}(clusters_cursors(SpikeData.s(cursor,3)),:,:);
				labels(spike_cursor, SpikeData.s(cursor,3)) = 1;
				spike_time(spike_cursor) = SpikeData.s(cursor,1);
				[time_difference index]=min(abs(Behavior.Pos(:,1)-SpikeData.s(cursor,1)));
				try
					positions(spike_cursor,:) = [Behavior.Pos(index,2) Behavior.Pos(index,3)];
					speed(spike_cursor) = Behavior.Speed(index);
				catch
					positions(spike_cursor,:) = [-1 -1];
					speed(spike_cursor) = -1;
				end
				spike_cursor = spike_cursor + 1;
				clusters_cursors(SpikeData.s(cursor,3)) = clusters_cursors(SpikeData.s(cursor,3)) + 1 ;
			end
			cursor = cursor + 1;

			if mod(cursor,1000) == 0
				disp(['Still here. Still calculating. ', num2str(cursor/size(SpikeData.s, 1)*100),' %']);
			end

			if cursor == size(SpikeData.s, 1)
				break
			end
		end

		nn_spikes.spikes = spikes;
		nn_spikes.labels = labels;
		nn_spikes.spike_time = spike_time;
		nn_spikes.positions = positions;
		nn_spikes.speed = speed;
	

		disp(['Tetrode ',num2str(polytrode),' extracted.']);
		save(['nnSpikes_t',num2str(polytrode)], 'nn_spikes', '-v7.3')
		disp(['Tetrode ',num2str(polytrode),' saved.']);
	else
		disp(['Tetrode ',num2str(polytrode),' empty.'])
	end
end


hist(Behavior.Speed, 100);
disp(['n_groups : ', num2str(n_tetrodes)]);
disp(['start_time : ', num2str(min(behavior.position_time)), ' sec']);
disp(['end_time : ', num2str(max(behavior.position_time)), ' sec']);