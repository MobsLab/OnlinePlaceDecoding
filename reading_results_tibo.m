% cd '/home/mobshamilton/Documents/online_place_decoding/dataset/RatCatanese'

%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%
Behavior=importdata('behavResources.mat');
SpikeData=importdata('SpikeData.mat');
Waveforms=importdata('Waveforms.mat');
disp('Data Loaded.')









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

% for tetrode=1:size(nb_clusters,2)
% 	clusters{tetrode}=(1:max(clusters{tetrode}));
% end

clearvars cluster tetrode info n_tetrode








%%%%%%%%%%%--- ERASE ARTEFACTS FROM POSITION ---%%%%%%%%%%%
places=Behavior.Pos(1:end-1,[2;3]);
places=[places Behavior.Speed];
cleaned_x=places(places(:,3)>9,1)';
cleaned_y=places(places(:,3)>9,2)';
cleaned_x=cleaned_x(cleaned_x~=-1);
cleaned_y=cleaned_y(cleaned_y~=-1);
if size(cleaned_x)==size(cleaned_y)
	cleaned_places=[cleaned_x;cleaned_y];
else
	error('Some problem has occured when cleaning positions');
end








%%%%%%%%%%%--- READ FEATURES PER TETRODE ---%%%%%%%%%%%
%% Several cursors are needed to know at each step what we have already read, in each cluster and overall
cursor=1;
for tetrode=1:size(nb_clusters,2)
	list_events=[];
	buffer_events=[];
	spike_pos=[];
	buffer_pos=[];
	unclassified_spikes=[];
	clusters_cursors=ones(clusters{tetrode}(end),1);



	%% Reading data of a given tetrode
	while SpikeData.s(cursor,2)==tetrode



		if SpikeData.s(cursor,3)==0
			%% This is cluster 0. This the trash. We don't use the trash.
		else
			features=Waveforms.W{1,sum(nb_clusters(1:tetrode-1))+find(clusters{tetrode}==SpikeData.s(cursor,3))}(clusters_cursors(SpikeData.s(cursor,3)),(1:end)',15);
			try
				posX=Behavior.Pos(floor(SpikeData.s(cursor,1)/(Behavior.Pos(end,1)/size(Behavior.Pos,1))),2);
				posY=Behavior.Pos(floor(SpikeData.s(cursor,1)/(Behavior.Pos(end,1)/size(Behavior.Pos,1))),3);
				%% We Select data where speed is at least 9 (see speed_plots)
				if Behavior.Speed(floor(SpikeData.s(cursor,1)/(Behavior.Pos(end,1)/size(Behavior.Pos,1))))>9
					buffer_events=[buffer_events [features';posX;posY]];
					buffer_pos=[buffer_pos SpikeData.s(cursor,1)];
				end
			catch
				%% Spikes are unclassified when video informations are lacking
				unclassified_spikes=[unclassified_spikes cursor];
			end
			clusters_cursors(SpikeData.s(cursor,3))=clusters_cursors(SpikeData.s(cursor,3))+1;
		end
		


		%% We use a buffer to reduce reading time. Here we empty it in the definitive table.
		if size(buffer_events,2)>20000
			list_events=[list_events buffer_events];
			buffer_events=[];
			spike_pos=[spike_pos buffer_pos];
			buffer_pos=[];
			disp('Buffer emptied');
		end



		%% From time to time, we give an update
		if mod(cursor,floor(size(SpikeData.s,1)/1000))==0
			disp(['Reading electrode number ',num2str(tetrode),', ', num2str(100*cursor/size(SpikeData.s,1)), '% of total achieved']);
		end

		%% We deal with arriving to the very end of the file
		if cursor~=size(SpikeData.s,1)
			cursor=cursor+1;
		else
			disp(['Reading electrode number ',num2str(tetrode),', 100% of total achieved'])
			break
		end
	end
	list_events=[list_events buffer_events];
	spike_pos=[spike_pos buffer_pos];



	%% Stocking Data, update infos.
	DATA(tetrode).events=list_events;
	DATA(tetrode).spikes=spike_pos;
	DATA(tetrode).positions=places;
	DATA(tetrode).cleaned_positions=cleaned_places;
	disp(['Electrode number ',num2str(tetrode),' had ',num2str(size(spike_pos,2)),' elements.'])
	if size(unclassified_spikes,2)~=0
		disp([num2str(size(unclassified_spikes,2)),' spikes have not been classified, due to lack of video information.']);
	end
end

clearvars -except DATA nb_clusters
disp('Saving ...')
save('DATA.mat','-v7.3');
disp('DATA.mat has been saved');
disp(sprintf('\n'));