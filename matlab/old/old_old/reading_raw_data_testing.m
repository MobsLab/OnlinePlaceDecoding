Reading_result={};

currentDirectory=pwd;
[upperPath,deepestFolder,~]=fileparts(currentDirectory);
Folder=currentDirectory(size(upperPath,2)+2:end);
clearvars currentDirectory upperPath deepestFolder

for polytrode=1:1

	data_id = fopen([Folder,'.dat'], 'r');
	spk_id = fopen([Folder,'.spk.',num2str(polytrode)], 'r');
	res_id = fopen([Folder,'.res.',num2str(polytrode)], 'r');
	lfp_id = fopen([Folder,'.lfp'], 'r');
	place_id = fopen([Folder,'.whl'], 'r');
	n_events=100000;
	n_channels=8;


	n_sample=5000;
	spikes=[];
	list_events=[];
	spike_pos=fscanf(res_id,'%i');
	places=fscanf(place_id,'%f %f %f %f',[4 Inf]);
	spike=fread(spk_id,[n_channels 32],'int16');
	data_raw=fread(data_id,55*n_sample,'int16');

	data=zeros(55,n_sample);
	for channel=1:55
		data(channel,:)=data_raw(channel:55:end);
	end


% 	%%%%%%%%%%%%%%% ---------  Reading the entries

% 	spikes=[];
% 	list_events=[];
% 	spike_pos=fscanf(res_id,'%i');
% 	places=fscanf(place_id,'%f %f %f %f',[4 Inf]);

	for event = 1:n_events
		% spikes=cat(3, spikes, fread(spk_id,[n_channels 32],'int16'));
		spike=fread(spk_id,[n_channels 32],'int16');
		features=zeros(n_channels,1);
		for channel = 1:n_channels
			features(channel)=spike(channel,15);
		end
		time_stamp=floor(spike_pos(event)*39.06/20000)+1;
		if (places(1,time_stamp)~=-1) && (places(2,time_stamp)~=-1) && (places(3,time_stamp)~=-1) && (places(4,time_stamp)~=-1)
			posX=(places(1,time_stamp)+places(3,time_stamp))/2;
			posY=(places(2,time_stamp)+places(4,time_stamp))/2;
			list_events=[list_events [features;posX;posY]];
		elseif (places(1,time_stamp)~=-1) && (places(2,time_stamp)~=-1)
			posX=places(1,time_stamp);
			posY=places(2,time_stamp);
			list_events=[list_events [features;posX;posY]];
		elseif (places(3,time_stamp)~=-1) && (places(4,time_stamp)~=-1)
			posX=places(3,time_stamp);
			posY=places(4,time_stamp);
			list_events=[list_events [features;posX;posY]];
		end

		if mod(10*event,n_events)==0
			disp(['Reading electrode number ',num2str(polytrode),', ', num2str(100*event/n_events), '% achieved']);
		end
	end

% 	try
% 		places=places(:,1:floor(spike_pos(end)*39.06/20000)+10);%%%BAD WAY TO DEAL WITH OUR ISSUE (file too big)
% 	catch
% 		warning(['places untouched for electrode ',num2str(polytrode)]);
% 	end
% 	cleaned_places=[];
% 	for sample=1:size(places,2)
% 		if (places(1,sample)~=-1) && (places(2,sample)~=-1) && (places(3,sample)~=-1) && (places(4,sample)~=-1)
% 			cleaned_places=[cleaned_places [(places(1,sample)+places(3,sample))/2;(places(2,sample)+places(4,sample))/2]];
% 		elseif (places(1,sample)~=-1) && (places(2,sample)~=-1)
% 			cleaned_places=[cleaned_places [places(1,sample);places(2,sample)]];
% 		elseif (places(3,sample)~=-1) && (places(4,sample)~=-1)
% 			cleaned_places=[cleaned_places [places(3,sample);places(4,sample)]];
% 		end
% 	end



% 	fclose(spk_id);
% 	fclose(lfp_id);
% 	fclose(data_id);
% 	fclose(res_id);
% 	fclose(place_id);

% 	%% I do not understand the structure of this file, but need it to extract the position of the mouse during a given event
% 	% led_id = fopen('/home/mobshamilton/Documents/online_place_decoding/dataset/ec014.793/ec014.793.led', 'r');

% 	% led=fread(led_id,500,'int16');

% 	% fclose(led_id);

% 	% Reading_result=[Reading_result {list_events;spike_pos;places;cleaned_places}];
% 	DATA(polytrode).events=list_events;
% 	DATA(polytrode).spikes=spike_pos;
% 	DATA(polytrode).positions=places;
% 	DATA(polytrode).cleaned_positions=cleaned_places;
% 	clearvars -except Reading_result n_events DATA
end
figure;
for channel=1:n_channels
	plot(data(channel,:)-(channel-1)*40);hold on;
end
spk=1;
while spike_pos(spk)<n_sample
	vline(spike_pos(spk),'k');
	spk=spk+1;
end
xlim([0 1000]);


disp('Reading completed.')
% clearvars -except Reading_result DATA



