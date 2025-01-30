
content = dir;

for i=1:size(content,1)
	if content(i).isdir==1
		continue;
	end

	if content(i).name(1:4)=='Rhyt'
		IntanTTL = load(content(i).name);
		IntanTTL.ups_timestamps = [];
		IntanTTL.downs_timestamps = [];
		for i=1:size(IntanTTL.timestamps,2)
			if IntanTTL.channel_states(i)>0
				IntanTTL.ups_timestamps = [IntanTTL.ups_timestamps IntanTTL.timestamps(i)];
			else
				IntanTTL.downs_timestamps = [IntanTTL.downs_timestamps IntanTTL.timestamps(i)];
			end
		end
	elseif content(i).name(1:4)=='Slee'
		SleepScoring = load(content(i).name);
	elseif content(i).name(1:4)=='Spk_'
		StimTTL = load(content(i).name);
		StimTTL.single_timestamps = StimTTL.timestamps(1:2:end);
		StimTTL.nStim = size(StimTTL.single_timestamps, 2);
	elseif content(i).name(1:4)=='Spik'
		Spikes = load(content(i).name);
	end
end




if exist('SleepScoring','var')==1
	StimTTL.sleep_states = findAllSleepStates(StimTTL.timestamps, SleepScoring);
	IntanTTL.sleep_states = findAllSleepStates(IntanTTL.ups_timestamps, SleepScoring);
	Spikes.sleep_states = findAllSleepStates(Spikes.spike_times, SleepScoring);
end