function res=findAllSleepStates(allTimestamps, SleepScoring)
	res = [];
	for i=1:size(allTimestamps,2)
		res = [res; findSleepState(allTimestamps(i), SleepScoring)];
	end