function res=findSleepState(time, SleepScoring)
	i = 1;
	while SleepScoring.timestamps(i) < time
		i = i + 1;
		if i == size(SleepScoring.timestamps,2)
			break
		end
	end
	if i==1
		i=2;
	end
	res = SleepScoring.text(i-1,:);