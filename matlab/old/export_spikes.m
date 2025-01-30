function export_spikes(target, split, whichsplit)
	%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%
	Behavior=importdata('behavResources.mat');
	disp('Data Loaded.')



	%%%%%%%%%%%--- BUILD nnBeahavior.mat ---%%%%%%%%%%%
	disp(['target: ', target]);
	if target=='pos'
		X = Data(Behavior.("Xtsd"));
		Y = Data(Behavior.("Ytsd"));
        V = Data(Behavior.("Vtsd"));
		behavior.positions     = [X Y];
		behavior.position_time = Range(Behavior.Xtsd)/10000;
        behavior.speed = V;
	else
		behavior.positions     = Data(Behavior.(target));
		behavior.position_time = Range(Behavior.(target))/10000;
	end

    if isfield(Behavior, 'trainEpochs') && isfield(Behavior, 'testEpochs')
        temp = [Start(Behavior.trainEpochs) Stop(Behavior.trainEpochs)]'
        behavior.trainEpochs = temp(:)/10000;
        temp = [Start(Behavior.testEpochs) Stop(Behavior.testEpochs)]'
        behavior.testEpochs = temp(:)/10000;
    else
        if strcmp(whichsplit, 'end')
            trainratio = 1 - str2double(split);
            cut = floor(size(behavior.position_time, 1)*trainratio);
            behavior.trainEpochs = [behavior.position_time(1); behavior.position_time(cut)];
            behavior.testEpochs  = [behavior.position_time(cut); behavior.position_time(end)];
        elseif strcmp(whichsplit, 'beg')
            trainratio = str2double(split);
            cut = floor(size(behavior.position_time, 1)*trainratio);
            behavior.testEpochs = [behavior.position_time(1); behavior.position_time(cut)];
            behavior.trainEpochs  = [behavior.position_time(cut); behavior.position_time(end)];
        end
    end

	save('nnBehavior.mat','behavior','-v7.3');
	disp('Behavior data extracted and saved.');





	disp(['start_time : ', num2str(min(behavior.position_time)), ' sec']);
	disp(['end_time : ', num2str(max(behavior.position_time)), ' sec']);
end