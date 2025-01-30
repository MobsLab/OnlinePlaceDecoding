%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%
try 
	Behavior=importdata('behavResources.mat');
catch
	Behavior=importdata('behavResourcesOPD.mat');
end
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
disp('Behavior data extracted and saved.');





disp(['start_time : ', num2str(min(behavior.position_time)), ' sec']);
disp(['end_time : ', num2str(max(behavior.position_time)), ' sec']);