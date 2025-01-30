%%-- This is automatic to a certain extent. In order for it to work, the value of the systematic variable must be at the very last of the folder name.


folders=dir();

systematic_variable=[];
mean_pvalue_all=[];
mean_pvalue_lowSigma_all=[];
nb_lowSigma_points_all=[];

for i=3:size(folders,1)
	if ~folders(i).isdir
		continue
	end
	cd(folders(i).name);

	digits_char=isstrprop(folders(i).name,'digit');
	j=0;
	while digits_char(end-j)
		j=j+1;
	end
	systematic_variable=[systematic_variable str2num(folders(i).name(end-j+1:end))];

	disp(sprintf('\n'));
	disp(['Reading folder : ',folders(i).name]);
	plot_position
	close all

	mean_pvalue_all=[mean_pvalue_all mean_pvalue];
	mean_pvalue_lowSigma_all=[mean_pvalue_lowSigma_all mean_pvalue_lowSigma'];
	nb_lowSigma_points_all=[nb_lowSigma_points_all nb_lowSigma_points'];

	clearvars -except mean_pvalue_all mean_pvalue_lowSigma_all nb_lowSigma_points_all folders systematic_variable
	cd('..');
end

[systematic_variable, SortIndex]=sort(systematic_variable);

for cut=1:size(mean_pvalue_lowSigma_all,1)
	plot(systematic_variable,mean_pvalue_all(SortIndex)-mean_pvalue_lowSigma_all(cut,SortIndex),'DisplayName',['cut in standard deviation : ',num2str(cut)]); hold on;
end
legend('show');
xlabel('Systematic variable'); ylabel('Difference between mean p-value of all points and of selected points');
title('Difference between the mean p-value for all points and of selected points, for different bandwidths');

figure;
for cut=1:size(mean_pvalue_lowSigma_all,1)
	plot(systematic_variable, mean_pvalue_lowSigma_all(cut,SortIndex));hold on;
end