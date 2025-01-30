% getSleepScoringThresholds
% 29.05.2019 t.balenbois@gmail.com
%
% [gammaThreshold thetaThreshold] = getSleepScoringThresholds('sleepScoring.txt', 'A')
% 
% This function returns the sleep scoring thresholds that separates the two distributions of gamma and theta
%
%
%INPUT
%  theta_signal      : fileName (with path if needed) of the 'sleepScoring.txt' type file created by the MOBSinterface version of INTAN
%  port              : (optional) select a specific port. Useful if recording several mice at the same time
%
%
%OUTPUT
%  gammaThresh       : threshold that distinguish the two gaussian on the gamma signal
%  theta_thresh      : threshold that distinguish the two gaussian on the theta/delta signal
%
%
%ASSUMPTIONS
%  Relies on GetGammaThresh.m and GetThetaThresh.m
%


function [gammaThresh,thetaThresh] = getSleepScoringThresholds(fileName, port)

	[allPorts length score gamma theta] = importSleepScoring(fileName);
	allPorts = cell2mat(allPorts);

	% if several recordings, user needs to specifiy the mouse's port
	if nargin > 1
		if port~='A' && port~='B' && port~='C' && port~='D'
			error('Port argument not understood. Must be ''A'' or ''B'' or ''C'' or ''D''')
		end
		selection = allPorts==port;
		if sum(selection)==0
			error(strcat('No Data from this port: ',port))
		end
		gamma = gamma(selection);
		theta = theta(selection);
	end

	% may not be useful anymore, but what are three points anyway
	gamma = gamma(4:end,:);
	theta = theta(4:end,:);

	gammaThresh = GetGammaThresh(gamma);
	thetaThresh = GetThetaThresh(log(theta));

	fMain = figure; fMain.Name = 'Hypnogram'
	plot(log(gamma), log(theta), '.b');
	vline(gammaThresh, 'k');
	hline(thetaThresh, 'k');

	gammaThresh = exp(gammaThresh);
	thetaThresh = exp(thetaThresh);
	
	% Updating some figures if threshold is selected by user
	fGamma = findobj('Type','Figure','Number',fMain.Number-2); fGamma.Name = 'Gamma'
	fTheta = findobj('Type','Figure','Number',fMain.Number-1); fTheta.Name = 'Theta'
	fGamma.Children(2).Children(1).XData = [log(gammaThresh) log(gammaThresh)]
	fTheta.Children(2).Children(1).XData = [log(thetaThresh) log(thetaThresh)]