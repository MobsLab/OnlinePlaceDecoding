%importThresholds
% 01.07.2019 t.balenbois@gmail.com
%
% thresholdData = importThresholds()
% thresholdData = importThresholds('/path/to/folder')
% thresholdData = importThresholds('/path/to/folder', 'B')
% thresholdData = importThresholds('/path/to/folder', false)
% 
% This function returns information about thresholds and boxes that
% were used during a recording session with the MOBSinterface software.
%
%
%INPUT
%  folderPath              : (optional) path to the folder where 'thresholds_portA.txt' is. 
%                                 default pwd
%  port                    : (optional) port the mouse was plugged in. 
%                                 default all ports
%  withBoxes               : (optional) checks for boxes in the file. 
%                                 default true
%
%
%OUTPUT
% thresholdData            : nx1 struct where n is the number of channel that where checked during the recording.
%                                 the fields names are transparents.
%
%
%ASSUMPTIONS
%  expects a file of the following format:
%  group ; channel ; threshold ; firstbox X ; firstbox Y ; firstBox size X ; firstBox sizeY ; secondBox X....
%  so the number of fields per line is 3 + (nBoxes*4)
%

function thresholdData = importThresholds(folderPath, port, withBoxes)

	if ~exist('folderPath','var')
		folderPath=pwd;
	end
	if ~exist('port','var')
		port = ['A' 'B' 'C' 'D'];
	end
	if ~exist('withBoxes','var')
		withBoxes=true;
	end

	for i=1:size(port,2)
		mousePort = port(i);
		fileName = strcat(folderPath, '/thresholds_port', mousePort, '.txt');
		if ~exist(fileName, 'file')
			continue
		end
		file = fopen(fileName);

		lineStr = fgetl(file);
		thresholdData=[];
		while lineStr~=(-1)
			data = strsplit(lineStr,';')
			data
			channelData.channel = str2num(data(2));
			channelData.threshold = str2num(data(3));
			channelData.boxes = getBoxes(data, withBoxes);

			thresholdData = [thresholdData; channelData];
			lineStr = fgetl(file);
		end

	end
end

function boxes = getBoxes(data, withBoxes);
	boxes = [];
	nBox = (size(data,2)-3)/4;
	if ~withBoxes
		return
	end
	if nBox==0
		return
	end

	idx = 4;
	while nBox > 0
		boxData.time = data(idx);
		boxData.voltage = data(idx+1);
		boxData.timeWidth = data(idx+2);
		boxData.voltageWidth = data(idx+3);

		boxes = [boxes; boxData];
		nBox = nBox - 1;
		idx = idx + 4;
	end
end