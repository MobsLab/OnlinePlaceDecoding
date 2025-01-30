function [MousePort,WindowLength,SleepScoring,GammaPower,ThetaPowerNorm] = importSleepScoring(filename, startRow, endRow)
%IMPORTSLEEPSCORING Import numeric data from a text file as column vectors.
%   [MOUSEPORT,WINDOWLENGTH,SLEEPSCORING,GAMMAPOWER,THETAPOWERNORM] =
%   IMPORTSLEEPSCORING(FILENAME) Reads data from text file FILENAME for the default
%   selection.
%
%   [MOUSEPORT,WINDOWLENGTH,SLEEPSCORING,GAMMAPOWER,THETAPOWERNORM] =
%   IMPORTSLEEPSCORING(FILENAME, STARTROW, ENDROW) Reads data from rows STARTROW
%   through ENDROW of text file FILENAME.
%
% Example:
%   [MousePort,WindowLength,SleepScoring,GammaPower,ThetaPowerNorm] = importSleepScoring('sleepScoring.txt',1, 7797);
%
%    See also TEXTSCAN.

% Auto-generated by MATLAB on 2019/05/29 09:58:39

%% Initialize variables.
delimiter = ';';
if nargin<=2
    startRow = 1;
    endRow = inf;
end

%% Format string for each line of text:
%   column1: text (%s)
%	column2: double (%f)
%   column3: text (%s)
%	column4: double (%f)
%   column5: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%f%s%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Allocate imported array to column variable names
MousePort = dataArray{:, 1};
WindowLength = dataArray{:, 2};
SleepScoring = dataArray{:, 3};
GammaPower = dataArray{:, 4};
ThetaPowerNorm = dataArray{:, 5};




