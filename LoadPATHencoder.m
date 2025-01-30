% created from LoadPATHhamilton


function LoadPATHencoder=LoadPATHencoder(username)

	res=pwd;
	
	cd(strcat('/home/',username,'/Dropbox/Kteam'))
	addpath(genpath(strcat('/home/',username,'/Dropbox/Kteam/PrgMatlab')))
	addpath(genpath(strcat('/home/',username,'/Dropbox/Kteam/PrgMatlab/MatFilesMarie')))
	eval(['cd(''',res,''')'])

	clear res

	cmap = colormap('jet');
	set(groot,'DefaultFigureColorMap',cmap);
	close all
	clear cmap

	set(0,'DefaultFigureWindowStyle','docked')

end
