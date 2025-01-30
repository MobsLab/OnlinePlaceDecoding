function data = readDecoding(filename)

	[filepath,name,ext] = fileparts(filename);
	if size(filepath,1)~=0
		filepath = strcat(filepath,'/');
	end

	npyNames = unzip(filename);

	if any(strcmp(npyNames,'arr_0.npy'))
		data.Occupation = readNPY(strcat(filepath,'arr_0.npy'));
		data.position_proba = readNPY(strcat(filepath,'arr_1.npy'));
		data.position = readNPY(strcat(filepath,'arr_2.npy'));
	else
		data.Occupation = readNPY(strcat(filepath,'/Occupation.npy'))
		data.position_proba = readNPY(strcat(filepath,'/position_proba.npy'))
		data.position = readNPY(strcat(filepath,'/position.npy'))
	end

	for i=1:size(npyNames,2)
		test = strcat(filepath,npyNames(i));
		delete(test{1,1});
	end