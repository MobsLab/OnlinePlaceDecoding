function data_reduction(varargin)

%% Initialization of variables
if nargin==0
	maxdim=4;
elseif nargin==1
	maxdim=varargin{1};
else
	maxdim=varargin{1};
	warning(['Only one argument needed, number of dimensions set to ',num2str(maxdim)]);
end


%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%
try isstr(DATA);
catch 
	try
		DATA=load('DATA.mat','DATA');
		DATA=DATA.DATA;
		nb_clusters=load('DATA.mat','nb_clusters');
		nb_clusters=nb_clusters.nb_clusters;
	catch
		error('Impossible to load DATA.mat, have you used the reading script ?');
	end
end




n_polytrode=size(nb_clusters,2);
for polytrode=1:n_polytrode
	ndim=size(DATA(polytrode).events,1)-2;

	if ndim>maxdim

		mean_value=[];
		deviation=[];
		for dim=1:ndim
			mean_value=[mean_value mean(DATA(polytrode).events(dim,:))];
			deviation=[deviation mean((DATA(polytrode).events(dim,:)-mean(DATA(polytrode).events(dim,:))).^2)];
		end

		best_dim_count=zeros(1,ndim);
		for event=1:size(DATA(polytrode).events,2)
			features=DATA(polytrode).events(1:end-2,event);
            features=features';
			strength=(mean_value(:)-features(:))./deviation(:);
			[max_strength,best_dim]=max(strength);
			best_dim_count(best_dim)=best_dim_count(best_dim)+1;
		end


		[sorted_dimentionss,sorting]=sort(best_dim_count);
		sorting=[sorting ndim+1 ndim+2];

		DATA(polytrode).events=DATA(polytrode).events(sorting(end-maxdim-2+1:end),:);
		disp(['Polytrode ',num2str(polytrode),' optimized to ',num2str(maxdim),'-trode.']);
	else
		disp(['Polytrode ',num2str(polytrode),' already has ',max(0,num2str(ndim)),' dimensions.']);
	end

end


clearvars -except DATA nb_clusters
disp(sprintf('\n'));
disp('Saving ...')
save('DATA_reduced.mat','-v7.3');
disp('DATA_reduced.mat has been saved');
disp(sprintf('\n'));