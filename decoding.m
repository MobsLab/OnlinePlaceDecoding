function decoding(varargin)
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

try isstr(ENCODED_DATA);
catch 
	try
		ENCODED_DATA=load('EncodedData.mat','ENCODED_DATA');
		ENCODED_DATA=ENCODED_DATA.ENCODED_DATA;
	catch
		error('Impossible to load EncodedData.mat, have you used the encoding script ?');
	end
end

try isstr(Pos);
catch
	Pos=load('behavResources.mat','Pos');
	Pos=Pos.Pos;
	Speed=load('behavResources.mat','Speed');
	Speed=Speed.Speed;
end

clearvars -except DATA ENCODED_DATA nb_clusters Pos Speed Hx Ha


if nargin==0
	Ha=100;
	Hx=30;
end



%%%%%%%%%%%--- META INFOS ---%%%%%%%%%%%
c=clock;
FileName=[date,'_',num2str(c(4)),':',num2str(c(5))];
mkdir(FileName);
FileName=[FileName,'/'];
logID = fopen([FileName,'decoding_log.txt'],'w');
time_bin=100/1000;
nb_bins=5000*10/10;
fprintf(logID,'Decoding %d bins of a length of %dms.\n',nb_bins,floor(time_bin*1000));








n_polytrode=size(nb_clusters,2);
position_proba=[];
position=[];
spike_rate=[];
n_bin=1;
n_bypass=0;

%%-- The while loop needs to be used, since we will skip points where speed is too low
while size(position_proba,3)<nb_bins

	%%-- Gives infos from time to time
	if mod(n_bin,floor(nb_bins/10))==0
		disp(['Decoding time bin number ',num2str(n_bin),' over ',num2str(nb_bins),' to decode (',num2str(size(position_proba,3)*100/nb_bins),' % achieved, ',num2str(n_bypass),' bypassed).']);
	end

	fprintf(logID,'\n---------BIN %d OF %d----------\n',n_bin,nb_bins);
	All_estimations=[];
	positionX=[];
	positionY=[];
	Mouse_Speed=[];
	n_spike=0;
	for polytrode = 1:n_polytrode
		n_events_learning=ENCODED_DATA(polytrode).n_events_learning;
		learning_time=ENCODED_DATA(polytrode).learning_time;

		%%%%%%%%%%%--- DECODING ---%%%%%%%%%%%
		if n_events_learning~=0
			%--- First we select all the spikes that are going to be in our bin
			bin_events=intersect(find(learning_time+(n_bin-1)*time_bin<DATA(polytrode).spikes),find(DATA(polytrode).spikes<learning_time+n_bin*time_bin));
			fprintf(logID,'%d spikes for electrode %d.\n',size(bin_events,2),polytrode);
			n_spike=[n_spike;size(bin_events,2)];

			%%-- We extract ground truth about position and speed of the mouse
			time=learning_time+(n_bin-0.5)*time_bin;
			[idx idx]=min(abs(Pos(:,1)-time));
			positionX=[positionX Pos(idx,2)];
			positionY=[positionY Pos(idx,3)];
			Mouse_Speed=[Mouse_Speed Speed(idx)];
		
			%%%--- Actual bayesian decoding, right there
			prior_estimate=normalise_deg1((ones(size(ENCODED_DATA(polytrode).Marginal_Rate_Function,1),size(ENCODED_DATA(polytrode).Marginal_Rate_Function,2))));
			current_event_likelihood=event_likelihood(DATA(polytrode).events(1:end-2,bin_events), time_bin, ENCODED_DATA(polytrode).Rate_Function, ENCODED_DATA(polytrode).Marginal_Rate_Function, ENCODED_DATA(polytrode).twofirstbins);
			position_estimates=position_density_proba(current_event_likelihood, prior_estimate);

			%%-- All_estimations contains 14 maps of probability, containing the result from each tetrodes (except empty ones)
			All_estimations=cat(3,All_estimations,position_estimates);
			clearvars position_estimates
		else
			fprintf(logID,'Electrode %d was found empty.\n',polytrode);
		end

	end
	Mouse_Speed=mean(Mouse_Speed);
	positionX=mean(positionX);
	positionY=mean(positionY);
	%%-- Converting position to grid-like coordinates
	positionX=floor((positionX-ENCODED_DATA(1).twofirstbins(end-1,1))/(ENCODED_DATA(1).twofirstbins(end-1,2)-ENCODED_DATA(1).twofirstbins(end-1,1)))+1;
	positionY=floor((positionY-ENCODED_DATA(1).twofirstbins(end,1))/(ENCODED_DATA(1).twofirstbins(end,2)-ENCODED_DATA(1).twofirstbins(end,1)))+1;
	fprintf(logID,'Position read by camera, in grid coordinates : %d %d\n',positionX,positionY);

	if isnan(positionX) || isnan(positionY)
		pause
		continue
	end
	

	
	%-- position_estimates contains the result for this bin
	position_estimates=ones(size(All_estimations,1),size(All_estimations,2));
	for polytrode=1:size(All_estimations,3)
		position_estimates(:,:)=position_estimates(:,:).*All_estimations(:,:,polytrode);
	end

	position_estimates(:,:)=normalise_deg1(position_estimates(:,:));


	if Mouse_Speed>9 && positionX<=size(position_estimates,1) && positionY<=size(position_estimates,2)
		position=[position [positionX;positionY]];
		position_proba=cat(3,position_proba,position_estimates);
		spike_rate=[spike_rate n_spike];
	else
		n_bypass=n_bypass+1;
	end

	n_bin=n_bin+1;
end

disp(['Decoding finished ! We had ',num2str(size(position,2)),' readable bins.']);

cd(FileName);
plot_position

clearvars -except DATA ENCODED_DATA position_proba position nb_clusters R Pos FileName Speed n_bypass nb_bins spike_rate time_bin nb_lowSigma_points mean_pvalue mean_pvalue_lowSigma

save([FileName,'decoding_results_',num2str(time_bin*1000),'ms.mat'],'position_proba','position','spike_rate','FileName','nb_lowSigma_points','mean_pvalue','mean_pvalue_lowSigma','-v7.3');