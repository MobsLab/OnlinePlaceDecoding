try isstr(position_proba);
catch
	try
		file=dir('decoding*.mat');
		decoding_results=importdata(file.name);
		position_proba=decoding_results.position_proba;
		position=decoding_results.position;
		spike_rate=decoding_results.spike_rate;
	catch
		error('Impossible to load decoding results, have you used the decoding script ?');
	end
end
FileName='';



guess_of_X=[];
guess_of_Y=[];
max_X=[];
max_Y=[];
ecartT_x=[];
ecartT_y=[];
p_value=[];
correlations=[];
generalized_error=[];
specific_error=[];
comparison_error=[];
errorX_map=zeros(size(position_proba,1),size(position_proba,2));
errorY_map=zeros(size(position_proba,1),size(position_proba,2));
deviationX_map=zeros(size(position_proba,1),size(position_proba,2));
deviationY_map=zeros(size(position_proba,1),size(position_proba,2));
counts=zeros(size(position_proba,1),size(position_proba,2));
errorX_image=zeros(size(position_proba,1),size(position_proba,2));
errorY_image=zeros(size(position_proba,1),size(position_proba,2));
deviationX_image=zeros(size(position_proba,1),size(position_proba,2));
deviationY_image=zeros(size(position_proba,1),size(position_proba,2));
counts_guess=zeros(size(position_proba,1),size(position_proba,2));
fired_spikes=[];

for n=1:size(position,2)	
	a(:,:)=position_proba(:,:,n);
	p_value=[p_value sum(a(a<a(position(1,n),position(2,n))))];

	%%%--- Method using maximum of probability
	[maxs,X]=max(position_proba(:,:,n));
	[testy,Y]=max(maxs);
	X=X(Y);
	max_X=[max_X X];
	max_Y=[max_Y Y];
	specific_error=[specific_error norm(position([1 2],n)'-[X Y])];

	%%%--- Method using expectancy
	for x=1:size(position_proba,1)
		X_proba(x)=sum(position_proba(x,:,n));
	end
	for y=1:size(position_proba,2)
		Y_proba(y)=sum(position_proba(:,y,n));
	end
	X=sum(X_proba.*(1:size(X_proba,2)));
	Y=sum(Y_proba.*(1:size(Y_proba,2)));
	sigma_X=sqrt(sum(X_proba.*((1:size(X_proba,2)).^2))-X^2);
	sigma_Y=sqrt(sum(Y_proba.*((1:size(Y_proba,2)).^2))-Y^2);




	guess_of_Y=[guess_of_Y Y];
	guess_of_X=[guess_of_X X];
	ecartT_x=[ecartT_x sigma_X];
	ecartT_y=[ecartT_y sigma_Y];

	if (abs(guess_of_X(n)-position(1,n))<2) && (abs(guess_of_Y(n)-position(2,n))<2)
		correlations=[correlations n];
	end
	generalized_error=[generalized_error norm(position([1 2],n)'-[X Y])];
	comparison_error=[comparison_error generalized_error(n)-specific_error(n)];

	errorX_map(position(1,n),position(2,n))=errorX_map(position(1,n),position(2,n))+abs(position(1,n)-X);
	errorY_map(position(1,n),position(2,n))=errorY_map(position(1,n),position(2,n))+abs(position(2,n)-Y);
	deviationX_map(position(1,n),position(2,n))=deviationX_map(position(1,n),position(2,n))+sigma_X;
	deviationY_map(position(1,n),position(2,n))=deviationY_map(position(1,n),position(2,n))+sigma_Y;
	counts(position(1,n),position(2,n))=counts(position(1,n),position(2,n))+1;

	errorX_image(floor(guess_of_X(n)),floor(guess_of_Y(n)))=errorX_image(floor(guess_of_X(n)),floor(guess_of_Y(n)))+abs(position(1,n)-X);
	errorY_image(floor(guess_of_X(n)),floor(guess_of_Y(n)))=errorY_image(floor(guess_of_X(n)),floor(guess_of_Y(n)))+abs(position(2,n)-Y);
	deviationX_image(floor(guess_of_X(n)),floor(guess_of_Y(n)))=deviationX_image(floor(guess_of_X(n)),floor(guess_of_Y(n)))+sigma_X;
	deviationY_image(floor(guess_of_X(n)),floor(guess_of_Y(n)))=deviationY_image(floor(guess_of_X(n)),floor(guess_of_Y(n)))+sigma_Y;
	counts_guess(floor(guess_of_X(n)),floor(guess_of_Y(n)))=counts_guess(floor(guess_of_X(n)),floor(guess_of_Y(n)))+1;

	fired_spikes=[fired_spikes sum(spike_rate(:,n))];



	if mod(n,floor(size(position,2)/10))==0
		disp(['Reading results of decoding, ', num2str(100*n/size(position,2)), '% achieved']);
	end
end













X=(1:size(position,2))/10;
ecartT=sqrt(ecartT_x.^2+ecartT_y.^2);
%%%%%%%%%%%%%%%%-------------- MOVIE ---------------%%%%%%%%%%%%%%%%%%%%%%%
arena1x=[14 7 7 29.5 29.5 21 14]; arena1y=[3 25 30 30 25 3 3];
arena2x=[17 13 17 17]; arena2y=[12 25 25 12];
arena3x=[20 20 25 20]; arena3y=[12 25 25 12];
figure;
spl3=subplot(1,9,3);
handle=fill([guess_of_X-ecartT_x fliplr(guess_of_X+ecartT_x)],[X,fliplr(X)],[176/255 224/255 230/255]);hold on;
set(handle,'edgecolor','none');
plot(guess_of_X,(1:size(guess_of_X,2))/10,'Color',[70/255 130/255 180/255]);hold on;
% plot(max_X);hold on;
plot(position(1,:),(1:size(position,2))/10,'Color',[220/255 20/255 60/255]);
% legend('estimation of X up to one-sigma','estimation of X','measurement of X');
xlabel('X');
ylim([-15 15]);
xlim([0 30]);
spl2=subplot(1,9,2);
handle=fill([guess_of_Y-ecartT_y fliplr(guess_of_Y+ecartT_y)],[X,fliplr(X)],[176/255 224/255 230/255]);hold on;
set(handle,'edgecolor','none');
plot(guess_of_Y,(1:size(guess_of_Y,2))/10,'Color',[70/255 130/255 180/255]);hold on;
% plot(max_Y);hold on;
plot(position(2,:),(1:size(position,2))/10,'Color',[220/255 20/255 60/255]);
% legend('estimation of Y up to one-sigma','estimation of Y','measurement of Y');
xlabel('Y');
ylim([-15 15]);
xlim([0 30]);
spl1=subplot(1,9,1);
handle=fill([ones(size(X))*7 fliplr(ecartT)],[X,fliplr(X)],[250/255 128/255 114/255]);hold on;
set(handle,'edgecolor','none');
handle=fill([zeros(size(X)) fliplr(ones(size(X))*7)],[X,fliplr(X)],'w');hold on;
set(handle,'edgecolor','none');
handle=fill([zeros(size(X)) fliplr(min(ecartT,ones(size(X))*7))],[X,fliplr(X)],[176/255 224/255 230/255]);hold on;
set(handle,'edgecolor','none');
plot(ecartT,(1:size(ecartT,2))/10,'Color',[47/255 79/255 79/255]);
xlabel('standard deviation'); ylabel('time (s)');
ylim([-15 15]);
xlim([min(ecartT) max(ecartT)]);
set(spl2,'YTick',[]);
set(spl3,'YTick',[]);
p1=get(spl1,'pos');
p2=get(spl2,'pos'); p2(3)=p2(3)+(p2(1)-p1(1)-p1(3)); p2(1)=p1(1)+p1(3); 
p3=get(spl3,'pos'); p3(3)=p3(3)+(p3(1)-p2(1)-p2(3)); p3(1)=p2(1)+p2(3); 
set(spl2,'pos',p2); set(spl3,'pos',p3);
pause
for n=400:size(position,2)
	if n~=400
		delete(lignes);
		delete(lignev);
		delete(lignex);
		delete(ligney);
	end
	colormap(parula);
	if ecartT(n)<7
		spl4=subplot(1,9,[4 5 6 7 8 9]);
		a(:,:)=position_proba(:,:,n);
		imagesc(a);hold on;colorbar;
		plot(arena3x,arena3y,'-g');
		plot(arena2x,arena2y,'-g');
		plot(arena1x,arena1y,'-g');
		plot(position(2,n),position(1,n),'o','Color','r','markerfacecolor','r');hold off;
		xlim([5 30]);
	else
		spl4=subplot(1,9,[4 5 6 7 8 9]);
		imagesc(zeros(size(a)));hold on;colorbar;
		fill([[0 31],fliplr([0 31])],[[0 0] fliplr([31 31])],[0.2081/2 0.1663/2 0.5292/2])
		plot(arena3x,arena3y,'-r');
		plot(arena2x,arena2y,'-r');
		plot(arena1x,arena1y,'-r');
		plot(position(2,n),position(1,n),'o','Color','r','markerfacecolor','r');hold off;
		xlim([5 30]);
	end
	subplot(spl1);
	lignes=hline((n-1)/10,'k');
	lignev=vline(7,'r');
	ylim([-15+(n-1)/10 15+(n-1)/10]);
	subplot(spl2);
	lignex=hline((n-1)/10,'k');
	ylim([-15+(n-1)/10 15+(n-1)/10]);
	subplot(spl3);
	ligney=hline((n-1)/10,'k');
	ylim([-15+(n-1)/10 15+(n-1)/10]);
	pause(0.05);
	if mod(n,floor(size(position,2)/100))==0
		disp(['Reading results of decoding, ', num2str(100*n/size(position,2)), '% achieved']);
	end
end



















% FileName='test'; mkdir(FileName);

clearvars maxs testy


f1=figure('Name','X&Y','NumberTitle','off');clf;
sb(1)=subplot(2,1,1);
handle=fill([X,fliplr(X)],[guess_of_X-ecartT_x fliplr(guess_of_X+ecartT_x)],[176/255 224/255 230/255]);hold on;
set(handle,'edgecolor','none');
plot(guess_of_X,'Color',[70/255 130/255 180/255]);hold on;
% plot(max_X);hold on;
plot(position(1,:),'Color',[220/255 20/255 60/255]);
legend('estimation of X up to one-sigma','estimation of X','measurement of X');
xlabel('time'); ylabel('position along X axis');
sb(2)=subplot(2,1,2);
handle=fill([X,fliplr(X)],[guess_of_Y-ecartT_y fliplr(guess_of_Y+ecartT_y)],[176/255 224/255 230/255]);hold on;
set(handle,'edgecolor','none');
plot(guess_of_Y,'Color',[70/255 130/255 180/255]);hold on;
% plot(max_Y);hold on;
plot(position(2,:),'Color',[220/255 20/255 60/255]);
legend('estimation of Y up to one-sigma','estimation of Y','measurement of Y');
xlabel('time'); ylabel('position along Y axis');
linkaxes(sb');
xlim([500*10/10 1000*10/10]);
savefig([FileName,'X&Y.fig']);



figure('Name','positions','NumberTitle','off');clf;
subplot(1,2,1);
nBins=[max(position(1,:))-min(position(1,:))+1 max(position(2,:))-min(position(2,:))+1];
h=histogram2(position(1,:),position(2,:),nBins);
title('Histogram of measured positions');
subplot(1,2,2);
h2=histogram2(guess_of_X,guess_of_Y,nBins);
title('Histogram of guessed positions');
savefig([FileName,'positions.fig']);



figure('Name','errors','NumberTitle','off');clf;
subplot(1,2,1);
nBin=floor(sqrt(size(position_proba,1)^2+size(position_proba,2)^2));
h5=histogram(generalized_error,nBin); hold on;
edges=h5.BinEdges;
h3=histogram(specific_error,edges);
legend(['error with respect to expected value | mean = ',num2str(mean(generalized_error))],['error with respect to most probable outcome | mean = ',num2str(mean(specific_error))]);
title(['Generalized error histogram | diff of mean : ',num2str(mean(generalized_error)-mean(specific_error))]);
subplot(1,2,2);
h4=histogram(comparison_error);
title(['Comparison of error histogram | mean : ',num2str(mean(comparison_error))]);
savefig([FileName,'errors.fig']);


figure('Name','standard_deviation','NumberTitle','off');clf;
subplot(2,3,1);
plot(ecartT_x,generalized_error,'b.');
xlabel('standard deviation of x');ylabel('generalized error');
subplot(2,3,2);
plot(ecartT_y,generalized_error,'b.');
xlabel('standard deviation of y');ylabel('generalized error');
subplot(2,3,4);
histogram(ecartT_x,50);
xlabel('standard deviation of x');
subplot(2,3,5);
histogram(ecartT_y,50);
xlabel('standard deviation of y');
subplot(2,3,3);
plot(ecartT,generalized_error,'b.');
xlabel('absolute standard deviation');ylabel('generalized error');
subplot(2,3,6);
histogram(ecartT,50);
xlabel('absolute standard deviation');
savefig([FileName,'standard_deviation']);




figure('Name','error_deviation_map','NumberTitle','off');clf;
subplot(2,2,1);
imagesc(errorX_map./counts);hold on;colorbar;
title('Error on X');
subplot(2,2,2);
imagesc(errorY_map./counts);hold on;colorbar;
title('Error on Y');
subplot(2,2,3);
imagesc(deviationX_map./counts);hold on;colorbar;
title('Standard deviation of X');
subplot(2,2,4);
imagesc(deviationY_map./counts);hold on;colorbar;
title('Standard deviation of Y');
savefig([FileName,'error_deviation_map.fig']);

figure('Name','error_deviation_image','NumberTitle','off');clf;
subplot(2,2,1);
imagesc(errorX_image./counts_guess);hold on;colorbar;
title('Error on X');
subplot(2,2,2);
imagesc(errorY_image./counts_guess);hold on;colorbar;
title('Error on Y');
subplot(2,2,3);
imagesc(deviationX_image./counts_guess);hold on;colorbar;
title('Standard deviation of X');
subplot(2,2,4);
imagesc(deviationY_image./counts_guess);hold on;colorbar;
title('Standard deviation of Y');
savefig([FileName,'error_deviation_image.fig']);

figure('Name','error_deviation_all','NumberTitle','off');clf;
subplot(2,2,1);
imagesc(clean_matrix(errorX_image./counts_guess)+clean_matrix(errorX_map./counts));hold on;colorbar;
title('Error on X');
subplot(2,2,2);
imagesc(clean_matrix(errorY_image./counts_guess)+clean_matrix(errorY_map./counts));hold on;colorbar;
title('Error on Y');
subplot(2,2,3);
imagesc(clean_matrix(deviationX_image./counts_guess)+clean_matrix(deviationX_map./counts));hold on;colorbar;
title('Standard deviation of X');
subplot(2,2,4);
imagesc(clean_matrix(deviationY_image./counts_guess)+clean_matrix(deviationY_map./counts));hold on;colorbar;
title('Standard deviation of Y');
savefig([FileName,'error_deviation_all.fig']);


disp('to find the nth point on the map : plot(position(2,n),position(1,n),''go'')');


% yyaxis right
lowSigma_points=[];
density=[0 0 0 0 0 0 0 0 0 0];
for j=11:5000
	S=0;
	for k=0:9
		S=S+ecartT(j-k)/(0.2*k+1);
	end
	density=[density S];
end
smoothed_density=smooth(density);
% plot(smoothed_density);
highdens=0;
for j=1:5000
	if smoothed_density(j)<max(smoothed_density)/3
		lowSigma_points=[lowSigma_points j];
	end
end	
% lowSigma_points=find(ecartT<7);
lowSigma_X=guess_of_X(lowSigma_points);
lowSigma_Y=guess_of_Y(lowSigma_points);
f1=figure('Name','X&Y_with_lowSigma','NumberTitle','off');clf;
sb(1)=subplot(2,1,1);
handle=fill([X,fliplr(X)],[guess_of_X-ecartT_x fliplr(guess_of_X+ecartT_x)],[176/255 224/255 230/255]);hold on;
set(handle,'edgecolor','none');
plot(guess_of_X,'Color',[70/255 130/255 180/255]);hold on;
plot(position(1,:),'Color',[220/255 20/255 60/255]);
%plot(list10ms,result10msX,'.','Color',[47/255 79/255 79/255]);
plot(lowSigma_points,lowSigma_X,'o','Color','k','markerfacecolor','k')
legend('estimation of X up to one-sigma','estimation of X','measurement of X','estimation of X results w/low sigma');
xlabel('time'); ylabel('position along X axis');
sb(2)=subplot(2,1,2);
handle=fill([X,fliplr(X)],[guess_of_Y-ecartT_y fliplr(guess_of_Y+ecartT_y)],[176/255 224/255 230/255]);hold on;
set(handle,'edgecolor','none');
plot(guess_of_Y,'Color',[70/255 130/255 180/255]);hold on;
plot(position(2,:),'Color',[220/255 20/255 60/255]);
plot(lowSigma_points,lowSigma_Y,'o','Color','k','markerfacecolor','k');
legend('estimation of Y up to one-sigma','estimation of Y','measurement of Y','estimation of Y results w/low sigma');
xlabel('time'); ylabel('position along Y axis');
linkaxes(sb');
xlim([500*10/10 1000*10/10]);
savefig([FileName,'X&Y_with_lowSigma.fig']);

mean_pvalue=mean(p_value);
mean_pvalue_lowSigma=[];
nb_lowSigma_points=[];
for i=1:20
	lowSigma_points=find(ecartT<i);
	mean_pvalue_lowSigma=[mean_pvalue_lowSigma mean(p_value(lowSigma_points))];
	nb_lowSigma_points=[nb_lowSigma_points size(lowSigma_points,2)];
end

try
	save('decoding_results_100ms.mat','mean_pvalue','mean_pvalue_lowSigma','nb_lowSigma_points','-append');
end