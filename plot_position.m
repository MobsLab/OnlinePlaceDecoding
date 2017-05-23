try isstr(position_proba);
catch
	try
		decoding_results=importdata('decoding_results_50ms.mat');
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












% arena1x=[16 9 29 20 16]; arena1y=[3 29 29 3 3];
% arena2x=[17 13 17 17]; arena2y=[12 25 25 12];
% arena3x=[20 20 25 20]; arena3y=[12 25 25 12];
% figure;
% pause
% for n=1:size(position,2)
% 	a(:,:)=position_proba(:,:,n);
% 	imagesc(a);hold on;colorbar;
% 	plot(arena3x,arena3y,'-g');
% 	plot(arena2x,arena2y,'-g');
% 	plot(arena1x,arena1y,'-g');
% 	plot(position(2,n),position(1,n),'o','Color','r','markerfacecolor','r');hold off;
% 	colormap(parula);
% 	pause(0.1);
% 	if mod(n,floor(size(position,2)/100))==0
% 		disp(['Reading results of decoding, ', num2str(100*n/size(position,2)), '% achieved']);
% 	end
% end











% FileName='test'; mkdir(FileName);

clearvars maxs testy

X=1:size(position,2);

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
subplot(2,2,1);
plot(ecartT_x,generalized_error,'b.');
xlabel('standard deviation of x');ylabel('generalized error');
subplot(2,2,2);
plot(ecartT_y,generalized_error,'b.');
xlabel('standard deviation of y');ylabel('generalized error');
subplot(2,2,3);
histogram(ecartT_x,50);
xlabel('standard deviation of x');
subplot(2,2,4);
histogram(ecartT_y,50);
xlabel('standard deviation of y');
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


disp('to find the nth point on the map : plot(position(2,n),position(1,n),''go'')');


f1=figure('Name','X&Y_w50ms','NumberTitle','off');clf;
sb(1)=subplot(2,1,1);
handle=fill([X,fliplr(X)],[guess_of_X-ecartT_x fliplr(guess_of_X+ecartT_x)],[176/255 224/255 230/255]);hold on;
set(handle,'edgecolor','none');
plot(guess_of_X,'Color',[70/255 130/255 180/255]);hold on;
plot(position(1,:),'Color',[220/255 20/255 60/255]);
%plot(list10ms,result10msX,'.','Color',[47/255 79/255 79/255]);
plot(list50ms,result50msX,'o','Color','k','markerfacecolor','k')
legend('estimation of X up to one-sigma','estimation of X','measurement of X','10ms results w/escartT<6');
xlabel('time'); ylabel('position along X axis');
sb(2)=subplot(2,1,2);
handle=fill([X,fliplr(X)],[guess_of_Y-ecartT_y fliplr(guess_of_Y+ecartT_y)],[176/255 224/255 230/255]);hold on;
set(handle,'edgecolor','none');
plot(guess_of_Y,'Color',[70/255 130/255 180/255]);hold on;
plot(position(2,:),'Color',[220/255 20/255 60/255]);
plot(list50ms,result50msY,'o','Color','k','markerfacecolor','k');
legend('estimation of Y up to one-sigma','estimation of Y','measurement of Y','10ms results w/escartT<6');
xlabel('time'); ylabel('position along Y axis');
linkaxes(sb');
xlim([500*10/10 1000*10/10]);
savefig([FileName,'X&Y_w50ms.fig']);