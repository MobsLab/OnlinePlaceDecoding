try isstr(s);
catch
	build_s;
	s(:,1) = s(:,1)*20000;
	spikes_tet1 = find(s(:,2)==1);
end
% try isstr




N_SAMPLE_TO_READ = 1000;
N_CHANNELS       = 71;
N_REAL_CHANNELS  = 43;
X=(1:N_SAMPLE_TO_READ)/20;







chnlStart = 2; chnlEnd = 4;
fid = fopen('ERC-Mouse-743-01062018-Hab.dat','r');
% fid = fopen('ProjectEmbReact_M509_20170204_Habituation.dat','r');
lfpid = fopen('ERC-Mouse-743-01062018-Hab.lfp','r');

pre=zeros(N_CHANNELS,N_SAMPLE_TO_READ);
post=zeros(N_CHANNELS,N_SAMPLE_TO_READ);
pre1=zeros(N_CHANNELS,1);
pre2=zeros(N_CHANNELS,1);
post1=zeros(N_CHANNELS,1);
post2=zeros(N_CHANNELS,1);


B = zeros(N_CHANNELS, N_SAMPLE_TO_READ);
n_window=1;
while 1

	spikes = intersect(intersect(spikes_tet1, find(s(:,1)>(n_window-1)*N_SAMPLE_TO_READ)), find(s(:,1)<n_window*N_SAMPLE_TO_READ));
	A = fread(fid, [N_CHANNELS, N_SAMPLE_TO_READ], 'int16');
	% B = fread(lfpid, [N_CHANNELS, N_SAMPLE_TO_READ], 'int16');


	figure(1);
	for channel = chnlStart:chnlEnd
		plot(X + (n_window-1)*N_SAMPLE_TO_READ/20, A(channel,:) - B(channel,:));
		if channel == chnlEnd
			hold off;
		else
			hold on;
		end
	end
	for spk=1:size(spikes,1)
		vline((s(spikes(spk),1))/20,'--k');
	end

	figure(2);
	for channel = chnlStart:chnlEnd
		this = INTANfilter(A(channel,:) - B(channel,:),pre1(channel),pre2(channel),post1(channel),post2(channel));
		% if n_window==1
		% 	this = A(channel,:);
		% else
		% 	this = highpassfilter(A(channel,:) - B(channel,:),pre(channel,:),post(channel,:));
		% end
		plot(X + (n_window-1)*N_SAMPLE_TO_READ/20, this);
		% ylim([-300 300])
		if channel == chnlEnd
			hold off;
		else
			hold on;
		end
		pre(channel, :) = A(channel, :);
		post(channel, :) = this;
		pre1(channel) = A(channel, end);
		pre2(channel) = A(channel, end-1);
		post1(channel) = this(end);
		post2(channel) = this(end-1);

	end
	for spk=1:size(spikes,1)
		vline((s(spikes(spk),1))/20,'--k');
	end
	pause;
	n_window = n_window+1;
end



fclose(fid); % don't forget to close the file











