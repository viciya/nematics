% Calculation of the angle for stripes
% OPTION OF TIME AVERAGE
% ------  CALCULATES: -----------
% 1) disribution 
% 2) plots polar and normal histogram for edge and middle
% 3) plots the angles in time
% ------  USAGE: -----------
% Import Orientation stack from OrientationJ
% 
% ------  PARAMETERS: -----------
% 1) frame rate :: frame_per_hr
% 3) First frame :: Fi
% 4) Last frame :: Fend
% 4) Middle/Edge stripe width :: dw
% ************   TO ADD   ***************
% ORDER PARAMETER
% TIME PROFILE
% WIDTH OF THE ROIs 
% PROFILE KYMOGRAPH OF ANGLES ALONG THE STRIPE
% MODIFY THE POLAR HISTOGRAM

% %%  PARAMETERS
clear all
close all
%%
frame_per_hr = 4;
px2mic = .74; %40x (0.185); 20x (0.37); 10x (0.74)

[filename, pathname,filterindex] = uigetfile( ...
    '*.*', 'Pick a file',...
    '\\Nausicaa\Victor\HT1080_small_stripes_glass_22112017\CROPPED\Orient');

info = imfinfo([pathname,filename]); % Place path to file inside single quotes
Nn = numel(info);

%% CHECK ANGLE  in ONE FRAME 
% add an ORDER PARAMETER

dw = 4; % Middle/Edge stripe width
i = 56;
% for i=1:Nn %Nn total number of images in your stack
Ang = imread([pathname,filename],i); %i
Ang(Ang <= 0) = 180 + Ang(Ang <= 0);

[l,w] = size(Ang);
midAng = Ang(:, floor(w/2)-floor(dw/2):floor(w/2)+ceil(dw/2));
edgeAng = Ang(:, [1:(dw+1), end-(dw+1):end]);

lw_mid = size(midAng,1)*size(midAng,2);
lw_edge = size(edgeAng,1)*size(edgeAng,2);

% --------   FOR ANGLE IN DEGREES ----------
midAng_vec = reshape(midAng,[lw_mid,1]);
midAng_vec(midAng_vec <= 0) = 180 + midAng_vec(midAng_vec <= 0); % add 180 to negative angles
edgeAng_vec = reshape(edgeAng,[lw_edge,1]);
edgeAng_vec(edgeAng_vec <= 0) = 180 + edgeAng_vec(edgeAng_vec <= 0);

% -----  polhistogram in RADIANS ------------------------
subplot(3,2,1);
if verLessThan('matlab', '9.1.0') == 0
   % for MATLAB > 2016
    polarhistogram(edgeAng_vec*pi/180,'BinLimits',[0 pi],'Normalization','pdf');  hold on;
    polarhistogram(midAng_vec*pi/180,'BinLimits',[0 pi], 'Normalization','pdf');
    axis([0 180 0 inf]);
else
     % for MATLAB < 2016
    rose(edgeAng_vec*pi/180,360);hold on;
    rose(midAng_vec*pi/180,360);hold off;
end
hold off;

subplot(3,2,2)
% -----  histogram in DEGREES ------------------------
histogram(edgeAng_vec,'Normalization','pdf');  hold on;
histogram(midAng_vec,'Normalization','pdf'); set(gca, 'xdir','reverse');
title(['frame:' num2str(i)]);
legend(['edge: ' num2str(mean(edgeAng_vec))...
,' | std:  ' num2str(std(edgeAng_vec)/sqrt(length(edgeAng_vec)))],...
['middle:  ' num2str(mean(midAng_vec))...
,' | std:  ' num2str(std(midAng_vec)/sqrt(length(edgeAng_vec)))],...
'Location','southoutside');
hold off;

%% (RUN 1) CHECK ANGLE FOR ALL FRAMES 
% add an ORDER PARAMETER
Ang = imread([pathname,filename],1); %i
[l,w] = size(Ang);

Fi = 1;
Fend = Nn;
T = Fend-Fi + 1;

dw = 10; % Middle/Edge stripe width

time = (1:T)';
time = time/frame_per_hr;
edgeAll  = zeros(T,1);
std_edgeAll = zeros(T,1);
midAll = zeros(T,1);
std_midAll = zeros(T,1);
mid_OP = zeros(T,1);
edge_OP = zeros(T,1);

lw_mid = l*(dw+1);
temp_mid = zeros(lw_mid,1);

lw_edge = l*(dw+1)*2;
temp_edge = zeros(lw_edge,1);

i = 1;
for ii = Fi:Fend %Nn total number of images in your stack
    
Ang = imread([pathname,filename],ii); %i

midAng = Ang(:, floor(w/2)-floor(dw/2):floor(w/2)+floor(dw/2));
edgeAng = Ang(:, [1:(dw+1), end-dw:end]);

% --------   FOR ANGLE IN DEGREES ----------
temp_mid = reshape(midAng,[lw_mid,1]);
temp_mid(temp_mid <= 0) = 180 + temp_mid(temp_mid <= 0); % add 180 to negative angles
% midAng_vec(:,i) = temp_mid;
midAll(i,1) = mean(temp_mid);
std_midAll(i,1) = std(temp_mid);

temp_edge = reshape(edgeAng,[lw_edge,1]);
temp_edge(temp_edge <= 0) = 180 + temp_edge(temp_edge <= 0);
% edgeAng_vec(:,i) = temp_edge;
edgeAll(i,1)  = mean(temp_edge);
std_edgeAll(i,1) = std(temp_edge);

mid_OP(i,1) = sqrt(sum(sum(cos(2*temp_mid*pi/180)))^2 ...
                    +sum(sum(sin(2*temp_mid*pi/180)))^2)/(lw_mid);
edge_OP(i,1) = sqrt(sum(sum(cos(2*temp_edge*pi/180)))^2 ...
                    +sum(sum(sin(2*temp_edge*pi/180)))^2)/(lw_edge);

i = i + 1;
end
%%
edge_mean = mean(edgeAll);
edge_sem = std(edgeAll)/sqrt(T);
mid_mean = mean(midAll);
mid_sem = std(midAll)/sqrt(T);

edge_OP_mean = mean(edge_OP);
edge_OP_sem = std(edge_OP)/sqrt(T);
mid_OP_mean = mean(mid_OP);
mid_OP_sem = std(mid_OP)/sqrt(T);

%%

subplot(3,2,3:4)
plot(time,edgeAll); hold on
plot(time,midAll); axis tight

ylabel('Angle (deg)');xlabel('time (hr)');
legend(['<edge>: ' num2str(edge_mean),...
    ' | sem:  ' num2str(edge_sem)],...
    ['<middle>:  ' num2str(mid_mean),...
    ' | std:  ' num2str(mid_sem)],'Location','northeast');hold off
axis([-2 time(end)+2 -inf inf]);


subplot(3,2,5:6)
plot(time,edge_OP); hold on
plot(time,mid_OP); axis tight
legend(['<edge>: ' num2str(edge_OP_mean),...
    ' | sem:  ' num2str(edge_OP_sem)],...
    ['<middle>:  ' num2str(mid_OP_mean),...
    ' | sem:  ' num2str(edge_OP_sem)],'Location','northeast');
ylabel('OP [0,1]');xlabel('time (hr)'); hold off
axis([-2 time(end)+2 -inf inf]);
%% (RUN 2) CHECK ANGLE FOR PART OF FRAMES 
% add an ORDER PARAMETER
% Choose sub time window
[tt,~] = ginput(2);

Ang = imread([pathname,filename],1); %i
[l,w] = size(Ang);

Fi = floor(tt(1)*frame_per_hr);
Fend = floor(tt(2)*frame_per_hr);
if tt(2)>time(end)
    Fend = floor(time(end)*frame_per_hr);
end
if tt(1)<=1
    Fi = 1;
end

T = Fend-Fi+1;

% dw = 10; % Middle/Edge stripe width

d_time = (1:T)';
d_time = d_time/frame_per_hr;
d_edgeAll  = zeros(T,1);
d_std_edgeAll = zeros(T,1);
d_midAll = zeros(T,1);
d_stdmidAll = zeros(T,1);
d_edge_OP = zeros(T,1);
d_mid_OP = zeros(T,1);

lw_mid = l*(dw+1);
d_temp_mid = zeros(lw_mid,1);
d_midAng_vec = zeros(lw_mid,T);

lw_edge = l*(dw+1)*2;
d_temp_edge = zeros(lw_edge,1);
d_edgeAng_vec = zeros(lw_edge,T);

i = 1;
for ii = Fi:Fend %Nn total number of images in your stack
    
Ang = imread([pathname,filename],ii); %i

midAng = Ang(:, floor(w/2)-floor(dw/2):floor(w/2)+ceil(dw/2));
edgeAng = Ang(:, [1:(dw+1), end-dw:end]);

% --------   FOR ANGLE IN DEGREES ----------
d_temp_mid = reshape(midAng,[lw_mid,1]);
d_temp_mid(d_temp_mid <= 0) = 180 + d_temp_mid(d_temp_mid <= 0); % add 180 to negative angles
d_midAng_vec(:,i) = d_temp_mid;
d_midAll(i,1) = mean(d_temp_mid);
d_stdmidAll(i,1) = std(d_temp_mid);

d_temp_edge = reshape(edgeAng,[lw_edge,1]);
d_temp_edge(d_temp_edge <= 0) = 180 + d_temp_edge(d_temp_edge <= 0);
d_edgeAng_vec(:,i) = d_temp_edge;
d_edgeAll(i,1)  = mean(d_temp_edge);
d_std_edgeAll(i,1) = std(d_temp_edge);

d_mid_OP(i,1) = sqrt(sum(sum(cos(2*d_temp_mid*pi/180)))^2 ...
                    +sum(sum(sin(2*d_temp_mid*pi/180)))^2)/(lw_mid);
d_edge_OP(i,1) = sqrt(sum(sum(cos(2*d_temp_edge*pi/180)))^2 ...
                    +sum(sum(sin(2*d_temp_edge*pi/180)))^2)/(lw_edge);

i = i + 1;
end
%%
d_edge_mean = mean(d_edgeAll);
d_edge_sem = std(d_edgeAll)/sqrt(T);
d_mid_mean = mean(d_midAll);
d_mid_sem = std(d_midAll)/sqrt(T);

d_edge_OP_mean = mean(d_edge_OP);
d_edge_OP_sem = std(d_edge_OP)/sqrt(T);
d_mid_OP_mean = mean(d_mid_OP);
d_mid_OP_sem = std(d_mid_OP)/sqrt(T);

d_edge_hist_mean = mean(d_edgeAng_vec(:));
d_edge_hist_sem = std(d_edgeAng_vec(:))/sqrt(size(d_edgeAng_vec,1)*size(d_edgeAng_vec,2));
d_mid_hist_mean = mean(d_midAng_vec(:));
d_mid_hist_mean_sem = std(d_midAng_vec(:))/sqrt(size(d_midAng_vec,1)*size(d_midAng_vec,2));

%% -------  PLOT TIME DEPENDENCE AND DISTRIBUTIONS  ----------
subplot(3,2,3:4)
plot(d_time,d_edgeAll); hold on
plot(d_time,d_midAll); axis tight
legend(['<edge>: ' num2str(d_edge_mean),...
' | sem:  ' num2str(d_edge_sem)],...
    ['<middle>:  ' num2str(d_mid_mean),...
' | sem:  ' num2str(d_mid_sem)],'Location','northeast');
ylabel('Angle (deg)');xlabel('time (hr)');
title(['stripe width: ', num2str(w*px2mic), ' \mum']); hold off

subplot(3,2,5:6)
plot(d_time,d_edge_OP); hold on
plot(d_time,d_mid_OP); axis tight
ylabel('OP[0,1]');xlabel('time (hr)');
legend(['<edge>: ' num2str(d_edge_OP_mean),...
    ' | sem:  ' num2str(d_edge_OP_sem)],...
    ['<middle>:  ' num2str(d_mid_OP_mean),...
    ' | sem:  ' num2str(d_edge_OP_sem)],'Location','northeast');
ylabel('OP [0,1]');xlabel('time (hr)'); hold off


subplot(3,2,1); cla
if verLessThan('matlab', '9.1.0') == 0
    % for MATLAB > 2016b
    ph1 = polarhistogram(d_edgeAng_vec*pi/180,'BinLimits',[0 pi],'Normalization','pdf');  hold on;
    ph1.EdgeAlpha = 0;
    ph1 = polarhistogram(d_midAng_vec*pi/180,'BinLimits',[0 pi], 'Normalization','pdf');
    ph1.EdgeAlpha = 0;
    ax = gca;
    ax.RAxisLocationMode = 'manual';
    ax.RAxisLocation = 210;
    ax.ThetaLim = [0 180];

else
    % for MATLAB < 2016b
    rose(edgeAng_vec*pi/180,360);hold on;
    rose(midAng_vec*pi/180,360);hold off;
end
hold off;

subplot(3,2,2); cla
h1 = histogram(d_edgeAng_vec,'Normalization','pdf');  hold on;
h1.EdgeAlpha = 0; ylabel('PDF'); xlabel('Angle (deg)');

h1 = histogram(d_midAng_vec,'Normalization','pdf');
h1.EdgeAlpha = 0;

title(['file: ', strrep(filename,'_','-')]);  % strrep: ignore underscore in string
legend(['edge: ' num2str(d_edge_hist_mean),...
' | sem:  ' num2str(d_edge_hist_sem)],...
['middle:  ' num2str(d_mid_hist_mean),...
' | sem:  ' num2str(d_mid_hist_mean_sem)], 'Location','southoutside');
axis([0 180 0 inf]); set(gca, 'xdir','reverse'); hold off;

fig = gcf;
set(fig,'Position',[100 100 800 800])

%%  SAVE THE DATA TO EXCEL
% saveas(gcf, [pathname, filename '_ST.png'])

EX = zeros(Nn+1,7);
EX(2:end,1) =  time;  
EX(2:end,2) =  edgeAll; 
EX(2:end,3) =  std_edgeAll; 
EX(2:end,4) =  midAll; 
EX(2:end,5) =  std_midAll; 
EX(2:end,6) =  edge_OP;
EX(2:end,7) =  mid_OP;
EX = num2cell(EX);
EX(1,:) = {'Time','Edge Angle','Edge Angle SEM',...
                  'Mid Angle','Mid Angle SEM',...
                  'Edge Order Parameter', 'Mid Order Parameter'};

EXcropped = zeros(T+1,7);
EXcropped(2:end,1) =  d_time;  
EXcropped(2:end,2) =  d_edgeAll; 
EXcropped(2:end,3) =  d_std_edgeAll; 
EXcropped(2:end,4) =  d_midAll; 
EXcropped(2:end,5) =  d_stdmidAll; 
EXcropped(2:end,6) =  d_edge_OP;
EXcropped(2:end,7) =  d_mid_OP;
EXcropped = num2cell(EXcropped);
EXcropped(1,:) = EX(1,:);

SUM = zeros(4,9);
SUM(2,:) = [w*px2mic, edge_mean, edge_sem, mid_mean, mid_sem,...
    edge_OP_mean, edge_OP_sem, mid_OP_mean, mid_OP_sem];
SUM(4,:) = [w*px2mic,d_edge_mean, d_edge_sem, d_mid_mean, d_mid_sem,...
    d_edge_OP_mean, d_edge_OP_sem, d_mid_OP_mean, d_mid_OP_sem];
SUM = num2cell(SUM);
SUM(1,:) = {'Width(um)', 'Av. Edge Angle','Edge Angle SEM',...
    'Av. Mid Angle','Mid Angle SEM',...
    'Av. Edge Order Parameter', 'Edge Order Parameter SEM',...
    'Av. Mid Order Parameter', 'Mid Order Parameter SEM'};
SUM(3,:) = {'Width(um)','Cropped Av. Edge Angle','Edge Angle SEM',...
    'Cropped Av. Mid Angle','Mid Angle SEM',...
    'Cropped Av. Edge Order Parameter', 'Edge Order Parameter SEM',...
    'Cropped Av. Mid Order Parameter', 'Mid Order Parameter SEM'};


d_edge_hist_mean = mean(d_edgeAng_vec(:));
d_edge_hist_sem = std(d_edgeAng_vec(:))/sqrt(size(d_edgeAng_vec,1)*size(d_edgeAng_vec,2));
d_mid_hist_mean = mean(d_midAng_vec(:));
d_mid_hist_mean_sem = std(d_midAng_vec(:))/sqrt(size(d_midAng_vec,1)*size(d_midAng_vec,2));

% xlswrite([pathname, filename,'_ST.xlsx'],SUM,1);
% xlswrite([pathname, filename,'_ST.xlsx'],EX,2);% xlswrite(filename,DATA,sheet)
% xlswrite([pathname, filename,'_ST.xlsx'],EXcropped,3);