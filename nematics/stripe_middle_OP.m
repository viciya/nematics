%%
Ddir = dir(['V:\HT1080_small_stripes_glass_22112017\CROPPED\Orient' '\*.tif']);
% a=dir(['V:\HT1080_small_stripes_glass_22112017\CROPPED\Orient' '\*.tif'])
% out=size(a,1)
folder_main = Ddir(1).folder;
filesInFolder = size(Ddir,1);
frame_per_hr = 4;
frames = 50;
dt=1;

px_size = .74*3;
qstep = 7;overlap = 1;
% qstep = 20;overlap = 6;
px2mic = px_size * frame_per_hr;
%%
n=130;
OP_mid = zeros(size(Ddir,1),1);
OP_mid_std = OP_mid;
width = OP_mid;
for i=1:size(Ddir,1)
    %     if contains(Ddir(i).name, '.tif' )
    Ddir(i).name
    disp(['file: ' num2str(i) ' from: ' num2str(size(Ddir,1))]);
    filepath = [Ddir(i).folder '\' Ddir(i).name];
    
    info = imfinfo(filepath); % Place path to file inside single quotes
    Nn = numel(info);
    
    Ang = imread(filepath,1); % k
    [l,w] = size(Ang);
    
    kk=1;
    OP_frame = zeros(Nn,1);
    %         for k=Nn-frames:Nn
    for k=1:Nn
        %             % --------------------------ORIENT import ---------------------------------
        Ang = imread(filepath,k); % k
        mAng = Ang(:, floor(w/2)-5:ceil(w/2)+5);
        mAng(mAng<0) = mAng(mAng<0)+180;
        mAngVec = reshape(mAng,[size(mAng,1)*size(mAng,2) 1]);
        OP_frame(k,1) = sqrt(sum(sum(cos(2*mAngVec*pi/180)))^2 ...
            +sum(sum(sin(2*mAngVec*pi/180)))^2)/(size(mAng,1)*size(mAng,2));
    end
    width(i,1) = px_size*w;
    OP_mid(i,1) = mean(OP_frame);
    OP_mid_std(i,1) = std(OP_frame)/Nn^.5;
end
EXP = [width,OP_mid,OP_mid_std];
%%
[UWidth,~,idx]  = unique(EXP(:,1));
N = histc(EXP(:,1), UWidth); % repetition number
UWidth_OP = [UWidth, accumarray(idx, EXP(:,2),[],@mean), accumarray(idx,EXP(:,2),[],@std)./sqrt(N)];
figure(11); errorbar(UWidth_OP(:,1),UWidth_OP(:,2),UWidth_OP(:,3));
%%
widthOP = UWidth_OP(:,1);
OP = UWidth_OP(:,2);

%% MAKE AVERAGE PLOT
% load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\midOP_All.mat')
load('F:\GD\Curie\DESKTOP\HT1080\midOP_All.mat')

figure(123)
dBins = 4;% Bin resolution
dd = 2*dBins;% range
totalBins = floor((max(widthOP)- min(widthOP))/dBins);
[N, edges] = histcounts(widthOP,totalBins);
edgess = mean([edges(1:end-1); edges(2:end)]);
[Peak,widthOfPeak] = findpeaks(N,edgess);
plot(edgess,N,'-'); hold on
plot(widthOfPeak,Peak,'o'); hold off
% histogram(pks,floor(length(pks)/3))
% NEW X-SCALE
avOP = zeros(length(widthOfPeak),4);
avOP(:,1) = widthOfPeak;
for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    avOP(i,2) = mean(OP(widthOP>=ww-dd & widthOP<=ww+dd));  
    avOP(i,3) = std(OP(widthOP>=ww-dd & widthOP<=ww+dd))./...
        sum(widthOP>=ww-dd & widthOP<=ww+dd)^.5;
    avOP(i,4) = sum(widthOP>=ww-dBins & widthOP<=ww+dBins);
end
figure(124)
scatter(width,OP_mid,10,'fill'); hold on
plot(widthOP,OP,'.');hold on
errorbar(avOP(:,1),avOP(:,2),avOP(:,3));hold off





