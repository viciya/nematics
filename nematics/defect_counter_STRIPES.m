%%
% Ddir = dir('V:\HT1080_small_stripes_glass_22112017\CROPPED\Orient');
Ddir = dir('C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\Orient');
folder_main = Ddir(1).folder;
filesInFolder = size(Ddir,1);
px_size = .74*3;
        Orient_area = zeros(1,size(Ddir,1));
        Orient_width = zeros(1,size(Ddir,1));
        defNum = cell(size(Ddir,1),1);
        defDensity = cell(size(Ddir,1),1); 


for i=1:size(Ddir,1)
%     i=22;
    if contains(Ddir(i).name, '.tif' )
        Ddir(i).name    
        disp(['file: ' num2str(i) ' from: ' num2str(size(Ddir,1))]);
        filepath = [Ddir(i).folder '\' Ddir(i).name];        
        info = imfinfo(filepath); % Place path to file inside single quotes
        Nn = numel(info);
       
        Orient_area(i) = px_size^2*info(1).Width*info(1).Height;
        Orient_width(i) = px_size*info(1).Width;
        
        qstep = 6; %mine 10 in pixels
        for k=1:Nn
%             k=65;
            Ang = imread(filepath,k); % k
            Ang = pi/180*Ang;
            [l,w] = size(Ang);
            
            qq = ordermatrixglissant_overlap(Ang,qstep,3);
            im2 = qq < min(qq(:))+0.4;%.2; % make binary image to use regionprops
            s = regionprops('table', im2,'centroid');
            defNum{i}(k,1) = size(s,1);
            defDensity{i}(k,1) = size(s,1)/Orient_area(i);
        end
    end
end
%%
imagesc(qq); axis equal; axis tight;hold on
scatter(s.Centroid(:,1),s.Centroid(:,2),20,[0 0 0],'filled');
% % Orientation
[Xu,Yu] = meshgrid(1:size(Ang,2),1:size(Ang,1));
step = 5;
q = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
    cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)),.7);
q.LineWidth=1;
q.Color = [.4 .4 .4];
q.ShowArrowHead='off';

view([-90 90])
hold off
%%
clear
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\defect number_All.mat')

emptyTF = cellfun(@isempty,defDensity); %check for empty cells
count = 1;
for i=1:size(defDensity,1)
    if emptyTF(i)==0
meanDefectDensity(i) = mean(defDensity{i});
stdDefectDensity(i) = std(defDensity{i})/size(defDensity{i},1)^.5;

meanDefectNum(i) = mean(defNum{i});
stdDefectNum(i) = std(defNum{i})/size(defDensity{i},1)^.5;
    end
end

[sortOrient_width, ind] = sort(Orient_width);
sortOrient_area = Orient_area(ind);
meanDefectDensity = meanDefectDensity(ind);
stdDefectDensity = stdDefectDensity(ind);
meanDefectNum = meanDefectNum(ind);
stdDefectNum = stdDefectNum(ind);
figure(3)
% errorbar(sortOrient_width,meanDefectDensity,stdDefectDensity);hold off
p1=plot(sortOrient_width,meanDefectDensity,'o');hold off
p1.MarkerSize = 10; p1.MarkerEdgeColor= [1 1 1]; p1.MarkerFaceColor= [0 .5 .8];
xlabel('Width (\mum)');ylabel('Defect Density (\mum^{-2})'); set(gca,'FontSize',20);
% axis([0 250 0 2.1e-5]);
figure(4)
errorbar(sortOrient_width,meanDefectNum,stdDefectNum);hold off

%% MAKE AVERAGE PLOT
figure(123)
maxBin = 5;% half of maximal delta width 
dBins = 5;% Bin resolution
dd = dBins + 6;% range
totalBins = floor((max(sortOrient_width)- min(sortOrient_width))/dBins);
[N, edges] = histcounts(sortOrient_width,totalBins);
edgess = mean([edges(1:end-1); edges(2:end)]);
[Peak,widthOfPeak] = findpeaks(N,edgess);
plot(edgess,N,'-'); hold on
plot(widthOfPeak,Peak,'o'); hold off
% histogram(pks,floor(length(pks)/3))
% NEW X-SCALE
MDefectDensity = zeros(length(widthOfPeak),4);
MDefectDensity(:,1) = widthOfPeak;
for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    MDefectDensity(i,2) = mean(meanDefectDensity(sortOrient_width>=ww-dd & sortOrient_width<=ww+dd));  
    MDefectDensity(i,3) = std(meanDefectDensity(sortOrient_width>=ww-dd & sortOrient_width<=ww+dd))./...
        sum(sortOrient_width>=ww-dd & sortOrient_width<=ww+dd)^.5;
    MDefectDensity(i,4) = sum(sortOrient_width>=ww-dBins & sortOrient_width<=ww+dBins);
end
figure(124)
% plot(Orient_width,meanDefectDensity,'.');hold on
plot(sortOrient_width,meanDefectDensity,'o');hold on
errorbar(MDefectDensity(:,1),MDefectDensity(:,2),MDefectDensity(:,3));hold off

%% get subfolders from main folder
% k=14;
% subDir = struct;
% kk=1;
% for k=1:size(Ddir,1)
%     if isdir([Ddir(k).folder '\' Ddir(k).name]) && ~contains(Ddir(k).name, '.' )
%         subDir = dir([Ddir(k).folder '\' Ddir(k).name]);
%         [Ddir(k).folder '\' Ddir(k).name]
%         for kk=3:3%:size(subDir,1)
%             if isdir([subDir(kk).folder '\' subDir(kk).name]) && ~contains(subDir(kk).name, '.' )
%                 subsubDir = dir([subDir(kk).folder '\' subDir(kk).name]);
%                 subsubDir.name
%             end
%         end
%     end
% end




