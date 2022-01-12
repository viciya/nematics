% DEFECT detection and classification

% %% Import IMAGE SERIES
[filename, pathname,filterindex] = uigetfile( ...
    '*.tif', 'Pick a file',...
    'G:\23_11_2016 HT1080 cells on stripes\CROPED\s10\Orientation');

info = imfinfo([pathname,filename]); % Place path to file inside single quotes
Nn = numel(info);

%% Check in one frame (k)
k=65;
    Ang = imread([pathname,filename],k); % k
    Ang = pi/180*Ang;

figure(10)
% % Orientation
[Xu,Yu] = meshgrid(1:w,1:l);
step = 10;
q = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
    cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)),0.7);
q.LineWidth=1; q.Color = [.4 .4 .4]; q.ShowArrowHead='off';

axis equal;%axis off;%axis tight
axis([0 size(Ang,2) 0 size(Ang,1)])
set(gca,'xtick',[]); set(gca,'ytick',[])
hold off
 
view([-90 90])
