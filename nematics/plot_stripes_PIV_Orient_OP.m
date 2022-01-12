%%
% PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
PC_path = 'D:\GD\';                         % RSIP notebook

addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

[dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);
%%
i = 12;
l_len = .7;step =2;
Sw = 600; % selectd width
dw = .1*Sw; % define delta
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);

ff = 5;
filt = fspecial('gaussian',ff,ff);

filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];
Ang = imread(filepathOP,1);
% %%
% info = imfinfo(filepathOP); % Place path to file inside single quotes
% Nn = numel(info);
load(filepathPIV,'x','y');

[uu,vv] = sumPIV(filepathPIV,filt);
% %%

UP = vv<0;
DOWN = vv>0;
xU = x{1,1}(UP);
yU = y{1,1}(UP);
uU = uu(UP);
vU = vv(UP);
xD = x{1,1}(DOWN);
yD = y{1,1}(DOWN);
uD = uu(DOWN);
vD = vv(DOWN);
figure(23);
q1 = quiver(xU(1:step:end,1:step:end),yU(1:step:end,1:step:end),...
    uU(1:step:end,1:step:end),vU(1:step:end,1:step:end),l_len);
q1.Color = [.9 .2 .2];hold on
q2 =  quiver(xD(1:step:end,1:step:end),yD(1:step:end,1:step:end),...
    uD(1:step:end,1:step:end),vD(1:step:end,1:step:end),l_len);
q2.Color = [.2 .2 .9];
axis equal tight off
hold off
set(gca,'View',[-90 90])
title (['width = ', num2str(size(Ang,2)*.75*3)])

figure(24); cla
colorS = [min(abs(uu(:))) :.5: max(abs(uu(:)))];
ncquiverref(x{1,1}(1:step:end,1:step:end),y{1,1}(1:step:end,1:step:end)...
    ,uu(1:step:end,1:step:end),vv(1:step:end,1:step:end)...
    ,'um/hr','mean',1,'col',colorS);
axis equal tight off
set(gca,'View',[-90 90])
hold off
%% GET ANGLE PROFILE
i=11;
Sw = 400;
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);


% LOOP THIS REGION FOR MULTIPLE FILES
% --------------------------------------------------
filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
[av_Ang, av_Std_circ] = sumOrient(filepathOP);
xx = linspace(-Sw/2,Sw/2,size(av_Ang,2));
xq = linspace(-Sw/2,Sw/2,Sw*1.5);
Ang_spline  = spline(xx,circ_mean(av_Ang),xq);
Std_spline = spline(xx,circ_mean(av_Std_circ),xq);
% --------------------------------------------------

% PLOT
figure(FIG);subplot(1, 2, 1);cla
% errorbar(linspace(-Sw/2,Sw/2,size(av_Ang,2)),circ_mean(av_Ang)*180/pi,circ_mean(av_Std_circ)*180/pi); hold on
[hl, hp] = boundedline(xq,Ang_spline*180/pi,...
    Std_spline*180/pi,'alpha'); %/(size(Ang3,3))
set(gca,'Fontsize',18);
ylabel('$ Angle\ (deg) $','Interpreter','latex','FontSize',24);
xlabel('$ Width\ (\mu m) $','Interpreter','latex','FontSize',24);
axis square; axis tight;
% ----------  plot STD map -------------
figure(FIG);subplot(1, 2, 2);cla
imagesc(av_Std_circ);
axis equal; axis tight; 
load('mycbar.mat')
set(gcf,'Colormap',mycbar); %this works
hold on

% ----------  plot directors -------------
[l,w] = size(av_Ang);
[X,Y] = meshgrid(1:w,1:l);
step = 10;
len = .6;
q = quiver(X(1:step:end,1:step:end),Y(1:step:end,1:step:end),...
        cos(av_Ang(1:step:end,1:step:end)),...
        sin(av_Ang(1:step:end,1:step:end)),len);
q.Color = [0 0 0];
q.ShowArrowHead='off';
q.LineWidth=1;
hold off
% set(gca,'View',[-90 90])


%%




function [uu,vv] = sumPIV(filepathPIV,filt)
load(filepathPIV);

[L,W] = size(u{1,1});
T = size(u,1);
uu = zeros(size(u{1,1}));
vv = uu;

for k =1:T
    uu = uu + imfilter(u{k}, filt) - mean2(u{k});
    vv = vv + imfilter(v{k}, filt) - mean2(v{k});
end

uu = uu/T;
vv = vv/T;
end












