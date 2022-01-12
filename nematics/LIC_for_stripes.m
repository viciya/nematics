%%
PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
% PC_path = 'D:\GD\';                         % RSIP notebook

addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

[dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);

%% Orientation for SELECT FILES BY SPECIFIC WIDTH
addpath('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\codes');
% clearvars ps_x ps_y plocPsi_vec

i = 4; k = 60; %experiment (i) and time frame (k)
Sw = 400; % selectd width
dw = .05*Sw; % define delta
px_size = 0.74;
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);

filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];
disp(filepathOP);
disp(filepathPIV);

Ang = imread(filepathOP,k); % read Angle file

if any( Ang(:)>2 ) % check if Ang is in RAD
    Ang = Ang * pi/180;
end
% Ang = imrotate(Ang-pi/4,90);
nx = cos(Ang);
ny = -sin(Ang);
% %%
LIC = fun_getLIC(nx,ny); % get LIC from vector matrix

subplot(3,1,2); %cla
imageplot(LIC,''); hold on
% colormap(ax1,gray)
view([-90 90]);
axis on

% Orientation PLOT
[Xu,Yu] = meshgrid(1:size(Ang,2),1:size(Ang,1));
step = 7; O_len = .7;
q6 = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
    cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)),O_len);
q6.LineWidth=1; q6.Color = [.95 0 0]; q6.ShowArrowHead='off';

% Display defects
try
    [ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec] = ...
        fun_get_pn_Defects_newDefectAngle(Ang);
%         fun_get_pn_Defects_newDefectAngle_blockproc(Ang);
        
    plot_pdefect_ndefect(ps_x, ps_y, plocPsi_vec,...
                   ns_x, ns_y, nlocPsi_vec);
%     fun_defectDraw(ps_x, ps_y, plocPsi_vec);
catch
    disp('no defects')
end
 colormap(ax1,'gray')
axis off
hold off

title(['width = ', num2str(size(Ang,2)*.75*3),...
    ' (file:', num2str(i),' frame: ',num2str(k), ')'])

% ------ RAW image --------
subplot(3,1,1);
imshow(imread('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\400\89_3\s89_30060.tif'))
axis on equal tight
view([-90 90]);
% ------ PIV data --------
load(filepathPIV);
ff = 3;
filt = fspecial('gaussian',ff,ff);
    uu = px_size * (imfilter(u{k}, filt));
    uu = uu - mean2(uu);
    vv = px_size * (imfilter(v{k}, filt));
    vv = vv - mean2(vv);
    xx = px_size * x{k};
    yy = px_size * y{k};
    d=1;
subplot(3,1,3);
q2 = quiver(xx(1:d:end,1:d:end),yy(1:d:end,1:d:end),...
            uu(1:d:end,1:d:end),vv(1:d:end,1:d:end),3);
q2.LineWidth = .5;
q2.Color = [0,0,0];
axis equal tight
view([-90 90]);

% sfolder = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\Supplement\';
% name = ['v_x-89_3-f60','.txt'];
% dlmwrite([sfolder,name], uu,'delimiter','\t','precision',5);
% name = ['v_y-89_3-f60','.txt'];
% dlmwrite([sfolder,name], vv,'delimiter','\t','precision',5);
% ----------------------------------

% sfolder = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\Supplement\';
% sname_raw = ['89_3-f60','.txt'];
% dlmwrite([sfolder,sname_raw],Ang,'delimiter','\t','precision',5);
%% MAKE LIC FOR PIV FOR COLLECTION OF DIFFERENT FLOW REGEMS COLLECTION
% --------------------------------------------------------
i = 10; k=8; 
Sw = 800; % selectd width
dw = .1*Sw; % define delta
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);
filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
Ang = imread(filepathOP,k); % read Angle file
[Xu,Yu] = meshgrid(1:size(Ang,2),1:size(Ang,1));
% --------------------------------------------------------

filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];
load(filepathPIV);
px_sizePIV = 0.748;
frame_per_hr = 4;
px2mic = px_sizePIV * frame_per_hr;

dx = px_sizePIV*(x{1}(1,2)-x{1}(1,1));
ff = 5;
filt = fspecial('gaussian',ff,ff);
    uu = px2mic*imfilter(u{k}, filt);
    vv = px2mic*imfilter(v{k}, filt);
    [u_x,u_y] = gradient(uu,dx);%/dx gradient need to be corrected for the dx
    [v_x,v_y] = gradient(vv,dx);%/dx
    vorticity = (v_x - u_y);
    filtN = 5;
    filt = fspecial('gaussian',filtN,filtN);
    u1 = imfilter(vorticity, filt);
    
    
    sc = size(Ang,1)/size(uu,1);
%     [Xorig,Yorig] = meshgrid((1/sc:size(uu,2))*sc,(1/sc:size(uu,1))*sc);
%     [Xorig,Yorig] = meshgrid(sc*(1:size(uu,2)),sc*(1:size(uu,1)));
% 
%     u_interp = interp2(Xorig,Yorig,uu,Xu,Yu,'cubic',0);
%     v_interp = interp2(Xorig,Yorig,vv,Xu,Yu,'cubic',0);

% LIC = fun_getLIC(u_interp,v_interp); % get LIC from vector matrix
figure(22);
ax2 = subplot(2,1,1:2); %cla
% imageplot(LIC,'');hold on
% imageplot(zeros(size(uu))-.1);hold on

step = 1;
O_len = 1.5;

% PIV quiver
[Xu,Yu] = meshgrid(1:size(uu,2),1:size(uu,1));
s1 = surf(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),u1(1:step:end,1:step:end)+5);view(2);shading interp;colormap jet;axis equal;axis tight;hold on
    load('mycbar.mat'); caxis([4,6])
    colormap(ax2,mycbar)
    
q6 = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
    uu(1:step:end,1:step:end),vv(1:step:end,1:step:end),O_len);
q6.LineWidth=1; q6.Color = [.0 0 0];
view([90 -90]);
title (['width = ', num2str(size(Ang,2)*.75*3),' | <v> = ', num2str(mean2(v{k}),2), ])
axis off
hold off
%%
Sw = 1500; % selectd width
dw = .1*Sw; % define delta
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);

for i=1:size(Range,1)
    filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];
    load(filepathPIV);
    
    vprof = zeros(size(u{1},2),1);
    vt = zeros(size(u,1),1);
    uprof = zeros(size(u{1},2),1);
    ut = zeros(size(u,1),1);
    vorticity = zeros(size(u,1),1);
    for j=1:size(u,1)
        vprof = vprof+mean(v{j})';
        vt(j,1) = mean2(v{j});
        uprof = uprof+mean(u{j})';
        ut(j,1) = mean2(u{j});    
        
        [u_x,u_y] = gradient(u{j}(:,10:end-10));%/dx gradient need to be corrected for the dx
        [v_x,v_y] = gradient(v{j}(:,10:end-10));%/dx
%         [u_x,u_y] = gradient(u{j});%/dx gradient need to be corrected for the dx
%         [v_x,v_y] = gradient(v{j});%/dx        
        vorticity(j,1) = mean2((v_x - u_y));
        
    end
    vprof = vprof/j;
    uprof = uprof/j;
%     figure(Sw);
%     subplot(2,1,1); plot(x{1}(1,:),vprof,'.'); hold on
%     subplot(2,1,2); plot(1:j,vt,'.'); axis([0 100 -5 5]); hold on
%     figure(Sw+1);
%     subplot(2,1,1); plot(x{1}(1,:),uprof,'.'); hold on
%     subplot(2,1,2); plot(1:j,ut,'.'); axis([0 100 -5 5]); hold on    
    
    meanV(i,1) = mean(vprof);
    meanU(i,1) = mean(uprof);
    
    meanVort(i,1) = mean(vorticity);
end
meanVall = mean(meanV);
meanUall = mean(meanU);
meanVortAll = mean(meanVort);
figure(Sw);subplot(2,1,2);
title(meanVall)
figure(Sw+1);subplot(2,1,2);
title(meanUall)



