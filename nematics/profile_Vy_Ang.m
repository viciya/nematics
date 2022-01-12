%%  profile for specific width
% data set from 'correlate_OP_PIV_Defect.m'
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new3.mat');
% data set from 'energy_enstrophy.m' (without velocity filtering and net flow removed)
load("C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_Ek_Ev_unfiltered1.mat");

V_OP_mAng = [width, Ang_mid,vR-vL];
[V_OP_mAng, indByWidth] = sortrows(V_OP_mAng,1);

%% SMALL STRIPES, RUN TWICE FOR LEFT AND RIGHT
folderOP = 'C:\Users\victo\Google Drive\DATA\HT1080\Orient'; 

px_sizeOP = 3* .74;
frame_per_hr = 4;
px2mic = px_sizeOP * frame_per_hr;


Sw = 400; %/ px_sizeOP; RUN FOR Sw=190,Dw = 0.1  
Dw = 0.1 * Sw;
xax = (0:Sw)';

% Right tilt  (V_OP_mAng(:,2)>100 & V_OP_mAng(:,3)> 10)
sRangeL = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw &...
                    V_OP_mAng(:,2)>100 & V_OP_mAng(:,3)> 10);
% Left tilt  (V_OP_mAng(:,2)<85 & V_OP_mAng(:,3) < 10))
sRangeR = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw &...
                    V_OP_mAng(:,2)<85 & V_OP_mAng(:,3) < 10);
% All tilt
sRangeA = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw & ...
                    abs(V_OP_mAng(:,3)) > 6);
               

% Choose one Right/Left/All
sRange = sRangeL;

clearvars AngPMean AngStdMean AngCenter AngCenterMat AngCenterCell

for i=1:length(sRange)
    clearvars AngP AngStd AngCenter
    %    i=1
    filepathOP = [folderOP '\' dirOP(sRange(i)).name];
    info = imfinfo(filepathOP); % Place path to file inside single quotes
    Nn = numel(info); 
    info(1).Width*px_sizeOP
    mid = round(info(1).Width/2);
    for k=1:Nn
        Ang = imread(filepathOP,k); % k
        if ~any( Ang(:)>2 ) % chek if Ang is in RAD
            Ang = Ang * 180/pi;
        end
        Ang(Ang<0) = Ang(Ang<0)+180;        
        AngP(k,:) = mean(Ang) ;
    end
    xax_temp = px_sizeOP* (0:size(Ang,2)-1);
    AngPMean(i,:) = spline(xax_temp, mean(AngP), xax);
    AngStdMean(i,:) = spline(xax_temp, std(AngP), xax);
    AngCenterCell{i,1} = AngP(:,mid-5:mid+5);
    
%     AngCenterCell{i,1} = AngCenter(:);
%             figure(Sw+20);plot(xax, AngPMean(i,:)); hold on
%             figure(Sw+20);plot(xax_temp, mean(AngP),'o'); hold on
end
%
AngProfile = double(mean(AngPMean)');
AngProfileSTD = double(mean(AngStdMean)');%/i^.5;%
xax1 = xax-Sw/2;

AngCenterMat = cell2mat(AngCenterCell);

figure(Sw+1); ph = polarhistogram(AngCenterMat*pi/180,90,'BinLimits',[0 pi],'Normalization','PDF');hold on
ph.EdgeAlpha = 0; thetalim([0 180]);ax = gca; ax.RTickLabel = {''};set(gca,'Fontsize',18);
ph.FaceColor = [1 0 0];

figure(Sw+2); h1 = histogram(AngCenterMat,90,'Normalization','PDF'); hold on
h1.EdgeAlpha = 0; set(gca,'Fontsize',18); 
h1.FaceColor = [1 0 0];
ylabel('$ PDF $','Interpreter','latex','FontSize',24);
xlabel('$Central\ angle\ \langle \theta _{(x=0 \pm \delta)} \rangle _{y }$','Interpreter','latex','FontSize',24);
axis([60 120 0 inf]); set(gca,'xdir','reverse') ; 

figure(Sw);
[llE, hE] = boundedline(xax1,AngProfile,AngProfileSTD,'alpha');set(gca,'Fontsize',18);
hold on
llE.LineWidth = 2; llE.Color = [1,0.1,0.1]; hE.FaceColor = [1,0.1,0.1];
ylabel('$ Angle\ (deg) $','Interpreter','latex','FontSize',28);
xlabel('$ Width\ (\mu m) $','Interpreter','latex','FontSize',28);
axis([xax1(1) xax1(end) 80 105]);
% errorbar(xax,AngProfile,AngProfileSTD); 
% axis tight
% hold off

%% WIDE STRIPES, RUN TWICE FOR LEFT AND RIGHT
folderOP = 'C:\Users\victo\Google Drive\DATA\HT1080\Orient'; 

px_sizeOP = 3* .74;
frame_per_hr = 4;
px2mic = px_sizeOP * frame_per_hr;


Sw = 400; %/ px_sizeOP; RUN FOR Sw=280,Dw = 0.1  V_OP_mAng(:,3)> 3
Dw = 0.05 * Sw;
xax = (0:3:Sw)';

% Right tilt 
sRangeL = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw &...
                    V_OP_mAng(:,2)>90 & V_OP_mAng(:,3)> 0);
% Left tilt
sRangeR = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw &...
                    V_OP_mAng(:,2)<90 & V_OP_mAng(:,3) < 0);
% All tilt
sRangeA = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw & ...
                    abs(V_OP_mAng(:,3)) > 6);

% Choose one Right/Left/All
sRange = sRangeL;

clearvars AngPMean AngStdMean AngCenter AngCenterMat AngCenterCell

for i=1:length(sRange)
    clearvars AngP AngStd AngCenter
    %    i=1
    filepathOP = [folderOP '\' dirOP(sRange(i)).name];
    info = imfinfo(filepathOP); % Place path to file inside single quotes
    Nn = numel(info); 
    info(1).Width*px_sizeOP
    mid = round(info(1).Width/2);
    for k=1:Nn
        Ang = imread(filepathOP,k); % k
        if ~any( Ang(:)>2 ) % chek if Ang is in RAD
            Ang = Ang * 180/pi;
        end
        Ang(Ang<0) = Ang(Ang<0)+180;        
        AngP(k,:) = mean(Ang) ;
    end
    xax_temp = px_sizeOP* (0:size(Ang,2)-1);
    AngPMean(i,:) = spline(xax_temp, mean(AngP), xax);
    AngStdMean(i,:) = spline(xax_temp, std(AngP), xax);
    AngCenterCell{i,1} = AngP(:,mid-5:mid+5);
    
%     AngCenterCell{i,1} = AngCenter(:);
%             figure(Sw+20);plot(xax, AngPMean(i,:)); hold on
%             figure(Sw+20);plot(xax_temp, mean(AngP),'o'); hold on
end
%
AngProfile = double(mean(AngPMean)');
AngProfileSTD = double(mean(AngStdMean)');%/i^.5;%
xax1 = xax-Sw/2;

AngCenterMat = cell2mat(AngCenterCell);

figure(Sw+10); ph = polarhistogram(AngCenterMat*pi/180,90,'BinLimits',[0 pi],'Normalization','PDF');set(gca,'Fontsize',18);hold on
ph.EdgeAlpha = 0; thetalim([0 180]);ax = gca; ax.RTickLabel = {''};
ph.FaceColor = [.47 .67 .19];

figure(Sw+2); h1 = histogram(AngCenterMat,90,'Normalization','PDF'); hold on
h1.EdgeAlpha = 0; set(gca,'Fontsize',18); 
h1.FaceColor = [.47 .67 .19];
ylabel('$ PDF $','Interpreter','latex','FontSize',24);
xlabel('$Central\ angle\ \langle \theta _{(x=0 \pm \delta)} \rangle _{y }$','Interpreter','latex','FontSize',24);
axis([60 120 0 0.08]); set(gca,'xdir','reverse') ; 

figure(Sw);
[llE, hE] = boundedline(xax1,AngProfile,AngProfileSTD,'alpha');set(gca,'Fontsize',18);
hold on
llE.LineWidth = 2; llE.Color = [.47 .67 .19]; hE.FaceColor = [.47 .67 .19];
ylabel('$ Angle\ (deg) $','Interpreter','latex','FontSize',28);
xlabel('$ Width\ (\mu m) $','Interpreter','latex','FontSize',28);
axis([xax1(1) xax1(end) 84 96]);
% errorbar(xax,AngProfile,AngProfileSTD); 
% axis tight
% hold off

%% WIDE STRIPES, RUN TWICE FOR LEFT AND RIGHT
% -------  VELOCITY PROFILE ---------

folderPIV = 'C:\Users\victo\Google Drive\DATA\HT1080\PIV_DATs'; 
folderOP = 'C:\Users\victo\Google Drive\DATA\HT1080\Orient'; 
clearvars u_interp v_interp Up Vp

px_sizePIV = .74;
px_sizeOP = 3*.74;
frame_per_hr = 4;
px2mic = px_sizeOP * frame_per_hr;

ff = 1;
filt = fspecial('gaussian',ff,ff);

Sw = 400; %/ px_sizeOP; RUN FOR Sw=280,Dw = 0.1  V_OP_mAng(:,3)> 3
Dw = 0.07 * Sw;

% Right tilt 
sRangeL = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw &...
                     V_OP_mAng(:,3)> 3);
                 
sRangeL1 = indX(sRangeL,2);
                
% Left tilt
sRangeR = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw &...
                     V_OP_mAng(:,3) < -1);
sRangeR1 = indX(sRangeR,2);
% All tilt
sRangeA = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw & ...
                    abs(V_OP_mAng(:,3)) > 6);
sRangeA1 = indX(sRangeA,2);                

% Choose one Right/Left/All
sRange = sRangeL1;
dd = 5; %  interpolated resolution
%
for i=1:length(sRange)
    %    i=1
    filepathPIV = [folderPIV '\' dirPIV(sRange(i)).name];
    load(filepathPIV);
    [L,W] = size(u{1,1});
    T = size(u,1);
    
    widthS(i,1) = x{1}(1,end)*px_sizePIV;
    
    clearvars u_t v_t
    
    for k =1:size(x,1)
        uf = imfilter(u{k}, filt);
        vf = imfilter(v{k}, filt);
        u_t(k,:) = px_sizePIV*(mean(uf)- mean2(uf));
        v_t(k,:) = px_sizePIV*(mean(vf)- mean2(vf));
%         u_std = u_std + std(u{k});
%         v_std = v_std + std(v{k});
    end
    xax_temp = px_sizePIV*x{1}(1,1:end);
    xax = xax_temp(1):dd:xax_temp(end);
%     figure(1); plot(xax_temp-mean(xax_temp), mean(u_t),'o'); hold on
    u_tt = pchip(px_sizePIV*x{1}(1,1:end), mean(u_t),xax);
    Up{i,1} = u_tt;
%     figure(1); plot(xax-mean(xax), u_tt)
    
%     figure(2); plot(xax_temp, mean(v_t),'o'); hold on
    v_tt = pchip(px_sizePIV*x{1}(1,1:end), mean(v_t),xax);
    Vp{i,1} = v_tt;
    Xp{i,1} = xax-mean(xax);
%     figure(2); plot(xax, v_tt)
end

% make of profiles of same size by streching the to maximal width
xax_interp = 0:max(cellfun('size',Vp,2))* dd; 

for i=1:size(Vp,1)
     dl = (xax_interp(end) - xax_interp(1))/(size(Vp{i},2)-1);
     xax_temp = xax_interp(1):dl :xax_interp(end);
     u_interp(i,:) = pchip(xax_temp, Up{i},xax_interp);
     v_interp(i,:) = pchip(xax_temp, Vp{i},xax_interp);
end

%  XaN = (xax_interp-mean(xax_interp))/max(xax_interp);% Normalized width
XaN = (xax_interp-mean(xax_interp)); % Non Normalized width  
% figure(Sw)
% plot(XaN, v_interp,'color',[.8,.8,.8,.6],'LineWidth',2);hold on
% figure(Sw+1)
% plot(XaN, u_interp,'color',[.8,.8,.8,.6],'LineWidth',2);hold on
% 
Sw=111
figure(Sw);
[llE, hE] = boundedline(1*XaN,mean(v_interp),std(v_interp),'alpha');set(gca,'Fontsize',18);
hold on
ylabel('$ \langle v_{y}(x) \rangle _{y} \ ( \mu m / h )$','Interpreter','latex','FontSize',24);
% xlabel('$ x / L  $','Interpreter','latex','FontSize',24);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
axis tight square
xlim([-200,200]);ylim([-2,2])
set(gcf,'Color',[1 1 1]);
llE.LineWidth = 2; llE.Color = [.47 .67 .19]; hE.FaceColor = [.47 .67 .19];

figure(Sw+1);
[llA, hA] = boundedline(1*XaN,mean(u_interp),std(u_interp),'alpha');set(gca,'Fontsize',18);
hold on
llA.LineWidth = 2; llA.Color = [1 0 0]; hA.FaceColor = [1 0 0];
ylabel('$ \langle v_{x}(x) \rangle _{y} \ ( \mu m / h )$','Interpreter','latex','FontSize',24);
% xlabel('$ x / L $','Interpreter','latex','FontSize',24);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
xlim([-200,200]);ylim([-3,3])
% errorbar(xax,AngProfile,AngProfileSTD); 
axis tight square
set(gcf,'Color',[1 1 1]);
% hold off
%% save profiles to txt
sfolder = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\v_profile\';
mkdir(sfolder)
sname_u = ['Uprofile_w_',num2str(Sw),'.txt'];
dlmwrite([sfolder,sname_u],u_interp','delimiter','\t','precision',3)
sname_v = ['Vprofile_w_',num2str(Sw),'.txt'];
dlmwrite([sfolder,sname_v],v_interp','delimiter','\t','precision',3)
sname_uav = ['Uav_w_',num2str(Sw),'.txt'];
dlmwrite([sfolder,sname_uav],[XaN',mean(u_interp)',std(u_interp)'],'delimiter','\t','precision',3)
sname_vav = ['Vav_w_',num2str(Sw),'.txt'];
dlmwrite([sfolder,sname_vav],[XaN',mean(v_interp)',std(v_interp)'],'delimiter','\t','precision',3)
figure(Sw);
saveas(gcf,[[sfolder,sname_vav],'.png'])
figure(Sw+1);
saveas(gcf,[[sfolder,sname_uav],'.png'])

%% WIDE STRIPES, RUN TWICE FOR LEFT AND RIGHT
% -------  Ek/Ew PROFILE ---------

folderPIV = 'C:\Users\victo\Google Drive\DATA\HT1080\PIV_DATs'; 
folderOP = 'C:\Users\victo\Google Drive\DATA\HT1080\Orient'; 
clearvars u_interp v_interp Up Vp ratio_Ek_Ev

px_sizePIV = .74;
frame_per_hr = 4;

px2mic = px_sizeOP * frame_per_hr;


Sw = 1000; %/ px_sizeOP; RUN FOR Sw=280,Dw = 0.1  V_OP_mAng(:,3)> 3
Dw = 0.05 * Sw;

% Right tilt 
% % sRangeL = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw);
sRangeL = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw &...
                     V_OP_mAng(:,3)> 3);
sRangeL1 = indX(sRangeL,2);          
   

% Choose one Right/Left/All
sRange = sRangeL1;
dd = 5; %  interpolated resolution
x_ax = -.5:.01:.5;

for i=1:length(sRange)
    %    i=1
    filepathPIV = [folderPIV '\' dirPIV(sRange(i)).name];
    load(filepathPIV);
    [L,W] = size(u{1});
    T = size(u,1);
    
    widthS(i,1) = x{1}(1,end)*px_sizePIV;
    dx = px_sizePIV*(x{1}(1,2)-x{1}(1,1));  
    clearvars u_t v_t
    vEnergy_temp = zeros(size(u{1}));
    kEnergy_temp = zeros(size(u{1}));
    av_v = zeros(1,size(v{1},2));
    for k =1:size(x,1)
        uu = px_sizePIV*u{k};
        vv = px_sizePIV*v{k};
        % Vorticity
        [u_x,u_y] = gradient(uu,dx);%/dx gradient need to be corrected for the dx
        [v_x,v_y] = gradient(vv,dx);%/dx
        vEnergy_temp = vEnergy_temp + 0.5*(v_x - u_y).^2;
        kEnergy_temp = kEnergy_temp + 0.5*(uu.^2 + vv.^2);
        av_v=av_v+mean(vv)-mean2(vv);
    end    
%     vEnergy{i,1} = vEnergy_temp/k;
%     kEnergy{i,1} = kEnergy_temp/k;
    
    temp = mean(kEnergy_temp./vEnergy_temp,1);
%     figure;plot(temp,'o')
    x_ax = -.5:.01:.5;
    x_prof  = ((0:size(temp,2)-1)-(size(temp,2)-1)/2)/size(temp,2);
    ratio_Ek_Ev(i,:) = sqrt(interp1(x_prof,mean(temp,1),x_ax)/310);    
%     figure(400);
%     plot(Sw*x_ax,ratio_Ek_Ev(i,:),'color',[.8,.8,.8,.6],'LineWidth',2); hold on
end

figure(Sw);
%     plot(Sw*x_ax,ratio_Ek_Ev,'color',[.8,.8,.8,.6],'LineWidth',2); hold on

[llA, hA] = boundedline(Sw*x_ax,mean(ratio_Ek_Ev),std(ratio_Ek_Ev),'alpha');set(gca,'Fontsize',18);
hold on
llA.LineWidth = 2; llA.Color = [1 0 0]; hA.FaceColor = [1 0 0];
ylabel('$ \sqrt{ (E_{k} / \Omega) / (E_{k}^{free} / \Omega^{free})} $',...
    'Interpreter','latex','FontSize',24);
% xlabel('$ x / L $','Interpreter','latex','FontSize',24);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
xlim([-200,200]);
% % ylim([-3,3])
% errorbar(xax,AngProfile,AngProfileSTD); 
axis    tight square
set(gcf,'Color',[1 1 1]);
hold off
% %% save profiles to txt
sfolder = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\Ek_vs_Ew_profile\';
mkdir(sfolder)
sname = ['Ek_vs_Ew_',num2str(Sw),'.txt'];
dlmwrite([sfolder,sname],ratio_Ek_Ev','delimiter','\t','precision',3)

sname_av = ['av_Ek_vs_Ew__',num2str(Sw),'.txt'];
dlmwrite([sfolder,sname_av],[Sw*x_ax',mean(ratio_Ek_Ev)',std(ratio_Ek_Ev)'],'delimiter','\t','precision',3)

figure(Sw);
saveas(gcf,[[sfolder,sname],'.png'])

%%
% -------  DEFECT DENSITY PROFILE ---------

folderPIV = 'C:\Users\victo\Google Drive\DATA\HT1080\PIV_DATs'; 
folderOP = 'C:\Users\victo\Google Drive\DATA\HT1080\Orient'; 

px_sizeOP = 3* .74;
px_sizePIV = .74;
frame_per_hr = 4;
px2mic = px_sizeOP * frame_per_hr;
wset = [300,400,500,600,700,800,1000];
for k=1:length(wset)
clearvars def_density Q S
% Sw = 10*100; %/ px_sizeOP; RUN FOR Sw=280,Dw = 0.1  V_OP_mAng(:,3)> 3
Sw = wset(k); 
Dw = 0.07 * Sw;

histbins = round(Sw/10);
bins = linspace(-1,1,histbins);
xax_norm = linspace(-1,1,histbins-1);
xax = Sw/2*xax_norm;

% Right tilt 
sRangeL = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw &...
                     V_OP_mAng(:,3)> 3);
sRangeL1 = indX(sRangeL,1);

% Choose one Right/Left/All
sRange = sRangeL1;
S=0;

for i=1:length(sRange)
    Q=cell2mat(defPOS{sRange(i)});
    %    shift center to 0 and normalize: (x-w/2)/w
    xpos_norm = (Q(:,1)- mean(Q(:,1)))/mean(Q(:,1));
    [N,~] = histcounts(xpos_norm,bins); %,'Normalization', 'probability'
    x_def_pos{i,1} = xpos_norm;    
    def_density(i,:) = N/sum(N);
    S = S + sum(N);
end
figure(3)
plot(Sw,S/i/Sw,'-o'); hold on


% figure(1)
% cla
% plot(xax,def_density,'color',[.8,.8,.8,.6],'LineWidth',2);hold on 
% 
% % [N,~] = histcounts(cell2mat(x_def_pos),bins);% ,'Normalization', 'probability'
% % x_ax = mean([edges(1:end-1);edges(2:end)])*Sw;
% 
% 
% [llA, hA] = boundedline(xax,mean(def_density),std(def_density),'alpha');set(gca,'Fontsize',18);
% hold on
% llA.LineWidth = 2; llA.Color = [1 0 0]; hA.FaceColor = [1 0 0];
% 
% 
% % %% save profiles to txt
% sfolder = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\defect_density_profile_unnorm\';
% mkdir(sfolder)
% sname = ['def_density_',num2str(Sw),'.txt'];
% dlmwrite([sfolder,sname],def_density','delimiter','\t','precision',3)
% 
% sname_av = ['av_def_density__',num2str(Sw),'.txt'];
% dlmwrite([sfolder,sname_av],[xax',mean(def_density)',std(def_density)'],'delimiter','\t','precision',3)
% % 
% ylabel('$Number \ of \ defects$','Interpreter','latex','FontSize',24);
% % xlabel('$ x / L $','Interpreter','latex','FontSize',24);
% xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
% xlim([-Sw/2,Sw/2]);ylim([0,inf])
% % errorbar(xax,AngProfile,AngProfileSTD); 
% axis  square
% set(gcf,'Color',[1 1 1]);
% 
% figure(1);
% saveas(gcf,[[sfolder,sname],'.png'])

end

%% ORIENTATION COS2Q SIN2Q
folderOP = 'C:\Users\victo\Google Drive\DATA\HT1080\Orient'; 

px_sizeOP = 3 * .74;
frame_per_hr = 4;
px2mic = px_sizeOP * frame_per_hr;

wset = [500,600,700,800,1000];%[300,400]%,
for k=1:length(wset)
    
   clearvars cos2_prof_all sin2_prof_all

Sw = wset(k); 
Dw = 0.07 * Sw;

histbins = round(Sw/10);
bins = linspace(-1,1,histbins);
xax_norm = linspace(-1,1,histbins-1);
xax = Sw/2*xax_norm;

xq = linspace(0,1,2*Sw/3);

% Right tilt 
sRangeL = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw &...
                    V_OP_mAng(:,3)> 0);
                
% Choose one Right/Left/All
sRange = sRangeL;

clearvars sin2_prof_all cos2_prof_all order_parameter

for i=1:min(length(sRange),30)
    disp([num2str(i), ' from: ', num2str(length(min(length(sRange),30)))])
    clearvars sin2q cos2q 
    %    i=1
    filepathOP = [folderOP '\' dirOP(sRange(i)).name];
    info = imfinfo(filepathOP); % Place path to file inside single quotes
    Nn = numel(info);
%     info(1).Width*px_sizeOP
    
    for k=1:Nn
        Ang = imread(filepathOP,k); % k
        if ~any( Ang(:)>2 ) % chek if Ang is in RAD
            Ang = Ang * 180/pi;
        end
        Ang(Ang<0) = Ang(Ang<0)+180;
        
        
        sin2q(:,:,k) = sind(2*Ang);
        cos2q(:,:,k) = cosd(2*Ang);
    end
    
    sin2_prof = mean(mean(sin2q,3),1);
    cos2_prof = mean(mean(cos2q,3),1);
    sin2_std = std(std(sin2q,[],3),[],1);
    cos2_std = std(std(cos2q,[],3),[],1);   
%     plot(sin2_prof,'r');hold on
%     plot(cos2_prof,'b');hold on
    
    x_temp = linspace(0,1,length(sin2_prof));
    sin2_prof_all(i,:) = interp1(x_temp,sin2_prof,xq); 
    cos2_prof_all(i,:) = interp1(x_temp,cos2_prof,xq);
    
    
    order_parameter(i,:) = interp1(x_temp,sqrt(cos2_prof.^2 +sin2_prof.^2),xq);
    
end

x_ax = (xq-.5)*Sw;

% figure(11); cla
% plot(x_ax,sin2_prof_all,'color',[.8,.8,.8,.6],'LineWidth',2);hold on
% [llA, hA] = boundedline(x_ax,mean(sin2_prof_all),std(sin2_prof_all),'alpha');set(gca,'Fontsize',18);
% llA.LineWidth = 2; llA.Color = [0 0 1]; hA.FaceColor = [0 0 1];
% ylabel('$<sin(2 \theta)> _{y,t} $','Interpreter','latex','FontSize',24);
% xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
% xlim([-Sw/2,Sw/2]);%
% ylim([-.3,.3])
% axis  square
% set(gcf,'Color',[1 1 1]);
% 
% figure(12); cla
% plot(x_ax,cos2_prof_all,'color',[.8,.8,.8,.6],'LineWidth',2);hold on
% [llA, hA] = boundedline(x_ax,mean(cos2_prof_all),std(cos2_prof_all),'alpha');set(gca,'Fontsize',18);
% llA.LineWidth = 2; llA.Color = [1 0 0]; hA.FaceColor = [1 0 0];
% ylabel('$<cos(2 \theta)> _{y,t} $','Interpreter','latex','FontSize',24);
% xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
% xlim([-Sw/2,Sw/2]);ylim([-1,.5])
% axis  square
% set(gcf,'Color',[1 1 1]);

figure(13); cla
plot(x_ax,order_parameter,'color',[.8,.8,.8,.6],'LineWidth',2);hold on
[llA, hA] = boundedline(x_ax,mean(order_parameter),std(order_parameter),'alpha');set(gca,'Fontsize',18);
llA.LineWidth = 2; llA.Color = [1 0 0]; hA.FaceColor = [1 0 0];
ylabel('$\Sigma $','Interpreter','latex','FontSize',24);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
xlim([-Sw/2,Sw/2]);ylim([0,1])
axis  square
set(gcf,'Color',[1 1 1]);

figure(14); 
[llA, hA] = boundedline(x_ax,mean(order_parameter),std(order_parameter),'alpha');set(gca,'Fontsize',18);
llA.LineWidth = 2; hold on

% save to data anf fig
sfolder = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\order_parameter\';
mkdir(sfolder)
sname1 = ['av_op_',num2str(Sw),'.txt'];
dlmwrite([sfolder,['av_',sname1]],[x_ax',mean(order_parameter)',std(order_parameter)'],...
    'delimiter','\t','precision',3);

figure(13);
saveas(gcf,[[sfolder,sname1],'.png'])
end

figure(14); 
ylabel('$\Sigma $','Interpreter','latex','FontSize',24);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
xlim([-Sw/2,Sw/2]); ylim([0,1])
axis  square
set(gcf,'Color',[1 1 1]);
saveas(gcf,[sfolder,'_all.png'])

%% 
% -------  OREDER PARAMETER PROFILE ---------
qstep=15;
x_ax = -.5:.01:.5;
clearvars interp_op_profile

folderPIV = 'C:\Users\victo\Google Drive\DATA\HT1080\PIV_DATs'; 
folderOP = 'C:\Users\victo\Google Drive\DATA\HT1080\Orient'; 

px_sizeOP = 3* .74;
px_sizePIV = .74;
frame_per_hr = 4;
px2mic = px_sizeOP * frame_per_hr;
wset = [300,400,500,600,700,800,1000];
k=1;

Sw = wset(k); 
Dw = 0.07 * Sw;

histbins = round(Sw/10);
bins = linspace(-1,1,histbins);
xax_norm = linspace(-1,1,histbins-1);
xax = Sw/2*xax_norm;

% Right tilt 
sRangeL = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw &...
                     V_OP_mAng(:,3)> 3);
sRangeL1 = indX(sRangeL,1);

% Choose one Right/Left/All
sRange = sRangeL1;
S=0;

for i=1:3%length(sRange)
    %    i=1
    filepathOP = [folderOP '\' dirOP(sRange(i)).name];
    info = imfinfo(filepathOP); % Place path to file inside single quotes
    Nn = numel(info);
    info(1).Width*px_sizeOP;
    clearvars op_profile
    for k=1:Nn
        Ang = imread(filepathOP,k);
        if any( Ang(:)>4 ) % check if Ang is in RAD
            Ang = Ang * pi/180;
        end
        
        qq = ordermatrixglissant_overlap(Ang,qstep,7);
        qqf = ordermatrixglissant_overlap(fliplr(Ang),qstep,7);        
        op_profile(k,:) = mean(qq+fliplr(qqf),1)/2;
        
    end
    x_prof  = ((1:size(op_profile,2))-size(op_profile,2)/2)/size(op_profile,2);
    interp_op_profile(i,:) = interp1(x_prof,mean(op_profile,1),x_ax);
    
end

figure(Sw)
plot(x_ax*Sw,interp_op_profile,'color',[.8,.8,.8,.6],'LineWidth',2);hold on

all_av_profile = mean(interp_op_profile,1);
all_std_profile = std(interp_op_profile,1);

[llA, hA] = boundedline(x_ax*Sw,all_av_profile,all_std_profile,'alpha');set(gca,'Fontsize',18);
hold on
llA.LineWidth = 2; llA.Color = [1 0 0]; hA.FaceColor = [1 0 0];
ylabel('$Order \ paremeter $','Interpreter','latex','FontSize',24);
% xlabel('$ x / L $','Interpreter','latex','FontSize',24);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
xlim([-Sw/2,Sw/2]);%ylim([0,1])
% errorbar(xax,AngProfile,AngProfileSTD); 
axis  square
set(gcf,'Color',[1 1 1]);

%% HISTOGRAM Edge and Mid
folderOP = 'C:\Users\victo\Google Drive\DATA\HT1080\Orient'; 

px_sizeOP = 3* .74;
frame_per_hr = 4;
px2mic = px_sizeOP * frame_per_hr;


Sw = 400; %/ px_sizeOP; RUN FOR Sw=280,Dw = 0.1  V_OP_mAng(:,3)> 3
Dw = 0.05 * Sw;
xax = (0:3:Sw)';
sRange = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw &...
                    V_OP_mAng(:,2)>90 & V_OP_mAng(:,3)> 0);

count = 1;
edge = 10;
clearvars edge_AngR edge_AngL edge_Ang mid_Ang
for i=1:length(sRange)
    %    i=1
    filepathOP = [folderOP '\' dirOP(sRange(i)).name];
    info = imfinfo(filepathOP); % Place path to file inside single quotes
    Nn = numel(info);
    info(1).Width*px_sizeOP;
    clearvars op_profile
    for k=1:Nn
        Ang = imread(filepathOP,k);
        if any( Ang(:)>4 ) % check if Ang is in RAD
            Ang = Ang * pi/180;
        end
        Ang(Ang<0) = Ang(Ang<0)+pi; 
%         temp1=[Ang(:,1:edge);Ang(:,end-edge+1:end)];
        edge_Ang{count} = [Ang(:,1:edge);Ang(:,end-edge+1:end)];
        edge_AngL{count} = Ang(:,1:edge);
        edge_AngR{count} = Ang(:,end-edge+1:end);
%         temp2=Ang(:,round(end/2)-edge:round(end/2)+edge);
        mid_Ang{count} = Ang(:,round(end/2)-edge:round(end/2)+edge);
        count = count+1;
    end
end

edge_Ang_mat = cell2mat(edge_Ang(:));
mid_Ang_mat = cell2mat(mid_Ang(:));

edge_AngL_mat = cell2mat(edge_AngL(:));
edge_AngR_mat = cell2mat(edge_AngR(:));

% %% PLOT HISTOGRAM
% edge_mat = edge_Ang_mat(:);
% 
% figure(Sw+4); 
% ph1 = polarhistogram(mid_Ang_mat,180,'BinLimits',[0 pi],'Normalization','PDF');hold on
% ph1.EdgeAlpha = 0; ph1.FaceColor = [.8,.1,.1]; ph1.FaceAlpha=.3;
% 
% ph2 = polarhistogram(edge_mat,180,'BinLimits',[0 pi],'Normalization','PDF');hold on
% ph2.EdgeAlpha = 0; ph2.FaceColor = [.1,.1,.8]; ph2.FaceAlpha=.3;
% % hold off
% title([num2str(circ_mean(edge_mat)*180/pi),...
%     '--',...
%     num2str(circ_std(edge_mat)*180/pi)]);
% thetalim([-10 190]);ax = gca; ax.RTickLabel = {''};set(gca,'Fontsize',18);
% set(gcf,'Color',[1 1 1]);
% %% PLOT HISTOGRAM AS LINE
edge_mat = edge_Ang_mat(:);

figure(Sw+3);
[Nm,edges] = histcounts(mid_Ang_mat,180,'BinLimits',[0 pi], 'Normalization', 'probability');
theta_mid = mean([edges(1:end-1);edges(2:end)]);
pp = polarplot(theta_mid,Nm);hold on
pp.LineWidth=2;pp.Color=[.8,.1,.1];

[Nedge,edges] = histcounts(edge_mat,180,'BinLimits',[0 pi], 'Normalization', 'probability');
theta_edge = mean([edges(1:end-1);edges(2:end)]);
pp = polarplot(theta_edge,Nedge);
pp.LineWidth=2;pp.Color=[.1,.1,.8];

thetalim([-10 190]);ax = gca; ax.RTickLabel = {''};set(gca,'Fontsize',18);
set(gcf,'Color',[1 1 1]);
legend({'midline',['edge', sprintf('\n'),...
    'mean: ', num2str(circ_mean(edge_mat)*180/pi),sprintf('\n'),...
    'std: ', num2str(circ_std(edge_mat)*180/pi)]}...
    ,'Fontsize',13, 'Location', 'southoutside');
%%
sfolder = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\edge_mid_ang\';
mkdir(sfolder)
sname_edge = ['edge_','.txt'];
dlmwrite([sfolder,sname_edge],[theta_edge',Nedge'],'delimiter','\t','precision',3);
sname_mid = ['mid_','.txt'];
dlmwrite([sfolder,sname_mid],[theta_mid',Nm'],'delimiter','\t','precision',3);
figure(Sw+3);
saveas(gcf,[[sfolder,'edge_mid_polar_plot'],'.png'])
%%
figure(Sw+1); 
ph1 = polarhistogram(edge_AngL_mat,180,'BinLimits',[0 2*pi],'Normalization','PDF');hold on
ph1.EdgeAlpha = 0; ph1.FaceColor = [.8,.1,.1];ph1.FaceAlpha=.3;
ph2 = polarhistogram(edge_AngR_mat,180,'BinLimits',[0 2*pi],'Normalization','PDF');hold on
ph2.EdgeAlpha = 0; 
ph2.FaceColor = [.1,.1,.8];ph2.FaceAlpha=.3;
hold off
thetalim([-10 190]);ax = gca; ax.RTickLabel = {''};set(gca,'Fontsize',18);
set(gcf,'Color',[1 1 1]);