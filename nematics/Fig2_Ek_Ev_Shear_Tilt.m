%%
PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
% PC_path = 'D:\GD\';                         % RSIP notebook

% data set from 'correlate_OP_PIV_Defect.m'
load([PC_path,'Curie\DESKTOP\HT1080\shear_OP_correlation_new3.mat']);
% data set from 'energy_enstrophy.m' (without velocity filtering and net flow removed)
load([PC_path,'Curie\DESKTOP\HT1080\shear_Ek_Ev_unfiltered1.mat']);

Ek = wEW(:,2);
Ev = wEW(:,3);

V_OP_mAng = [width, vR, vL, Ek, Ev, Ang_mid];
[V_OP_mAng, indByWidth] = sortrows(V_OP_mAng,1);
% scatter(V_OP_mAng(:,1),V_OP_mAng(:,7),15,[.9 .2 .3],'filled');
%% ENERGY/ENSTROPHY RATIO
dBins = 5;% Bin resolution
dd = 1*dBins;% range
totalBins = floor((max(width)- min(width))/dBins);
[N, edges] = histcounts(width,totalBins);
edgess = [edges(1)-dBins, mean([edges(1:end-1); edges(2:end)]), edges(end)] ;
N = [0 N 0];
[Peak,widthOfPeak] = findpeaks(N,edgess);
% plot(edgess,N,'-'); hold on
% plot(widthOfPeak,Peak,'o'); hold off

width = V_OP_mAng(:,1);
% ENERGY/ENSTROPHY^1/2=typical vortex scale
vortSC = sqrt(V_OP_mAng(:,4)./V_OP_mAng(:,5));

AV = zeros(length(widthOfPeak),4);
AV(:,1) = widthOfPeak;

for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    AV(i,2) = mean(vortSC(width>=ww-dd & width<=ww+dd));  
    AV(i,3) = std(vortSC(width>=ww-dd & width<=ww+dd));%./sum(width>=ww-dd & width<=ww+dd)^.5;
    AV(i,4) = sum(width>=ww-dBins & width<=ww+dBins);
end

figure(143);
scatter(V_OP_mAng(:,1),vortSC/AV(end,2),15,[.8 .8 .8],'filled');set(gca,'Fontsize',18);hold on
[llB, ~] = boundedline(AV(:,1),AV(:,2)/AV(end,2),AV(:,3)/AV(end,2), 'alpha');
llB.LineWidth = 2; llB.Color = [0.1,0.5,0.8];
ylabel('$(\frac{E_{k}} {\Omega})^ {-\frac{1}{2}} / (\frac{E_{k}^{free}} {\Omega^{free}})^ {-\frac{1}{2}}$','Interpreter','latex','FontSize',28);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
axis([0 1040 0 1.01]);
% errorbar(AV(:,1),AV(:,2),AV(:,3));
hold off

%% SHEAR 

dBins = 4.9;%4.8;% Bin resolution
dd = 1*dBins;% range
totalBins = floor((max(width)- min(width))/dBins);
[N, edges] = histcounts(width,totalBins);
edgess = [edges(1)-dBins, mean([edges(1:end-1); edges(2:end)]), edges(end)] ;
N = [0 N 0];
[Peak,widthOfPeak] = findpeaks(N,edgess);

% figure(333); plot(edgess,N,'-'); hold on
% plot(widthOfPeak,Peak,'o'); hold off

% Exclude last experiments without stripes
lst = 1322; % makes a limit by width
V_OP_mAng = V_OP_mAng(1:lst,:);

width = V_OP_mAng(:,1);
shear = V_OP_mAng(:,2)-V_OP_mAng(:,3);

overlap = 0;
widthCW = width(shear>0-overlap);
shearCW = shear(shear>0-overlap);

widthCCW = width(shear<0+overlap);
shearCCW = shear(shear<0+overlap);

shearAll = abs(shear);

AV = zeros(length(widthOfPeak),8);
AV(:,1) = widthOfPeak;

for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    ddd = 0.19*ww;
    AV(i,2) = mean(shearCW(widthCW>=ww-ddd & widthCW<=ww+ddd));  
%     AV(i,3) = std(shearCW(widthCW>=ww-ddd & widthCW<=ww+ddd));
    AV(i,3) = std(shearCW(widthCW>=ww-ddd & widthCW<=ww+ddd))...
        ./sum(widthCW>=ww-ddd & widthCW<=ww+ddd)^.5;

    AV(i,4) = mean(shearCCW(widthCCW>=ww-ddd & widthCCW<=ww+ddd));
%     AV(i,5) = std(shearCCW(widthCCW>=ww-ddd & widthCCW<=ww+ddd));
    AV(i,5) = std(shearCCW(widthCCW>=ww-ddd & widthCCW<=ww+ddd))...
        /sum(widthCCW>=ww-ddd & widthCCW<=ww+ddd)^.5;
 
    AV(i,6) = mean(shearAll(width>=ww-ddd & width<=ww+ddd));  
    AV(i,7) = std(shearAll(width>=ww-ddd & width<=ww+ddd)) ...
        /sum(width>=ww-ddd & width<=ww+ddd).^.5; 
end


figure(144);%subplot(2,1,1);
cla;
% sA = scatter(widthCW, shearCW ,15,[0.1,0.5,0.8],'filled');hold on
% sA.MarkerFaceAlpha = 0.2;
[llA, ~] = boundedline(AV(:,1),AV(:,2), AV(:,3),'alpha');hold on
llA.LineWidth = 2; llA.Color = [0.1,0.5,0.8];

% sC = scatter(widthCCW, shearCCW ,15,[1,0.1,0.1],'filled');hold on
% sC.MarkerFaceAlpha = 0.2;
[llC, ~] = boundedline(AV(:,1),AV(:,4), AV(:,5), '-r');hold on
llC.LineWidth = 2; llC.Color = [1,0.1,0.1]; 
set(gca,'Fontsize',18);
% boundedline(AV(:,1),AV(:,4)-90,AV(:,5), 'alpha');
% errorbar(AV(:,1),AV(:,2),AV(:,3));
% ylabel('$ \Delta v = v_{y,\ (x=\frac{L}{2})}  - v_{y,\ (x=-\frac{L}{2})}\ (\mu m/h)$','Interpreter','latex','FontSize',28);
ylabel('$ \Delta v\ (\mu m/h)$','Interpreter','latex','FontSize',32);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',28);
axis([0 1040 -8 14]);
hold off
% errorbar(AV(:,1),AV(:,2),AV(:,3));

figure(145);%subplot(2,1,2);
cla;
% scatter(widthAll,AngAll,15,[.8 .8 .8],'filled');hold on
[llB, ~] = boundedline(AV(:,1),AV(:,6),AV(:,7), 'alpha');
llB.LineWidth = 2; llB.Color = [0.1,0.5,0.8]; set(gca,'Fontsize',18);
ylabel('$| \Delta v |\ (\mu m/h)$','Interpreter','latex','FontSize',24);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
axis([0 1040 0 14]);
hold off

%%  SHEAR VS. TILT
dBins = 0.57;% 0.92;%4.8;% Bin resolution
dd = 1*dBins;% range

% Exclude last experiments without stripes
lst = 500; % makes a limit by width #500=200micron
nV_OP_mAng = V_OP_mAng(1:lst,:);
% after transition to turbulence 250-1000um
nV_OP_mAng = V_OP_mAng(700:1300,:); 
nV_OP_mAng = nV_OP_mAng(nV_OP_mAng(:,6)>60, :);% remove cases with angles <60

Ang = nV_OP_mAng(:,6);
shear = nV_OP_mAng(:,2)-nV_OP_mAng(:,3);
shear = px_sizePIV*shear;

totalBins = floor((max(shear)- min(shear))/dBins);
[N, edges] = histcounts(shear,totalBins);
edgess = [mean([edges(1:end-1); edges(2:end)]), edges(end)] ;
N = [N 0];
[Peak,dv] = findpeaks(N,edgess);

% figure(333); plot(edgess,N,'-'); hold on
% plot(dv,Peak,'o'); hold off
% histogram(shear);histogram(Ang);

AV = zeros(length(dv),4);
AV(:,1) = dv;

for i=1:length(dv)
    ww = dv(i);
    ddd =  0.31*abs(ww); 
    AV(i,2) = mean(Ang(shear>=ww-ddd & shear<=ww+ddd));  
    AV(i,3) = std(Ang(shear>=ww-ddd & shear<=ww+ddd));%./sum(width>=ww-dd & width<=ww+dd)^.5;
    AV(i,4) = sum(shear>=ww-dBins & shear<=ww+dBins);
end

figure(141);
% hold on
scatter(shear,Ang,15,[.6 .6 .8],'filled');set(gca,'Fontsize',18);hold on
[llD, hD] = boundedline(AV(:,1),AV(:,2),AV(:,3), 'alpha');
llD.LineWidth = 2; llD.Color = [.3 .3 .8]; hD.FaceColor = [.3 .3 .8];
% ylabel('$ Central\ angle \langle \theta (x=0) \rangle _{y} (deg)$','Interpreter','latex','FontSize',24);
ylabel('$ Central\ angle\ (deg)$','Interpreter','latex','FontSize',28);
xlabel('$ \Delta v_x\ (\mu m/hr) $','Interpreter','latex','FontSize',28);
axis([-16 30 76 106]);
% errorbar(AV(:,1),AV(:,2),AV(:,3));
axis square tight; hold off
set(gcf,'Color',[1 1 1]);
%% save profiles to txt
sfolder = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\ang_vs_shear\';
mkdir(sfolder)
sname_raw = ['ang_vs_shear_','.txt'];
dlmwrite([sfolder,sname_raw],[shear,Ang],'delimiter','\t','precision',3);
sname = ['av_ang_vs_shear_','.txt'];
dlmwrite([sfolder,sname],AV(:,1:3),'delimiter','\t','precision',3);
figure(141);
saveas(gcf,[[sfolder,sname],'.png'])
%% LEFT_RIGHT SHEAR DISTRIBUTION
% Take the range above 200 micron
first = 600; % 700=250micron
tV_OP_mAng = V_OP_mAng(first:1322,:);
tV_OP_mAng = tV_OP_mAng(tV_OP_mAng(:,6)>60, :);

twidth = tV_OP_mAng(:,1);
tshear = tV_OP_mAng(:,2) - tV_OP_mAng(:,3);
% EXCLUDE LOW SHEAR VALUES LESS THAN |dv|
dv = 0.3;
tshear = tshear(tshear<-dv | tshear>dv);
tshearR = tshear(tshear>.5);
tshearL = tshear(tshear<-.5);

% Take the range below 200 micron
lst = 400; % makes a limit by width #500=200micron
nV_OP_mAng = V_OP_mAng(1:lst,:);
nV_OP_mAng = nV_OP_mAng(nV_OP_mAng(:,6)>60, :);% remove cases with angles <60
nwidth = nV_OP_mAng(:,1);
nshear = nV_OP_mAng(:,2) - nV_OP_mAng(:,3);
nshear = nshear(nshear<-dv | nshear>dv);

nshearR = nshear(nshear>0);
nshearL = nshear(nshear<0);

figure(145);
tedges = -20:1.7:26;
h1 = histogram(tshear,tedges,'Normalization','PDF'); hold on
h1.EdgeAlpha=0.1;h1.FaceColor=[1,0.1,0.1];
nedges = -16:1.9:30;
h2 = histogram(nshear,nedges,'Normalization','PDF'); hold on
h2.EdgeAlpha=0.1;h2.FaceColor=[0.1,0.5,0.8];
% histogram(tshearR); hold on
% histogram(tshearL); hold on
hold off
axis([-15 29 0 .09]);set(gca,'Fontsize',18);
ylabel('$PDF$','Interpreter','latex','FontSize',24); 
xlabel('$\Delta v( \frac {\mu m}{h} )$','Interpreter','latex','FontSize',24);
L1 = legend('width > 200 \mum', 'width < 200 \mum'); %L1.FontSize = 20;
l=axis;
T1 = text(l(1)+.5,l(4)-.01,['> 200 | ' num2str(mean(tshear),2) newline '< 200 | ' num2str(mean(nshear),2)]); 
T1.FontSize = 18;
l=axis;
%% HISTOGRAMS OF THE vR and vL BEFORE AND AFTER THE TURBULENCE ONSET
figure(146);subplot(2,1,2);
tedgesR = -5:1:18;
h1 = histogram(tV_OP_mAng(:,2),50,'Normalization','PDF'); hold on
h1.EdgeAlpha=0.1;h1.FaceColor=[1,0.1,0.1];
tedgesL = -10:1:10;
h2 = histogram(tV_OP_mAng(:,3),50,'Normalization','PDF'); hold on
h2.EdgeAlpha=0.1;h2.FaceColor=[0.1,0.5,0.8];
hold off
axis([-15 15 0 .18]);set(gca,'Fontsize',18);
ylabel('PDF','FontSize',24); xlabel('v_y (\mum/hr)','FontSize',24);
L1 = legend('x=L/2', 'x=-L/2');L1.FontSize = 20;
title('width > 200 \mum')

figure(146);subplot(2,1,1);
nedgesR = -8:1.1:21;
h1 = histogram(nV_OP_mAng(:,2),50,'Normalization','PDF'); hold on
h1.EdgeAlpha=0.1;h1.FaceColor=[1,0.1,0.1];
nedgesL = -13:1:12;
h2 = histogram(nV_OP_mAng(:,3),50,'Normalization','PDF'); hold on
h2.EdgeAlpha=0.1;h2.FaceColor=[0.1,0.5,0.8];
hold off
axis([-15 15 0 .18]);set(gca,'Fontsize',18);
ylabel('PDF','FontSize',24); xlabel('','FontSize',1);
L1 = legend('x=L/2', 'x=-L/2');L1.FontSize = 20;
title('width < 200 \mum')
%% VALIDATION OF ANIPARALLEL SHEAR FLOWS 
% clear all
% load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new3.mat');
% load("C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_Ek_Ev_unfiltered.mat");
% 
% Ek = wEW(:,2);
% Ev = wEW(:,3);
% % load filtered velocities from 'correlate_OP_PIV_Defect.m'
% for i=1:size(dirOP,1)
%     vR(i,1) = EXP{i,6}(end,2);
%     vL(i,1) = EXP{i,6}(1,2);
% end
% 
% V_OP_mAng = [width, vR, vL, Ek, Ev, Ang_mid];
% [V_OP_mAng, indByWidth] = sortrows(V_OP_mAng,1);
%% VALIDATION OF ANIPARALLEL SHEAR FLOWS
% RATIO OF CCW AND CW FLOWS
dBins = 9%4.5;%4.8;% Bin resolution
dd = 1*dBins;% range
totalBins = floor((max(width)- min(width))/dBins);
[N, edges] = histcounts(width,totalBins);
edgess = [edges(1)-dBins, mean([edges(1:end-1); edges(2:end)]), edges(end)] ;
N = [0 N 0];
[Peak,widthOfPeak] = findpeaks(N,edgess);

% figure(333); plot(edgess,N,'-'); hold on
% plot(widthOfPeak,Peak,'o'); hold off

% Exclude last experiments without stripes
lst = 1322; % makes a limit by width
nV_OP_mAng = V_OP_mAng(1:lst,:);

width = nV_OP_mAng(:,1);
shearProduct = nV_OP_mAng(:,2).*nV_OP_mAng(:,3);
shearDir = nV_OP_mAng(:,2)- nV_OP_mAng(:,3);

AV = zeros(length(widthOfPeak),9);
AV(:,1) = widthOfPeak;

for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    ddd = .06*ww;
    AV(i,2) = mean(shearProduct(width>=ww-ddd & width<=ww+ddd));  
%     AV(i,3) = std(shearCW(widthCW>=ww-ddd & widthCW<=ww+ddd));
    AV(i,3) = std(shearProduct(width>=ww-ddd & width<=ww+ddd))...
        ./sum(width>=ww-ddd & width<=ww+ddd)^.5;
    temp = shearProduct(width>=ww-ddd & width<=ww+ddd);
    AV(i,4) = sum(temp<0) / (sum(temp<0)+sum(temp>0));
    AV(i,5) = (sum(temp<0)*sum(temp>0))^.5  / (sum(temp<0)+sum(temp>0))^1.5;
    
    temp1 = shearProduct(width>=ww-ddd & width<=ww+ddd);
    temp2 = shearDir(width>=ww-ddd & width<=ww+ddd);
    temp3  = temp2(temp1<0);
    AV(i,6) = sum(temp3<0)/(sum(temp3<0)+sum(temp3>0));
    AV(i,7) = (sum(temp3<0)*sum(temp3>0))^.5 /(sum(temp3<0)+sum(temp3>0))^1.5;
    AV(i,8) = sum(temp3>0)/(sum(temp3<0)+sum(temp3>0));
    AV(i,9) = (sum(temp3<0)*sum(temp3>0))^.5 /(sum(temp3<0)+sum(temp3>0))^1.5;
end
figure(148);
scatter(width,shearProduct,15,[.8 .8 .8],'filled');set(gca,'Fontsize',18);hold on
[llD, hD] = boundedline(AV(:,1),AV(:,2),AV(:,3), 'alpha');
llD.LineWidth = 2; llD.Color = [1,0.1,0.1]; hD.FaceColor = [1,0.1,0.1];
ylabel('$\langle v_{y}(x=\frac{L}{2}) \rangle _{y} \cdot \langle v_{y}(x=-\frac{L}{2})\rangle _{y}$','Interpreter','latex','FontSize',28);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
axis([0 1040 -100 30]);
% errorbar(AV(:,1),AV(:,2),AV(:,3));
hold off

figure(149);
[llD, hD] = boundedline(AV(:,1),AV(:,4),AV(:,5), 'alpha');set(gca,'Fontsize',18);hold on
llD.LineWidth = 2; llD.Color = [1,0.1,0.1]; hD.FaceColor = [1,0.1,0.1];
ylabel('$Shear\ flow\ probability $','Interpreter','latex','FontSize',28);
xlabel('$ Width\ (\mu m) $','Interpreter','latex','FontSize',24);
axis([0 1040 0 1.05]);
% errorbar(AV(:,1),AV(:,2),AV(:,3));
hold off

figure(150);
[llE, hE] = boundedline(AV(:,1),AV(:,6),AV(:,7), 'alpha');set(gca,'Fontsize',18); hold on
llE.LineWidth = 2; llE.Color = [1,0.1,0.1]; hE.FaceColor = [1,0.1,0.1];
[llF, hF] = boundedline(AV(:,1),AV(:,8), AV(:,9), 'alpha');
llF.LineWidth = 2; llF.Color = [.2,0.1,0.8]; hF.FaceColor = [.2,0.1,0.8];
ylabel('$ Probability $','Interpreter','latex','FontSize',24);
xlabel('$ Width\ (\mu m) $','Interpreter','latex','FontSize',24);

% errorbar(AV(:,1),AV(:,6),AV(:,7));
% errorbar(AV(:,1),AV(:,8), AV(:,9));
hold off
axis tight square
xlim([250,1040])
set(gcf,'Color',[1 1 1]);
%% save probability plot and data
sfolder = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\v_profile\';
mkdir(sfolder)
sname = ['chiral_flow_probability','.txt'];
dlmwrite([sfolder,sname],[AV(:,1),AV(:,6),AV(:,7),AV(:,8), AV(:,9)],'delimiter','\t','precision',3)


%% DISTRIBUTION of VORTEX POSITIONS
clearvars cwY ccwY cwX ccwX ccw_vortPOS1 ccw_vortPOS2
range = 1102:1322;
ccw_vortPOS1 = ccw_vortPOS(indByWidth);
ccw_vortPOS2 = ccw_vortPOS1(range,1);
cw_vortPOS1 = cw_vortPOS(indByWidth);
cw_vortPOS2 = cw_vortPOS1(range,1);
for i=1:size(ccw_vortPOS2,1)
    ccwXY = cell2mat(ccw_vortPOS2{i});
    ccwX{i,1} = ccwXY(:,1); 
    ccwY{i,1} = ccwXY(:,2); 
    cwXY = cell2mat(cw_vortPOS2{i});
    cwX{i,1} = cwXY(:,1);
    cwY{i,1} = cwXY(:,2); 
%     plot(vNet{i}); hold on
end
figure(900);
histogram(cell2mat(ccwX)); hold on
histogram(cell2mat(cwX)); hold off
figure(901);
histogram(cell2mat(ccwY)); hold on
histogram(cell2mat(cwY)); hold off
%% NET FLOW  indByWidth
for i=1:size(vNet,1)
    vy_net(i,1) = mean(vNet{i});
%     plot(vNet{i}); hold on
end
vy_net_sort = vy_net(indByWidth);
%%
figure(500);
scatter(V_OP_mAng(:,1),vy_net_sort,15,[.9 .2 .3],'filled');
figure(501);
range = 700:1300;
scatter(V_OP_mAng(range,2)-V_OP_mAng(range,3),vy_net_sort(range),15,[.9 .2 .3],'filled');
