%%
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new3.mat')
V_OP_mAng = [width, zeros(size(width)), OP_mid, Ang_mid, Ang_mid_std, ccw_vortNumAv, vEnergy, cw_vortNumAv];
[V_OP_mAng, indByWidth] = sortrows(V_OP_mAng,1);

% figure(6);hold on;scatter(V_OP_mAng(:,1),V_OP_mAng(:,9)-V_OP_mAng(:,10),15,[.9 .2 .3],'filled');
%% Order Parameter
dBins = 5.5;% Bin resolution
dd = 1*dBins;% range
totalBins = floor((max(width)- min(width))/dBins);
[N, edges] = histcounts(width,totalBins);
edgess = [edges(1), mean([edges(1:end-1); edges(2:end)]), edges(end)] ;
N = [0 N 0];
[Peak,widthOfPeak] = findpeaks(N,edgess);
% plot(edgess,N,'-'); hold on
% plot(widthOfPeak,Peak,'o'); hold off

AV = zeros(length(widthOfPeak),4);
AV(:,1) = widthOfPeak;
% shear = abs(shear);
% width = V_OP_mAng(:,1);
for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    AV(i,2) = mean(OP_mid(width>=ww-dd & width<=ww+dd));  
    AV(i,3) = std(OP_mid(width>=ww-dd & width<=ww+dd));%./sum(width>=ww-dd & width<=ww+dd)^.5;
    AV(i,4) = sum(width>=ww-dBins & width<=ww+dBins);
end

figure(143);
[llC, bC] = boundedline([0 1040],[AV(end,2) AV(end,2)],[AV(end,3) AV(end,3)], 'alpha'); hold on
llC.LineWidth = 2;llC.LineStyle = '--'; llC.Color = [0.6,0.6,0.6];
bC.FaceColor = [0.9,0.9,0.9]; bC.FaceAlpha = 0.5;
scatter(V_OP_mAng(:,1),V_OP_mAng(:,3),15,[.8 .8 .8],'filled');set(gca,'Fontsize',18);hold on
[llB, ~] = boundedline(AV(:,1),AV(:,2),AV(:,3), 'alpha');
llB.LineWidth = 2; llB.Color = [0.1,0.5,0.8];
% ylabel('$Q = \sqrt{\langle cos 2\theta_{x=0}  \rangle ^2 + \langle sin 2\theta_{x=0}  \rangle ^2}$','Interpreter','latex','FontSize',24);
ylabel('$ Central\ order\ parameter $','Interpreter','latex','FontSize',32);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',28);
hold off
axis([0 1040 0 1]);
% errorbar(AV(:,1),AV(:,2),AV(:,3));
hold off

%% Angle (TILT)
dBins = 3.5%4.8;% Bin resolution
dd = 1*dBins;% range
totalBins = floor((max(width)- min(width))/dBins);
[N, edges] = histcounts(width,totalBins);
edgess = [edges(1), mean([edges(1:end-1); edges(2:end)]), edges(end)] ;
N = [0 N 0];
[Peak,widthOfPeak] = findpeaks(N,edgess);

% figure(333); plot(edgess,N,'-'); hold on
% plot(widthOfPeak,Peak,'o'); hold off

width = V_OP_mAng(:,1);
Ang = V_OP_mAng(:,4);
AngSTD = V_OP_mAng(:,5);

overlap = 1.5;
widthR = width(Ang<90+overlap & Ang>60);
AngR = Ang(Ang<90+overlap & Ang>60);
AngR_STD = AngSTD(Ang<90+overlap & Ang>60);

widthL = width(Ang>90-overlap);
AngL = Ang(Ang>90-overlap);
AngL_STD = AngSTD(Ang>90-overlap);

widthAll = width(Ang>60);
AngAll = abs(Ang(Ang>60)-90);
AngAll_STD = AngSTD(Ang>60);

AV = zeros(length(widthOfPeak),8);
AV(:,1) = widthOfPeak;

for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    ddd = .1*ww;
    AV(i,2) = mean(AngR(widthR>=ww-ddd & widthR<=ww+ddd));  
%     AV(i,3) = std(AngR(widthR>=ww-ddd & widthR<=ww+ddd));
    AV(i,3) = mean(AngR_STD(widthR>=ww-ddd & widthR<=ww+ddd)); 
    AV(i,4) = mean(AngL(widthL>=ww-ddd & widthL<=ww+ddd));  
%     AV(i,5) = std(AngL(widthL>=ww-ddd & widthL<=ww+ddd));
    AV(i,5) = mean(AngL_STD(widthL>=ww-ddd & widthL<=ww+ddd)); 
    AV(i,6) = mean(AngAll(widthAll>=ww-ddd & widthAll<=ww+ddd));  
%     AV(i,7) = std(AngAll(widthAll>=ww-ddd & widthAll<=ww+ddd)) ...
%         /sum(widthAll>=ww-dd & widthAll<=ww+dd).^.5;
    AV(i,7) = mean(AngAll_STD(widthAll>=ww-ddd & widthAll<=ww+ddd)); 
end

figure(144);%subplot(2,1,1);
cla;
% scatter(V_OP_mAng(:,1),V_OP_mAng(:,4),15,[.8 .8 .8],'filled');set(gca,'Fontsize',18);hold on
[llA, hA] = boundedline(AV(:,1),[AV(:,2), AV(:,4)],[AV(:,3), AV(:,5)], 'alpha');hold on
llA(1).LineWidth = 2; llA(1).Color = [0.1,0.5,0.8];
llA(2).LineWidth = 2; llA(2).Color = [1,0.1,0.1]; set(gca,'Fontsize',18);
% boundedline(AV(:,1),AV(:,4)-90,AV(:,5), 'alpha');
ylabel('$ Central\ angle,\  \langle \theta _{(x=0)} \rangle _{y}$','Interpreter','latex','FontSize',32);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',28);
axis([0 1040 83 97]);hold off
% errorbar(AV(:,1),AV(:,2),AV(:,3));

figure(145);%subplot(2,1,2);
cla;
% scatter(widthAll,AngAll,15,[.8 .8 .8],'filled');hold on
[llB, hB] = boundedline(AV(:,1),AV(:,6),AV(:,7), 'alpha');
llB.LineWidth = 2; llB.Color = [0.1,0.5,0.8]; set(gca,'Fontsize',18);
ylabel('$|\langle \theta _{(x=0 \pm \delta)} \rangle _{y}| - \frac{\pi}{2}$','Interpreter','latex','FontSize',28);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
axis([0 400 0 6.5]);
hold off
%%
figure(555);
plot(AV(:,1),AV(:,7)); hold on
scatter(V_OP_mAng(:,1),V_OP_mAng(:,5),15,[.8 .8 .8],'filled');
scatter(widthAll,AngSTD);
hold off

axis([0 1540 0 2]);

%%  DEFECT DENSITY VORTEX DENSITY vs WiDTH
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new3.mat')
vortDensityAv = (ccw_vortNumAv+cw_vortNumAv)./(1500*width);
V_OP_mAng = [width, def_DensityAv,vortDensityAv, ccw_vortNumAv, cw_vortNumAv];
[V_OP_mAng, indByWidth] = sortrows(V_OP_mAng,1);
V_OP_mAng1 = V_OP_mAng(V_OP_mAng(:,3)<1.5e-4,:);
%%
scatter(V_OP_mAng(:,1),V_OP_mAng(:,2),15,[.1 .1 .8],'filled');set(gca,'Fontsize',18);hold on
scatter(V_OP_mAng(:,1),V_OP_mAng(:,3),15,[.1 .8 .1],'filled');set(gca,'Fontsize',18);hold on

scatter(V_OP_mAng(:,1),V_OP_mAng(:,4)./(1500*V_OP_mAng(:,1)),15,[.8 .1 .1],'filled');set(gca,'Fontsize',18);hold on
scatter(V_OP_mAng(:,1),V_OP_mAng(:,5)./(1500*V_OP_mAng(:,1)),15,[.1 .8 .1],'filled');set(gca,'Fontsize',18);hold on

%%
% lst = 1322; % makes a limit by width
% V_OP_mAng1 = V_OP_mAng(1:lst,:);

dBins = 4.5;% Bin resolution
dd = 1*dBins;% range
totalBins = floor((max(width)- min(width))/dBins);
[N, edges] = histcounts(V_OP_mAng1(:,1),totalBins);
edgess = [edges(1), mean([edges(1:end-1); edges(2:end)]), edges(end)] ;
N = [30 N 0];
[Peak,widthOfPeak] = findpeaks(N,edgess);
% plot(edgess,N,'-'); hold on
% plot(widthOfPeak,Peak,'o'); hold off

AV = zeros(length(widthOfPeak),5);
AV(:,1) = widthOfPeak;

width1 = V_OP_mAng1(:,1);
def_DensityAv1 = V_OP_mAng1(:,2);
vortDensityAv1 = V_OP_mAng1(:,3);

for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    ddd = 0.19*ww;
    AV(i,2) = mean(def_DensityAv1(width1>=ww-ddd & width1<=ww+ddd));  
    AV(i,3) = std(def_DensityAv1(width1>=ww-ddd & width1<=ww+ddd));%./sum(width>=ww-dd & width<=ww+dd)^.5;
    AV(i,4) = mean(vortDensityAv1(width1>=ww-ddd & width1<=ww+ddd));  
    AV(i,5) = std(vortDensityAv1(width1>=ww-ddd & width1<=ww+ddd));
end
NormDef = mean(def_DensityAv1(width1>=1100));
NormVort = mean(vortDensityAv1(width1>=1100));
figure(143);cla
sc1 = scatter(V_OP_mAng(:,1),V_OP_mAng(:,2)/NormDef,15,[.1 .1 .8],'filled');hold on
sc1.MarkerFaceAlpha = .2;
sc2 = scatter(V_OP_mAng(:,1),V_OP_mAng(:,3)/NormVort,15,[.8 .1 .1],'filled');hold on
sc2.MarkerFaceAlpha = .2;
[llB, hhB] = boundedline(AV(:,1),AV(:,2)/NormDef,AV(:,3)/NormDef, 'alpha');hold on
llB.LineWidth = 2; llB.Color = [.1 .1 .8];hhB.FaceColor = llB.Color;
[llC, hhC] = boundedline(AV(:,1),AV(:,4)/NormVort,AV(:,5)/NormVort, 'alpha');
llC.LineWidth = 2; llC.Color = [.8 .1 .1];hhC.FaceColor = llC.Color;
set(gca,'Fontsize',18);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',28);

hold off
ylabel('$ Normalized\ density$','Interpreter','latex','FontSize',32);
axis([0 1040 0 2.4]);

% ylabel('$ Density\ (\mu m^{-2})$','Interpreter','latex','FontSize',32);
% axis([0 1040 0 1.2e-4]);



%% COMPARE CCW vs CW VOTREX NUMBER 
nV_OP_mAng = V_OP_mAng(700:end,:);
width = nV_OP_mAng(:,1);

dBins = 4.7;%4.8;% Bin resolution
dd = 1*dBins;% range
totalBins = floor((max(width)- min(width))/dBins);
[N, edges] = histcounts(width,totalBins);
edgess = [edges(1)-dBins, mean([edges(1:end-1); edges(2:end)]), edges(end)] ;
N = [0 N 0];
[Peak,widthOfPeak] = findpeaks(N,edgess);

% figure(333); plot(edgess,N,'-'); hold on
% plot(widthOfPeak,Peak,'o'); hold off

% Exclude last experiments without stripes
% lst = 1322; % makes a limit by width
% nV_OP_mAng = V_OP_mAng(1:lst,:);

vortNum = (nV_OP_mAng(:,6)- nV_OP_mAng(:,8))./(nV_OP_mAng(:,6)+ nV_OP_mAng(:,8));

AV = zeros(length(widthOfPeak),8);
AV(:,1) = widthOfPeak;

for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    ddd = .06*ww;
    AV(i,2) = mean(vortNum(width>=ww-ddd & width<=ww+ddd));  
%     AV(i,3) = std(shearCW(widthCW>=ww-ddd & widthCW<=ww+ddd));
    AV(i,3) = std(vortNum(width>=ww-ddd & width<=ww+ddd))...
        ./sum(width>=ww-ddd & width<=ww+ddd)^.5;
end
figure(201);
scatter(width,vortNum,15,[.8 .8 .8],'filled');set(gca,'Fontsize',18);hold on
[llD, hD] = boundedline(AV(:,1),AV(:,2),AV(:,3), 'alpha');
llD.LineWidth = 2; llD.Color = [1,0.1,0.1]; hD.FaceColor = [1,0.1,0.1];
ylabel('$(\langle CW \rangle -\langle CCW \rangle) / (\langle CW \rangle +\langle CCW \rangle)$','Interpreter','latex','FontSize',24);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
axis([0 1040 -.012 .012]);
% errorbar(AV(:,1),AV(:,2),AV(:,3));
hold off