%% IMPORT AND SORT the EXP

load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new2.mat')
% load('C:\Users\vici\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new1.mat')
[~,idx] = sort(cell2mat(EXP(:,1)));
sortEXP = EXP(idx,:);
sortAng = Ang_mid(idx,:);
sortOP = OP_mid(idx,:);
% select stripes only
sortEXP = sortEXP(1:1322,:); 
sortAng = sortAng(1:1322,:); 
sortOP = sortOP(1:1322,:); 
%% PLOT ALL VELOCITY PROFILES 
%[1)width, 2)vy, 3)vx, 4)num of flips, 5){x,u_profile,u_std}, 6){x,v_profile,v_std}]
c1=jet(size(sortEXP,1));
edgeVy_abs = zeros(size(sortEXP,1),1);
edgeVy = zeros(size(sortEXP,1),1);
for i=1:size(sortEXP,1)
%     figure(100);set(gca,'Fontsize',18); xlabel('width (\mum)'); ylabel('v_x (\mum/hr)');
%     plot(sortEXP{i,5}(:,1),sortEXP{i,5}(:,2),'color',[c1(i,:),.5],'LineWidth',2); hold on
%     figure(101);set(gca,'Fontsize',18); xlabel('width (\mum)'); ylabel('v_y (\mum/hr)');
%     plot(sortEXP{i,6}(:,1),sortEXP{i,6}(:,2),'color',[c1(i,:),.5],'LineWidth',2); hold on
    
    edgeVy_abs(i,1) = 0.5*(abs(sortEXP{i,6}(1,2))+abs(sortEXP{i,6}(end,2)));
%     edgeVy(i,1) = 0.5*(sortEXP{i,6}(1,2)+sortEXP{i,6}(end,2));
    edgeVy(i,1) = sortEXP{i,6}(1,2)-sortEXP{i,6}(end,2);
end

Xdata = cell2mat(sortEXP(:,1));
% Ydata = edgeVy_abs;
Ydata = edgeVy;

figure(103)
scatter(Xdata,Ydata,15,'filled');
set(gca,'Fontsize',18); xlabel('width (\mum)'); ylabel('v_y (x=?L/2) (\mum/hr)');%axis([0 1040 -inf inf]);%
axis ([0,1050,-inf,inf]);%tight% 
% q = polyfit(Xdata,Ydata,4);
% rq = refcurve(q); rq.Color = 'c';rq.LineWidth = 2;
hold on
%% BINNED AVERAGE OF VELOCITY (RUN TWICE <=0 and >=0)
Xdata = cell2mat(sortEXP(:,1));
% Ydata = edgeVy_abs;
Ydata = edgeVy;
% Run TWICE Ydata<=0 and Ydata>=0
Xdata = Xdata(Ydata>=0);
Ydata = Ydata(Ydata>=0);
% step 1 - get the data distribution
b1 = 0:5:90 ; b2 = 90:8:200 ; b3 = 200:10:500 ;b4 = 500:25:Xdata(end)+50;
bins = cat(1, b1',b2', b3',b4') ;
[N, edges] = histcounts(Xdata,bins);
edgess = mean([edges(1:end-1), edges(2:end)],2);
[Peak,PeakWidth] = findpeaks(N,edgess);
% plot(edgess,N,'-'); hold on
% plot(PeakWidth,Peak,'o'); hold off
% step 2 - VELOCITY
for i=1:length(PeakWidth)
    ww  = PeakWidth(i); dd=5;
    if ww>b2(1) && ww<b3(1)
        dd=8;
    elseif ww>b2(1) && ww<b3(1)
        dd=8;
    elseif ww>b3(1) && ww<b4(1)
        dd=10;
    elseif ww>b4(1)
        dd=25;  
    end
    AV(i,2) = mean(Ydata(Xdata>=ww-dd & Xdata<=ww+dd));  
    AV(i,3) = std(Ydata(Xdata>=ww-dd & Xdata<=ww+dd))./...
        sum(Ydata(Xdata>=ww-dd & Xdata<=ww+dd))^.5;
end

figure(102)
scatter(Xdata,Ydata,15,'filled');hold on
set(gca,'Fontsize',18); xlabel('width (\mum)'); ylabel('v_y(x=?L/2) (\mum/hr)');%axis([0 1040 -inf inf]);%
axis ([0,1050,-inf,inf]);%tight;
e1 = errorbar(PeakWidth,AV(:,2),AV(:,3));
e1.LineWidth = 2;
%%
scatter(cell2mat(sortEXP(:,1)),sortAng-90,15,'filled');
figure;scatter(cell2mat(sortEXP(:,1)),sortOP,15,'filled');
figure;scatter(edgeVy,sortAng-90,15,'filled');

%%
AV = zeros(length(PeakWidth),4);
AV(:,1) = PeakWidth;
for i=1:length(PeakWidth)
    ww = PeakWidth(i);
    AV(i,2) = mean(Ang_mid(shear>=ww-dd & shear<=ww+dd));  
    AV(i,3) = std(Ang_mid(shear>=ww-dd & shear<=ww+dd))./...
        sum(shear>=ww-dd & shear<=ww+dd)^.5;
    AV(i,4) = sum(shear>=ww-dBins & shear<=ww+dBins);
end
figure(24); hold on
% scatter(shear,OP_mid,10,'fill'); hold on
% plot(widthOP,OP,'.');hold on
errorbar(AV(:,1),AV(:,2),AV(:,3));hold off

%% IMPORT AND SORT 
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new2.mat')
% load('C:\Users\vici\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new1.mat')
for i=1:size(dirOP,1)
%     plot(EXP{i,6}(:,1),EXP{i,6}(:,2)); hold on
    shear(i,1) = EXP{i,6}(1,2)-EXP{i,6}(end,2);
end
cd=150;% shift jet colors
c1=jet(size(OP_mid,1)+cd);
c=c1(1:end-cd,:);
% shear = cell2mat(EXP(:,3));
V_OP_mAng = [width, shear,OP_mid, Ang_mid, vortNumAv, vEnergy];
[V_OP_mAng, indByWidth] = sortrows(V_OP_mAng,1);

%% SHEAR FLOW vs WIDTH
% widthS = V_OP_mAng(V_OP_mAng(V_OP_mAng(:,2)<0),1);
range = 1:size(V_OP_mAng,1);%w0+200;
Xdata = V_OP_mAng((V_OP_mAng(:,2)<0),1);
Ydata = V_OP_mAng((V_OP_mAng(:,2)<0),2);

scatter(Xdata,Ydata,15,'filled');hold on
set(gca,'Fontsize',18); xlabel('width (\mum)'); ylabel('shear velocity (\mum/hr)');%axis([0 1040 -inf inf]);%
axis tight%([0,10,.02,.08]);
q = polyfit(Xdata,Ydata,4);
rq = refcurve(q); rq.Color = 'c';rq.LineWidth = 2;
%% DEFECT DENSITY vs SHEAR FLOW
figure
% V_OP_mAng = [width, shear,OP_mid, Ang_mid, vortNumAv, vEnergy];
w0=1;
range = w0:200;%size(V_OP_mAng,1);%w0+200;
% scatter(abs(V_OP_mAng(range,2)),V_OP_mAng(range,5)./V_OP_mAng(range,1),15,c(range,:),'filled');
scatter(abs(V_OP_mAng(range,2)),V_OP_mAng(range,5)./V_OP_mAng(range,1),15,'filled');
set(gca,'Fontsize',18); xlabel('v_y (\mum/hr)'); ylabel('Vortex Density');%axis([0 1040 -inf inf]);%
axis([0,10,.02,.08]);hold on

ll = lsline; ll.Color = 'r'; ll.LineWidth = 2;
q = polyfit(abs(V_OP_mAng(range,2)),V_OP_mAng(range,5)./V_OP_mAng(range,1),3);
rq = refcurve(q); rq.Color = 'c';rq.LineWidth = 2;
legend(['width: ',num2str(V_OP_mAng(range(1),1)),'-',num2str(V_OP_mAng(range(end),1))],...
    'line fit', 'polyfit')
