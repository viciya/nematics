%% LOAD FILES
PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
% PC_path = 'D:\GD\';                         % RSIP notebook

addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

[dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);
%% SELECT WIDTH AND PARAMETERS
clearvars -except dirOP  dirPIV  Sorted_Orient_width  indX PC_path pathOP pathPIV NET

i = 1;
Sw = 400; % selectd width
dw = .05*Sw; % define delta
box = 50;
s_box = floor(sqrt(box^2/2));
px_size = .74;
pix2mic = 3 * px_size;

ff = 3;
filt = fspecial('gaussian',ff,ff);

Edge = 70;

Ltot=0;
Rtot=0;
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<1100,1);
% Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>280 & Sorted_Orient_width(:,2)<1200,1);
% Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);
% Range = Sorted_Orient_width(Sorted_Orient_width(:,2)<Sw,1);
% Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw & Sorted_Orient_width(:,2)<1600,1);
% Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>0,1);

% LOAD FILES FROM DIRECTORY
% ----------------------------------------------------
% pathList = dir(['D:\GD\DATA2\save\s2\PIV_DATs\','\*.mat']);
% Range = 1:size(pathList,1);
% indX = Range';
% indX(:,2) = Range';
% dirPIV = pathList;
% ----------------------------------------------------
%
stripe = struct([]);

% %%
v_cent = [];
u_cent = [];
v_theta = [];
av_v = [];
av_u = [];
av_shear = 0;
std_shear = 0;
count = 0;
% figure(Sw);
for i = 1:numel(Range)
    
    filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
    filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];
    
    load(filepathPIV);
    info = imfinfo(filepathOP);
    disp(['File ',num2str(i), ' from ',num2str(numel(Range)),'--', num2str(size(v,1))]);
    
    dx = px_size*(x{1}(1,2)-x{1}(1,1));
    v_temp = [];
    u_temp = [];
    for k = 1:size(v,1)
        
        %         iv_net(k,1) = mean2(v{k});
        %         ishear(k,1) = (mean(v{k}(:,end))-...
        %             mean(v{k}(:,1)));%- iv_net(k,1);
        % 90 deg rotation
%         iv_net(k,1) = mean2(u{k});
%         ishear(k,1) = (mean(u{k}(end,:))-...
%             mean(v{k}(1,:)));%- iv_net(k,1);
        
        uf = imfilter(u{k}, filt);
        vf = imfilter(v{k}, filt);
        
%         vrms(k,i) = mean2((uf.^2 + vf.^2).^.5);
%         
%         [u_x,u_y] = gradient(px_size * uf,dx);
%         [v_x,v_y] = gradient(px_size * vf,dx);
%         enstrophy(k,i) = mean2(0.5*(v_x - u_y).^2);
%         energy(k,i) = mean2(0.5*(uf.^2 + vf.^2));
        
        v_temp(k,:) = vf(:,round(size(vf,2)/2));%-mean2(vf);%reshape(vf(:,1:3),[],1);%
        u_temp(k,:) = uf(:,round(size(uf,2)/2));%-mean2(uf);%reshape(vf(:,end-2:end),[],1);%
        av_u_temp{k,1} = uf(:);
        av_v_temp{k,1} = vf(:);

        
    end
    v_cent = [v_cent; v_temp(:)- mean(cell2mat(av_v_temp))];
    u_cent = [u_cent; u_temp(:)- mean(cell2mat(av_u_temp))];
%     lim = .1;
%     vv = v_cent(abs(v_cent)>lim & abs(u_cent)>lim);
%     uu = u_cent(abs(v_cent)>lim & abs(u_cent)>lim);
%     v_theta = [v_theta; atan2(vv,uu)];
%     %     figure(23)
%     %     plot(nanmean(energy./enstrophy,2)); hold on
%     
%     stripe(i).shear = ishear;
%     stripe(i).width = pix2mic * info(1).Width;
%     stripe(i).av_shear = mean(ishear);
%     stripe(i).std_shear = std(ishear);
%     stripe(i).av_v_net = mean(iv_net);
%     stripe(i).CW = mean(ishear)>0;
end
%%
v_cent_mic = px_size * v_cent;
u_cent_mic = px_size * u_cent;

f = figure(26);
% polarhistogram(v_theta, 30);hold on
th = 0;
v_th = v_cent_mic(abs(v_cent_mic)>th & abs(u_cent_mic)>th);
u_th = u_cent_mic(abs(v_cent_mic)>th & abs(u_cent_mic)>th);
[theta,tbins] = histcounts(atan2(v_th,u_th),35,'Normalization' ,'pdf');
pol = polarplot(theta,'r','LineWidth',2); hold on
title({['$ \langle v_x \rangle',num2str(mean(u_cent)),'|',num2str(std(u_cent)),' (\ \mu m/hr)$'],...
    ['$ \langle v_y \rangle',num2str(mean(v_cent)),'|',num2str(std(v_cent)),' (\ \mu m/hr)$']},'Interpreter','latex','FontSize',16)
% legend({['$ \langle v_x \rangle',num2str(11),' (\ \mu m)$']},'Interpreter','latex','FontSize',16)
% hold off
%%
figure(Sw); 
v_cent_mic = px_size * v_cent;
u_cent_mic = px_size * u_cent;
% histogram(v_cent_mic,-60:1:60);hold on
% histogram(u_cent_mic,-60:1:60);hold off
bins = -60:3:60;
xbin = mean([bins(1:end-1);bins(2:end)]);
[v_counts,~] = histcounts(v_cent_mic,bins,'Normalization' ,'pdf');
[u_counts,~] = histcounts(u_cent_mic,bins,'Normalization' ,'pdf');
plot(xbin, u_counts,'LineWidth',3,'Color',[.9,.1,.1, .8]); hold on
plot(xbin, v_counts, 'LineWidth',3,'Color',[.1,.1, .9, .8]); hold on
xlabel('$ Velocity \ (\mu m/hr)$','Interpreter','latex')
ylabel('$ PDF $','Interpreter','latex')
title(['$ Velocity \ distribution \ (', num2str(Sw), '\ \mu m)$'],'Interpreter','latex')
legend({'$ v_x$', '$ v_y $'},'Interpreter','latex','FontSize',14) 
xlim([-60,60])
axis square
hold off

%%
figure(24);cla
ratio = px_size^2 *nanmean(energy./enstrophy, 2);
ratio_std = px_size * std(energy./enstrophy, [], 2);
[la, ~] = boundedline((1:length(ratio))/4, ratio, ratio_std);hold on
la.LineWidth = 2;
la.Color = [.0, .4, .75];
xlabel('$ Time (hr) $','Interpreter','latex')
ylabel('$ E_k/E_w \ (\mu m^2)$','Interpreter','latex')
axis square
ylim([400,600])
hold off
set(gca,'Fontsize',18);

figure(23);
vrms(vrms==0) = NaN;
vrms_mean = px_size * nanmean(vrms,2);
vrms_std = px_size * std(vrms, [], 2);
[lb, ~] = boundedline((1:length(vrms_mean))/4,vrms_mean,vrms_std);hold on
lb.LineWidth = 2;
lb.Color = [.0, .4, .75];
xlabel('$ Time (hr) $','Interpreter','latex')
ylabel('$ v_{rms} \ (\mu m / hr)$','Interpreter','latex')
axis square
ylim([10,14])
hold off
%%
ipath ='C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\Supplement\coherency';
files = dir([ipath  '\*.txt']);
coherency = [];
for i=1:size(files,1)
    idata = importdata([files(i).folder,'\',files(i).name]);
    coherency(:,i) = idata(:,2);
%     plot(idata(:,1),idata(:,2)); hold on
end
% coherency = coher
[lb, ~] = boundedline((1:size(coherency,1))/4,mean(coherency,2),std(coherency, [], 2));
lb.LineWidth = 2;
lb.Color = [.75, .4, 0];
xlabel('$ Time (hr) $','Interpreter','latex')
ylabel('$ v_{rms} \ (\mu m / hr)$','Interpreter','latex')
axis square
ylim([0,1])
xlim([0,25])
hold off

%%
A = 450;
B = A+50;
CW = [stripe(:).av_shear]>-10;
figure
boundedline([stripe(:).width],[stripe(:).av_shear],[stripe(:).std_shear]);
figure
sc = scatter([stripe(:).width],[stripe(:).av_v_net],'filled');
sc.MarkerFaceAlpha = .3;
% load('stripe_shear_Large.mat')
% load('stripe_shear_Small.mat')
% load('stripe_shear_All.mat')
av = mean([stripe(A+1).av_shear])
std = mean([stripe(CW).std_shear])
floor(stripe(A).width)
%%
figure( floor(stripe(A).width))
color = rand(1,3);
for i = A:B
    if stripe(i).av_shear>0
        plot([stripe(i).shear],'color',[color,.4],'LineWidth',2);hold on
    end
end

xlim([0,100])
ylim([-20,25])
% axis tight
hold off
%%
i=2;
filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];
load(filepathPIV);
k=58;
(mean2(v{k}(:,end-1:end))-...
    mean2(v{k}(:,1:2)))-mean2(v{k})