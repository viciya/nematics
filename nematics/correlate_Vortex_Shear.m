%%
dirPIV = dir(['C:\Users\victo\Google Drive\DATA\HT1080\PIV_DATs'  '\*.mat']);

px_sizePIV = .74;

OP_mid = zeros(size(dirPIV,1),1);
EXP = cell(size(dirPIV,1),10);

for i=1:1%size(dirPIV,1)
        %-------------PIV---------------------------------------------------
    filepathPIV = [dirPIV(i).folder '\' dirPIV(i).name];
    load(filepathPIV);
%     clear('resultslist'); % If it's really not needed any longer.

    X = px_sizePIV*x{1,1}; Y = px_sizePIV*y{2,1};
    u_profile = zeros(1,size(X,2));
    v_profile = zeros(1,size(X,2));
    u_std = zeros(1,size(X,2));
    v_std = zeros(1,size(X,2));
    
    for k=1:size(x,1)
        % --------------------------PIV import ---------------------------------
        u_profile = u_profile + mean(u{k});
        u_std = u_std + std(u{k});
        v_profile = v_profile + mean(v{k});
        v_std = u_std + std(v{k});
        
    end
    u_profile = (u_profile-mean(u_profile))/k;
    % - NORMALISATION
    %         u_profile = 2*u_profile/(max(u_profile) - min(u_profile));
    v_profile = (v_profile-mean(v_profile))/k;
    
    jj=1;
    if mean(v_profile(1:2))> mean(v_profile(end-1:end))
%         v_profile = flip(v_profile);
        jj=jj+1;
    end
    % %         jj_count(count,1) = jj-1;
    
    % - NORMALISATION
    %         v_profile = 2*v_profile/(max(v_profile) - min(v_profile));
    u_std = u_std/k  / sqrt(k*size(X,1));
    v_std = v_std/k  / sqrt(k*size(X,1));
    
    XX = X(1,:);%-X(1,1);
    XX = (XX - XX(end)/2);%/XX(end); % - NORMALISATION
    
    % % % % % %  insert velocity profiles to EXP   % % % % % %
    EXP{i,1} = 2*XX(end);
    %         EXP{count,2} = 1/2*(abs(v_profile(end))+abs(v_profile(end)))/(2*XX(end));% norm by width
    %         EXP{count,3} = 1/2*(abs(u_profile(end))+abs(u_profile(end)))/(2*XX(end));% norm by width
    EXP{i,2} = 1/2*(abs(v_profile(1))+abs(v_profile(end)));
    EXP{i,3} = 1/2*(abs(u_profile(1))+abs(u_profile(end)));
    EXP{i,4} = jj-1;
    EXP{i,5} = [XX', u_profile',u_std'];
    EXP{i,6} = [XX', v_profile',v_std'];
    u_profile = 2*u_profile/(max(u_profile) - min(u_profile));
    EXP{i,7} = [XX'/XX(end), u_profile',u_std'];
    v_profile = 2*v_profile/(max(v_profile) - min(v_profile));
    EXP{i,8} = [XX'/XX(end), v_profile',v_std'];
    
    EXP{i,9} = v_profile(1)-v_profile(end);
    EXP{i,10} = u_profile(1)-u_profile(end);

end
%%
% load('F:\GD\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
for i=1:size(dirOP,1)
%     plot(EXP{i,6}(:,1),EXP{i,6}(:,2)); hold on
    shear(i,1) = EXP{i,6}(1,2)-EXP{i,6}(end,2);
end
cd=150;% shift jet colors
c1=jet(size(OP_mid,1)+cd);
c=c1(1:end-cd,:);
% shear = cell2mat(EXP(:,3));
V_OP_mAng = [width, shear,OP_mid, Ang_mid];
V_OP_mAng = sortrows(V_OP_mAng,1);
figure(22);subplot(1,3,1)
scatter(V_OP_mAng(:,3),V_OP_mAng(:,2),15,c,'filled');set(gca,'Fontsize',18);
xlabel('Order Parameter'); ylabel('v_y (\mum/hr)');axis([0 1 -10 10]);%axis tight;%hold off
figure(22);subplot(1,3,2)
scatter(V_OP_mAng(:,1),V_OP_mAng(:,3),15,c,'filled');set(gca,'Fontsize',18);
ylabel('Order Parameter'); xlabel('Width (\mum)');axis([0 1040 0 1]);
figure(22);subplot(1,3,3)
scatter(V_OP_mAng(:,1),V_OP_mAng(:,2),15,c,'filled');set(gca,'Fontsize',18);
ylabel('v_y (\mum/hr)'); xlabel('Width (\mum)');axis([0 1040 -10 10]);
figure(24);subplot(1,3,1:3)
scatter(V_OP_mAng(:,2),V_OP_mAng(:,4),15,c,'filled');set(gca,'Fontsize',18);
ylabel('Angle (deg)'); xlabel('v_y (\mum/hr)');%axis([-inf inf 0 6]);%axis tight;%hold off
figure(25);subplot(1,3,1:3)
scatter(V_OP_mAng(:,1),abs(V_OP_mAng(:,4)-90),15,c,'filled');set(gca,'Fontsize',18);
ylabel('Angle of Tilt (deg)'); xlabel('Width (\mum)');axis([0 1040 -inf inf]);%axis tight;%hold off

%%
figure(123)
widthPIV = cell2mat(EXP(:,1));
plot(widthPIV,width,'.')
figure(124)
plot(widthPIV-width,'.')

%% MAKE AVERAGE PLOT (Shear/Angle)
% load('F:\GD\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
figure(223)
dBins = .3;% Bin resolution
dd = 1.5*dBins;% range
totalBins = floor((max(shear)- min(shear))/dBins);
[N, edges] = histcounts(shear,totalBins);
edgess = mean([edges(1:end-1); edges(2:end)]);
[Peak,widthOfPeak] = findpeaks(N,edgess);
plot(edgess,N,'-'); hold on
plot(widthOfPeak,Peak,'o'); hold off
% histogram(pks,floor(length(pks)/3))
% NEW X-SCALE
AV = zeros(length(widthOfPeak),4);
AV(:,1) = widthOfPeak;
for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    AV(i,2) = mean(Ang_mid(shear>=ww-dd & shear<=ww+dd));  
    AV(i,3) = std(Ang_mid(shear>=ww-dd & shear<=ww+dd))./...
        sum(shear>=ww-dd & shear<=ww+dd)^.5;
    AV(i,4) = sum(shear>=ww-dBins & shear<=ww+dBins);
end
figure(24); hold on
% scatter(shear,OP_mid,10,'fill'); hold on
% plot(widthOP,OP,'.');hold on
errorbar(AV(:,1),AV(:,2),AV(:,3));hold off

%% MAKE AVERAGE PLOT (width/abs(Tilt))
% load('F:\GD\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
figure(223)
dBins = 7;% Bin resolution
dd = 1*dBins;% range
totalBins = floor((max(width)- min(width))/dBins);
[N, edges] = histcounts(width,totalBins);
edgess = mean([edges(1:end-1); edges(2:end)]);
[Peak,widthOfPeak] = findpeaks(N,edgess);
plot(edgess,N,'-'); hold on
plot(widthOfPeak,Peak,'o'); hold off
% histogram(pks,floor(length(pks)/3))
% NEW X-SCALE
% widthOfPeak = min(width):dBins:max(width);
AV = zeros(length(widthOfPeak),4);
AV(:,1) = widthOfPeak;
% Ang_mid = abs(Ang_mid-90);
for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    AV(i,2) = median(Ang_mid(width>=ww-dd & width<=ww+dd & Ang_mid>90));  
    AV(i,3) = std(Ang_mid(width>=ww-dd & width<=ww+dd & Ang_mid>90))./...
        sum(width>=ww-dd & width<=ww+dd & Ang_mid>90)^.5;
    AV(i,4) = sum(width>=ww-dBins & width<=ww+dBins);
end

figure(122);hold on
scatter(V_OP_mAng(:,1),(V_OP_mAng(:,4)),15,c,'filled');set(gca,'Fontsize',18);
ylabel('Angle (deg)'); xlabel('Width (\mum)');axis([0 1040 90 inf]);%axis tight;%hold off
errorbar(AV(:,1),AV(:,2),AV(:,3));hold off

%% MAKE AVERAGE PLOT (width/OP)
% load('F:\GD\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
figure(223)
dBins = 7;% Bin resolution
dd = 1*dBins;% range
totalBins = floor((max(width)- min(width))/dBins);
[N, edges] = histcounts(width,totalBins);
edgess = mean([edges(1:end-1); edges(2:end)]);
[Peak,widthOfPeak] = findpeaks(N,edgess);
plot(edgess,N,'-'); hold on
plot(widthOfPeak,Peak,'o'); hold off
% histogram(pks,floor(length(pks)/3))
% NEW X-SCALE
% widthOfPeak = min(width):dBins:max(width);
AV = zeros(length(widthOfPeak),4);
AV(:,1) = widthOfPeak;
shear = abs(shear);
for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    AV(i,2) = mean(OP_mid(width>=ww-dd & width<=ww+dd));  
    AV(i,3) = std(OP_mid(width>=ww-dd & width<=ww+dd))./...
        sum(width>=ww-dd & width<=ww+dd)^.5;
    AV(i,4) = sum(width>=ww-dBins & width<=ww+dBins);
end

figure(122);hold on
scatter(V_OP_mAng(:,1),V_OP_mAng(:,3),15,c,'filled');set(gca,'Fontsize',18);
ylabel('Order Parameter'); xlabel('Width (\mum)');axis([0 1040 0 1]);
errorbar(AV(:,1),AV(:,2),AV(:,3));hold off