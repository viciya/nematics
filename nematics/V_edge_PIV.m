%% correspond PIV and Orientation names
dirOP = dir(['C:\Users\vici\Google Drive\DATA\HT1080\Orient'  '\*.tif']);
dirPIV = dir(['C:\Users\vici\Google Drive\DATA\HT1080\PIV_DATs'  '\*.mat']);

% dirOP = dir(['C:\Users\victo\Google Drive\DATA\HT1080\Orient'  '\*.tif']);
% dirPIV = dir(['C:\Users\victo\Google Drive\DATA\HT1080\PIV_DATs'  '\*.mat']);

for i = 1:size(dirOP,1)
    
    nameOP = dirOP(i).name;
    p1=strfind(nameOP,'s');
    p2=strfind(nameOP,'.tif');
    patternOP = nameOP(p1:p2);
    
%     pA=strfind(nameOP,'Orient');
%     pB=strfind(nameOP,'oldOrient');
    pC=strfind(nameOP,'old');
    pD=strfind(nameOP,'09_07_2018_Orient');
    pE=strfind(nameOP,'26072018_Orient');
    pF=strfind(nameOP,'01082018_Orient');
    
    if ~isempty(pF)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name,'01082018') &&...
                   contains(dirPIV(j).name, patternOP)
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;
            end
        end
    elseif ~isempty(pE)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name,'26072018') &&...
               contains(dirPIV(j).name, patternOP)
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;                
            end
        end
     elseif ~isempty(pD)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name,'09072018') &&...
               contains(dirPIV(j).name, patternOP)
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;                
            end
        end
     elseif ~isempty(pC)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name,'old') &&...
               contains(dirPIV(j).name, patternOP)
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;                
            end
        end 
     elseif isempty(pC)&isempty(pD)&isempty(pE)&isempty(pF)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name, patternOP)&&...
                ~contains(dirPIV(j).name,'01082018') &&...
                ~contains(dirPIV(j).name,'26072018') &&...
                ~contains(dirPIV(j).name,'09072018') &&...
                ~contains(dirPIV(j).name,'old')                
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;                
            end
        end         
    end
end
%% check name compatability for PIV and OP
count =1;
for i = 1:size(dirOP,1)
    nameOP = dirOP(i).name;
    namePIV = dirPIV(indX(i,2)).name;
    p7=strfind(nameOP,'09_07_2018');
    p8=strfind(namePIV,'09072018');
    if ~isempty(p7) && ~isempty(p8)
    namePIV 
    nameOP
    count=count+1;
    end
end
%%
px_sizeOP = .74*3;
px_sizePIV = .74;

OP_mid = zeros(size(dirOP,1),1);
OP_mid_std = OP_mid;
width = OP_mid;
Orient_area = OP_mid;
Orient_width = OP_mid;
Ang_mid = zeros(size(dirOP,1),1);
Ang_mid_std = Ang_mid;
defNum = cell(size(dirOP,1),1);
defDensity = cell(size(dirOP,1),1);
def_NumAv = OP_mid;
def_DensityAv = OP_mid;
defPOS = cell(size(dirOP,1),1);

EXP = cell(size(dirOP,1),10);
vortNum = cell(size(dirOP,1),1);
vortPOS = cell(size(dirOP,1),1);
vortNumAv = zeros(size(dirOP,1),1);
vEnergy = zeros(size(dirOP,1),1);

% kk = 11;kkk=29;
% for i=kk:kk
for i=1:size(dirOP,1)
%--------------------PIV---------------------------------------------------
    ff = 7; filt = fspecial('gaussian',ff,ff);
    clear var Ok_Wi
    filepathPIV = [dirPIV(indX(i,2)).folder '\' dirPIV(indX(i,2)).name];
    load(filepathPIV);    
    X = px_sizePIV*x{1,1}; Y = px_sizePIV*y{2,1};
    u_profile = zeros(1,size(X,2));
    v_profile = zeros(1,size(X,2));
    u_std = zeros(1,size(X,2));
    v_std = zeros(1,size(X,2));
    vEnergy_temp = 0;    
    
%     for k=kkk:kkk
    for k=1:size(x,1)
        u_profile = u_profile + mean(u{k});
        u_std = u_std + std(u{k});
        v_profile = v_profile + mean(v{k});
        v_std = u_std + std(v{k});
        % ------ Vortex counter -------
        uu = zeros(size(u(k))); vv = uu;
        uu = imfilter(u{k}, filt);
        vv = imfilter(v{k}, filt);
        dx = px_sizePIV*(x{1}(1,2)-x{1}(1,1));
        [u_x,u_y] = gradient(uu,dx);%/dx gradient need to be corrected for the dx
        [v_x,v_y] = gradient(vv,dx);%/dx
        vorticity = (v_x - u_y);
        vEnergy_temp = mean2(vEnergy_temp + 0.5*vorticity.^2);
        ff = 5;
        filt = fspecial('gaussian',ff,ff);
        u1 = imfilter(vorticity, filt);
        
        Ok_Wi(:,:)  = (u_x+v_y).^2-4*(u_x.*v_y-v_x.*u_y);
        ff1 = 5; ffilt = fspecial('gaussian',ff1,ff1);
        u1 = vorticity(:,:);
        u2 = imfilter(Ok_Wi(:,:), ffilt);
        u2_1 = u2 < min(u2(:))/7;
        u2_1 = bwareaopen(u2_1, 4, 4);
        WS = bwlabel(u2_1);
%         imagesc(qq);hold on;quiver(uu,vv,2,'color','r');axis equal;hold off
        s = regionprops('table', WS, vorticity,'centroid','Area','MeanIntensity');
        if ~isempty(s)
            vortNum{i}(k,1) = size(s,1);
            vortPOS{i}{k,1}(:,:) = [s.Centroid(:,1),s.Centroid(:,2)];%size(s,1);  
        end
    end
    vEnergy(i) = vEnergy_temp/k;
    vortNumAv(i) = mean(vortNum{i});
    
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

% -------------------------PIV END-----------------------------------------


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
% %%
% save 'C:\Users\vici\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new2.mat'
%%
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new2.mat')
% load('C:\Users\vici\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new2.mat')
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
% ---------------------------------------------------------------------
figure(22);
    subplot(1,3,1)
scatter(V_OP_mAng(:,3),V_OP_mAng(:,2),15,c,'filled');set(gca,'Fontsize',18);
xlabel('Order Parameter'); ylabel('v_y (\mum/hr)');axis([0 1 -10 10]);%axis tight;%hold off
	subplot(1,3,2)
scatter(V_OP_mAng(:,1),V_OP_mAng(:,3),15,c,'filled');set(gca,'Fontsize',18);
ylabel('Order Parameter'); xlabel('Width (\mum)');axis([0 1040 0 1]);
	subplot(1,3,3)
scatter(V_OP_mAng(:,1),V_OP_mAng(:,2),15,c,'filled');set(gca,'Fontsize',18);
ylabel('v_y (\mum/hr)'); xlabel('Width (\mum)');axis([0 1040 -10 10]);
% ---------------------------------------------------------------------
figure(23);
    subplot(1,3,1)
scatter(V_OP_mAng(:,2),V_OP_mAng(:,4),15,c,'filled');set(gca,'Fontsize',18);
ylabel('Angle (deg)'); xlabel('v_y (\mum/hr)');%axis([-inf inf 0 6]);%axis tight;%hold off
	subplot(1,3,2)
scatter(V_OP_mAng(:,1),abs(V_OP_mAng(:,4)-90),15,c,'filled');set(gca,'Fontsize',18);
ylabel('|Angle of Tilt| (deg)'); xlabel('Width (\mum)');axis([0 1040 -inf inf]);%axis tight;%hold off
	subplot(1,3,3)
scatter(V_OP_mAng(:,1),V_OP_mAng(:,4)-90,15,c,'filled');set(gca,'Fontsize',18);
ylabel('Angle of Tilt (deg)'); xlabel('Width (\mum)');axis([0 1040 -inf inf]);%axis tight;%hold off
% ---------------------------------------------------------------------
figure(24);
    subplot(1,3,1)
scatter(abs(V_OP_mAng(:,2)),V_OP_mAng(:,5),15,c,'filled');set(gca,'Fontsize',18);
xlabel('v_y (\mum/hr)'); ylabel('Number of Vortices');%axis([0 1040 -inf inf]);%
axis tight;hold off
	subplot(1,3,2)
scatter(abs(V_OP_mAng(:,2)),V_OP_mAng(:,5)./V_OP_mAng(:,1),15,c,'filled');set(gca,'Fontsize',18);
xlabel('v_y (\mum/hr)'); ylabel('Vortex Density');%axis([0 1040 -inf inf]);%
axis tight;hold off
	subplot(1,3,3)
scatter(V_OP_mAng(:,1),V_OP_mAng(:,5),15,c,'filled');set(gca,'Fontsize',18);
xlabel('Width (\mum)'); ylabel('Number of Vortices');%axis([0 1040 -inf inf]);%axis tight;
hold off
% ---------------------------------------------------------------------
%% OP mid vs. |shear| + fit data
    ww = 177; dw = 20;% choose width and +/- delta
    width1 = V_OP_mAng(:,1); 
    shear1 = V_OP_mAng(width1<ww+dw & width1>ww-dw,2);
    OP1 = V_OP_mAng(width1<ww+dw & width1>ww-dw,3);
    vorts = V_OP_mAng(width1<ww+dw & width1>ww-dw,5)./...
        V_OP_mAng(width1<ww+dw & width1>ww-dw,1);
    vEnrg = V_OP_mAng(width1<ww+dw & width1>ww-dw,6);

figure(25);
    subplot(1,3,1:3)
scatterFit = fit( abs(shear1), vEnrg, 'poly1','Normalize','on','Robust','Bisquare');
scatter(abs(shear1),vEnrg,15,'filled');hold on
plot(scatterFit, 'predobs' );
xlabel('Shear (\mum/hr)'); ylabel('Vortex Density');set(gca,'Fontsize',12);
% axis([0 10 0 1]);
% axis tight;
hold off
scatterFit
confint(scatterFit,0.95)  %95% confidence bounds
%%
figure(123)
widthPIV = cell2mat(EXP(:,1));
plot(widthPIV,width,'-o')
figure(124)
plot(widthPIV-width,'.')

%% MAKE AVERAGE PLOT (Shear/Angle)
% load('F:\GD\Curie\DESKTOP\HT1080\shear_OP_correlation_new1.mat')
% load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new1.mat')
figure(223)
dBins = .25;% Bin resolution
dd = 2*dBins;% range or delta/2
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
% load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
figure(223)
dBins = 15;% Bin resolution
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

figure(122);
scatter(V_OP_mAng(:,1),(V_OP_mAng(:,4)),15,c,'filled');set(gca,'Fontsize',18); hold on
ylabel('Angle (deg)'); xlabel('Width (\mum)');axis([0 1040 90 inf]);%axis tight;%hold off
p1 = errorbar(AV(:,1),AV(:,2),AV(:,3));
p1.LineWidth=2;p1.Color = [0 0 0];
hold off
%% MAKE AVERAGE PLOT (width/OP)
% load('F:\GD\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
% load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new.mat')
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