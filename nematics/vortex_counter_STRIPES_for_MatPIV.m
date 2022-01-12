%%
% Ddir = dir(['V:\HT1080_small_stripes_glass_22112017\CROPPED\PIV_DATs' '\*.tif']);
Ddir = dir(['C:\Users\victo\Google Drive\DATA\HT1080\PIV_DATs' '\*.mat']);
folder_main = Ddir(1).folder;
filesInFolder = size(Ddir,1);
px_size = .74;

vort_area = zeros(size(Ddir,1),1);
vort_width = zeros(size(Ddir,1),1);
vortNum = cell(size(Ddir,1),1);
vortDensity = cell(size(Ddir,1),1);
vortArea = cell(size(Ddir,1),1);
% vortPosition = cell(size(Ddir,1),1);
dt=1;
%%
for i=1:size(Ddir,1)
%     i=4
%     if contains(Ddir(i).name, '.mat' )
        
        clear var Ok_Wi
        Ddir(i).name
        disp(['file: ' num2str(i) ' from: ' num2str(size(Ddir,1))]);
        filepath = [Ddir(i).folder '\' Ddir(i).name];        
        load(filepath);
        X = x{1}; Y = y{1}; dx=x{1}(1,2)-x{1}(1,1);
        vort_area(i) = px_size^2*Y(end,end)*X(end,end); %STRIPE AREA in um
        vort_width(i) = px_size*X(end,end);             %STRIP WIDTH in um
        
        for k=1:size(x,1)
%             k=10
            ff = 7; filt = fspecial('gaussian',ff,ff);    
            % --------------------------PIV import ---------------------------------
            uu = zeros(size(u(k))); vv = uu;
            uu = imfilter(u{k}, filt);
            vv = imfilter(v{k}, filt);

            % Vorticity
            [u_x,u_y] = gradient(uu);%/dx gradient need to be corrected for the dx
            [v_x,v_y] = gradient(vv);%/dx
            vorticity = (v_x - u_y);
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
%             bw = u2_1;
%             D = bwdist(~bw);
%             D = -D;
%             D(~bw) = Inf;
%             WS = watershed(D,4);
%             WS(~bw) = 0;
%             WSflipped = flipud(WS);
            
            s = regionprops('table', WS, vorticity,'centroid','Area','MeanIntensity');
            vortNum{i}(k,1) = size(s,1);
            vortDensity{i}(k,1) = size(s,1)/vort_area(i); 
            vortArea{i}(k,1) = px_size^2*(X(1,2)-X(1,1))^2*mean(s.Area);
            vortAreaAll{i,k} = px_size^2*(X(1,2)-X(1,1))^2*s.Area;
% ------------------VORTEX AREA DISTRIBUTION--------------------------------------           
% --------------------------------------------------------------------------------
% ------------------VORTEX AREA DISTRIBUTION--------------------------------------           
% --------------------------------------------------------------------------------

%             % Velocity
%             figure(23);
%             vstep = 1;
%             q0 = quiver(1/dx*X(1:vstep:end,1:vstep:end),1/dx*Y(1:vstep:end,1:vstep:end),...
%                 uu(1:vstep:end,1:vstep:end),vv(1:vstep:end,1:vstep:end),3);
%             q0.LineWidth=.5;
%             q0.Color = [0 0 0];hold on;axis equal;
% 
%             figure(23);
%             surf(WS-100); hold on
%             scatter(s.Centroid(:,1),s.Centroid(:,2),20,[0 1 0],'filled');
%             hold off
%             view(2);shading interp;colormap jet;axis equal;axis tight;%axis off
%         end
    end
end

%% Area distribution by fitting vortex area distribution
clear
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\vortex number_All.mat')

emptyTF = cellfun(@isempty,vortAreaAll(:,1)); %check for empty cells
vortRadii = zeros(sum(emptyTF==0),1);
vortWidth = vortRadii;
edges = 50:100:6e3;
count=1;
kk = 4;
for i=1:size(vortAreaAll,1)
    if emptyTF(i)==0
        if vort_width(i)>150
        areaMatAll = cell2mat(vortAreaAll(i,:)');
        N = histcounts(areaMatAll,edges, 'Normalization', 'probability');
        N1 = N(N>0);
        edges1 = edges(N>0);       
        f = fit(edges1',N1','exp1'); %Y = a*exp(b*x)
        
%         figure(100)%;plot(edges1,N1);hold on
%         plot(f,edges1,N1);set(gca, 'YScale', 'log');set(gca, 'YScale', 'log');hold on
        
        vortRadii(count,1) =  2* sqrt(-1/f.b/pi);% notice "2*" (its actuallly diameter)
        vortWidth(count,1) = vort_width(i);
        count=count+1;
        end
    end
end
figure(100); xlabel('Vortex area (\mum^2)');ylabel('PDF'); set(gca,'FontSize',20);
% hold off
figure(101); hold on
p1 = plot(vortWidth,vortRadii,'o');
p1.MarkerSize = 10; p1.MarkerEdgeColor= [1 1 1]; p1.MarkerFaceColor= [0 .5 .8];
xlabel('Width (\mum)');ylabel('Vortex diameter (\mum)'); set(gca,'FontSize',20);
%% MAKE AVERAGE PLOT
figure(123)
maxBin = 5;% half of maximal delta width 
dBins = 6;% Bin resolution
dd = dBins+6;% range
totalBins = floor((max(vortWidth)- min(vortWidth))/dBins);
[N, edges] = histcounts(vortWidth,totalBins);
edgess = mean([edges(1:end-1); edges(2:end)]);
[Peak,widthOfPeak] = findpeaks(N,edgess);
plot(edgess,N,'-'); hold on
plot(widthOfPeak,Peak,'o'); hold off
% histogram(pks,floor(length(pks)/3))
% NEW X-SCALE
MvortRadii = zeros(length(widthOfPeak),4);
MvortRadii(:,1) = widthOfPeak;
for i=1:length(widthOfPeak)
    ww = widthOfPeak(i);
    MvortRadii(i,2) = mean(vortRadii(vortWidth>=ww-dd & vortWidth<=ww+dd));  
    MvortRadii(i,3) = std(vortRadii(vortWidth>=ww-dd & vortWidth<=ww+dd))./...
        sum(vortWidth>=ww-dd & vortWidth<=ww+dd)^.5;
    MvortRadii(i,4) = sum(vortWidth>=ww-dBins & vortWidth<=ww+dBins);
end
figure(124)
plot(vortWidth,vortRadii,'.');hold on
errorbar(MvortRadii(:,1),MvortRadii(:,2),MvortRadii(:,3));hold off

%%
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\vortex number_All.mat')

emptyTF = cellfun(@isempty,vortDensity); %check for empty cells
count=1;
for i=1:size(vortDensity,1)
    if emptyTF(i)==0
        vortWidth(count,1) = vort_width(i);
        meanVortDensity(count,1) = mean(vortDensity{i});
        stdVortDensity(count,1) = std(vortDensity{i})/size(vortDensity{i},1)^.5;
        
        meanVortNum(count,1) = mean(vortNum{i});
        stdVortNum(count,1) = std(vortNum{i})/size(vortDensity{i},1)^.5;
        
        meanVortArea(count,1) = mean(vortArea{i});
        stdVortArea(count,1) = std(vortArea{i})/size(vortDensity{i},1)^.5;
        count=count+1;
    end
end

[sortVort_width, ind] = sort(vortWidth);
sortVort_area = vort_area(ind);
meanVortDensity = meanVortDensity(ind);
stdVortDensity = stdVortDensity(ind);
meanVortNum = meanVortNum(ind);
stdVortNum = stdVortNum(ind);
figure(1)
% errorbar(sortVort_width,meanVortDensity,stdVortDensity);hold on
p1=plot(sortVort_width,meanVortDensity,'o');hold off
p1.MarkerSize = 10; p1.MarkerEdgeColor= [1 1 1]; p1.MarkerFaceColor= [0 .5 .8];
xlabel('Width (\mum)');ylabel('Density of vortices (\mum^{-2})'); set(gca,'FontSize',20);
axis([0 1000 0 2e-4]);
figure(2)
errorbar(sortVort_width,meanVortNum,stdVortNum);hold on
figure(3)
% errorbar(sortVort_width,meanVortArea,stdVortArea);hold on
p3=plot(sortVort_width,meanVortArea,'o');hold off
p3.MarkerSize = 10; p3.MarkerEdgeColor= [1 1 1]; p3.MarkerFaceColor= [0 .5 .8];
xlabel('Width (\mum)');ylabel('Area (\mum^{2})'); set(gca,'FontSize',20);
% axis([0 1000 0 2e-4]);