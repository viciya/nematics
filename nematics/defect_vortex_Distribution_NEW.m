%%  ALL OPTIONS COMBINED
% 1. EXPERIMENT
% 2. RANDOM SET NUMBER OF POINTS SET BY EXPERIMENT
% 2. RANDOM SET FINITE SIZE (non overlaping) DISKS NUMBER OF POINTS SET BY EXPERIMENT
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
clear var DistAll Dist DistMode RDist RdDist posArr
nDef = 50;
dw = 20;% choose width and +/- delta
Dist = []; RDist = []; RdDist = []; posArr=[];
vortRadi = 27;
mask0 = zeros(2*vortRadi+1,2*vortRadi+1);
[XX,YY] = meshgrid(-vortRadi:vortRadi,-vortRadi:vortRadi);

vNumAll = 0;
kk = 1;
for ww=600 % loop to check different stripe width
    count = 1;
    for i=1:size(vortPOS,1)  % loop to go over all stripes
        ccount = 0;
        vNumCount = 0;
        stripeIdx = indByWidth(i);% sorts stipes by width
        % ----  choose only size around width=ww+/-
%         if width(stripeIdx)<ww+dw && width(stripeIdx)>ww-dw
        if width(stripeIdx)>800
            % ----  go trough all time frames
              for time = size(vortPOS{stripeIdx},1)-size(vortPOS{stripeIdx},1)+1:size(vortPOS{stripeIdx},1)
%             for time = size(vortPOS{stripeIdx},1)-30:size(vortPOS{stripeIdx},1)
                vNum = size(vortPOS{stripeIdx}{time},1);% number of vorticies
                if vNum>4
                    pos = vortPOS{stripeIdx}{time};
                    [~,DistTemp] = knnsearch(pos,pos,'k',nDef,'distance','euclidean');
                    %                                     disp('*');
                    Temp = NaN(size(DistTemp,1),nDef);
                    Temp(1:size(DistTemp,1),1:size(DistTemp,2))= DistTemp;
                    Dist = vertcat(Dist,Temp);
                    %- - - - - RANDOM ARRAY - - -- -
                    Rpos = rand(vNum,2)...
                        .*(max(vortPOS{stripeIdx}{time})- min(vortPOS{stripeIdx}{time}));
                    [~,DistTemp] = knnsearch(Rpos,Rpos,'k',nDef,'distance','euclidean');
                    %                                     disp('*');
                    Temp = NaN(size(DistTemp,1),nDef);
                    Temp(1:size(DistTemp,1),1:size(DistTemp,2))= DistTemp;
                    RDist = vertcat(RDist,Temp);
                    %- - - - - RANDOM DISCS - - -- -
                    Frame = zeros(1500+2*vortRadi,round(width(stripeIdx))+2*vortRadi);
                    n = vNum;
                    cl = 1;
                    ccl = 1;
                    while n>0
                        if ccl<1e2   
                        p0 = ceil(rand(1,2).*[1500,round(width(stripeIdx))]);
                        if sum(sum(Frame(p0(1):p0(1)+2*vortRadi,p0(2):p0(2)+2*vortRadi)))==0
                            Frame(p0(1):p0(1)+2*vortRadi,p0(2):p0(2)+2*vortRadi) = mask0;
                            posArr(cl,:) = p0;
                            cl=cl+1;
                            n=n-1;
                        end
                        ccl=ccl+1;
                        end
                    end
                    [~,DistTemp] = knnsearch(posArr,posArr,'k',nDef,'distance','euclidean');
                    Temp = NaN(size(DistTemp,1),nDef);
                    Temp(1:size(DistTemp,1),1:size(DistTemp,2))= DistTemp;
                    RdDist = vertcat(RdDist,Temp);
                    %-------------------------------------
                    
                    vNumCount = vNumCount+vNum;
                    ccount = ccount+1;
                end
            end
            vNumAll(count) = vNumCount/ccount;
            count = count+1;
        end
        kk=kk+1;
    end
    
    for j=1:nDef
        S(j) = nanstd(Dist(:,j))...
            /sqrt(sum(~isnan(Dist(:,j))));
        RS(j) = nanstd(RDist(:,j))...
            /sqrt(sum(~isnan(RDist(:,j))));
        RdS(j) = nanstd(RdDist(:,j))...
            /sqrt(sum(~isnan(RdDist(:,j))));
    end
    M = nanmedian(Dist);%mode(round(Dist));
    DistMode(:,1) = M'/M(2);
    DistMode(:,2) = S'/M(2);
    
    RM = nanmedian(RDist);%mode(round(RDist));
    DistMode(:,3) = RM'/RM(2);
    DistMode(:,4) = RS'/RM(2);
    
    RdM = nanmedian(RdDist);%mode(round(PDist));
    DistMode(:,5) = RdM'/RdM(2);
    DistMode(:,6) = RdS'/RdM(2);
    %     errorbar((1:nDef)'-1,DistMode(:,2*kk-1),DistMode(:,2*kk)); hold on
    figure(ww)
    shadedErrorBar((1:nDef)'-1,M'/M(2),S'/M(2),'lineprops','-ob','transparent',1);hold on
    shadedErrorBar((1:nDef)'-1,RM'/RM(2),RS'/RM(2),'lineprops','-or','transparent',1);
    shadedErrorBar((1:nDef)'-1,RdM'/RdM(2),RdS'/RdM(2),'lineprops','-og','transparent',1);
    
    kk=kk+1;

legend(['defect: ',num2str(nanmean(vNumAll),3),'|',num2str(nanstd(vNumAll),3)])% axis equal
set(gca,'Fontsize',18);axis tight
ylabel('Normalized Distance'); xlabel('Neighbor');
axis([0 round(nanmean(vNumAll)) 0 10]);
end


%%
load("C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new2.mat")
%% DEFECT NEW VERSION (GETS MODE/mean VALUE for N-th neighbour FOR EACH STRIPE WIDTH)
clear var DistAll Dist DistMode
nDef = 20;
dw = 20;% choose width and +/- delta
Dist = [];
count = 1;

for ww=200 % loop to check different stripe width
    for i=1:size(defPOS,1)  % loop to go over all stripes
        stripeIdx = indByWidth(i);% sorts stipes by width
        % ----  choose only size around width=ww+/-
        if width(stripeIdx)<ww+dw && width(stripeIdx)>ww-dw            
%             if width(stripeIdx)>250
%             stripeIdx
%             width(stripeIdx)
            % ----  go trough all time frames
            for time = size(defPOS{stripeIdx},1)-size(defPOS{stripeIdx},1)+1:size(defPOS{stripeIdx},1)
%             for time = size(defPOS{stripeIdx},1)-30:size(defPOS{stripeIdx},1)
                if size(defPOS{stripeIdx}{time},1)>10
                    pos = defPOS{stripeIdx}{time};
                    [~,DistTemp] = knnsearch(pos,pos,'k',nDef,'distance','euclidean');
%                                     disp('*');
                    Temp = NaN(size(DistTemp,1),nDef);
                    Temp(1:size(DistTemp,1),1:size(DistTemp,2))= DistTemp;
                    Dist = vertcat(Dist,Temp);                    
                end
            end            
        end        
    end  
    for j=1:nDef
%         j=10
        S(j) = std(Dist(~isnan(Dist(:,j)),j))...
            /sqrt(sum(~isnan(Dist(:,j))));
    end
    M = mode(round(Dist));     
    DistMode(:,2*count-1) = M';%/M(2);
    DistMode(:,2*count) = S';%/M(2);
%     
%     errorbar((1:nDef)'-1,DistMode(:,2*count-1),DistMode(:,2*count)); hold on
    shadedErrorBar((1:nDef)'-1,DistMode(:,2*count-1)...
        ,DistMode(:,2*count),'lineprops','-o','transparent',1);
    count=count+1;
end
% axis equal 
% % hold off
%% -1- VORTEX NEW VERSION (GETS MODE/mean VALUE for N-th neighbour FOR EACH STRIPE WIDTH)
clear var DistAll Dist DistMode
nDef = 20;
dw = 20;% choose width and +/- delta
Dist = [];
count = 1;

for ww=200 % loop to check different stripe width
    for i=1:size(vortPOS,1)  % loop to go over all stripes
        stripeIdx = indByWidth(i);% sorts stipes by width
        % ----  choose only size around width=ww+/-
        if width(stripeIdx)<ww+dw && width(stripeIdx)>ww-dw            
%             if width(stripeIdx)>250
%             stripeIdx
%             width(stripeIdx)
            % ----  go trough all time frames
            for time = size(vortPOS{stripeIdx},1)-size(vortPOS{stripeIdx},1)+1:size(vortPOS{stripeIdx},1)
%             for time = size(vortPOS{stripeIdx},1)-30:size(vortPOS{stripeIdx},1)
                if size(vortPOS{stripeIdx}{time},1)>4
                    pos = vortPOS{stripeIdx}{time};
                    [~,DistTemp] = knnsearch(pos,pos,'k',nDef,'distance','euclidean');
%                                     disp('*');
                    Temp = NaN(size(DistTemp,1),nDef);
                    Temp(1:size(DistTemp,1),1:size(DistTemp,2))= DistTemp;
                    Dist = vertcat(Dist,Temp);                    
                end
            end            
        end        
    end  
    for j=1:nDef
%         j=10
        S(j) = std(Dist(~isnan(Dist(:,j)),j))...
            /sqrt(sum(~isnan(Dist(:,j))));
    end
    M = mode(round(Dist));     
    DistMode(:,2*count-1) = M';%/M(2);
    DistMode(:,2*count) = S';%/M(2);
%     
%     errorbar((1:nDef)'-1,DistMode(:,2*count-1),DistMode(:,2*count)); hold on
    shadedErrorBar((1:nDef)'-1,DistMode(:,2*count-1)...
        ,DistMode(:,2*count),'lineprops','-o','transparent',1);
    count=count+1;
end
% axis equal 
% % hold off
%% -2- VORTEX NEW VERSION compares to RAND and PERIODIC Distribution
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
clear var DistAll Dist DistMode
nDef = 20;
dw = 20;% choose width and +/- delta
Dist = []; RDist = []; PDist = []; 

vNumAll = 0; 
cloop = 1;
for ww=90 % loop to check different stripe width
    count = 1;
    for i=1:size(vortPOS,1)  % loop to go over all stripes
        ccount = 0;
        vNumCount = 0;
        stripeIdx = indByWidth(i);% sorts stipes by width
        % ----  choose only size around width=ww+/-
        if width(stripeIdx)<ww+dw && width(stripeIdx)>ww-dw            
%             if width(stripeIdx)>600

            % ----  go trough all time frames
%             for time = size(vortPOS{stripeIdx},1)-size(vortPOS{stripeIdx},1)+1:size(vortPOS{stripeIdx},1)
            for time = size(vortPOS{stripeIdx},1)-30:size(vortPOS{stripeIdx},1)
                vNum = size(vortPOS{stripeIdx}{time},1);% number of vorticies
                if vNum>4
                    pos = vortPOS{stripeIdx}{time}; 
                    [~,DistTemp] = knnsearch(pos,pos,'k',nDef,'distance','euclidean');
%                                     disp('*');
                    Temp = NaN(size(DistTemp,1),nDef);
                    Temp(1:size(DistTemp,1),1:size(DistTemp,2))= DistTemp;
                    Dist = vertcat(Dist,Temp); 
                    %- - - - - RANDOM ARRAY - - -- - 
                    Rpos = rand(vNum,2)...
                        .*(max(vortPOS{stripeIdx}{time})- min(vortPOS{stripeIdx}{time}));
                    [~,DistTemp] = knnsearch(Rpos,Rpos,'k',nDef,'distance','euclidean');
%                                     disp('*');
                    Temp = NaN(size(DistTemp,1),nDef);
                    Temp(1:size(DistTemp,1),1:size(DistTemp,2))= DistTemp;
                    RDist = vertcat(RDist,Temp);                 
                    %- - - - - PERIODIC ARRAY - - -- - 
%                     Ppos = ones(vNum,2).*(1:vNum)'/vNum...
%                         .*(max(vortPOS{stripeIdx}{time})- min(vortPOS{stripeIdx}{time}));                    
%                     [~,DistTemp] = knnsearch(Ppos,Ppos,'k',nDef,'distance','euclidean');
% %                                     disp('*');
%                     Temp = NaN(size(DistTemp,1),nDef);
%                     Temp(1:size(DistTemp,1),1:size(DistTemp,2))= DistTemp;
%                     PDist = vertcat(PDist,Temp); 
                    %-------------------------------------
                    
                    vNumCount = vNumCount+vNum;
                    ccount = ccount+1;
                end
            end 
            vNumAll(count) = vNumCount/ccount;
            count = count+1;
         end        
    end
    
    for j=1:nDef
        %         j=10
        S(j) = std(Dist(~isnan(Dist(:,j)),j))...
            /sqrt(sum(~isnan(Dist(:,j))));
        RS(j) = std(RDist(~isnan(RDist(:,j)),j))...
            /sqrt(sum(~isnan(RDist(:,j))));
%         PS(j) = std(PDist(~isnan(PDist(:,j)),j))...
%             /sqrt(sum(~isnan(PDist(:,j))));        
    end
    M = nanmedian(Dist);%mode(round(Dist));     
    DistMode(:,2*count-1) = M'/M(2);
    DistMode(:,2*count) = S'/M(2);
    
    RM = nanmedian(RDist);%mode(round(RDist));     
    RDistMode(:,2*count-1) = RM'/RM(2);
    RDistMode(:,2*count) = RS'/RM(2);
    
%     PM = nanmedian(PDist);%mode(round(PDist));     
%     PDistMode(:,2*count-1) = PM';%/PM(2);
%     PDistMode(:,2*count) = PS';%/PM(2);    
%     
%     errorbar((1:nDef)'-1,DistMode(:,2*count-1),DistMode(:,2*count)); hold on
figure(ww)
    shadedErrorBar((1:nDef)'-1,DistMode(:,2*count-1)...
        ,DistMode(:,2*count),'lineprops','-ob','transparent',1);hold on
    shadedErrorBar((1:nDef)'-1,RDistMode(:,2*count-1)...
        ,RDistMode(:,2*count),'lineprops','-or','transparent',1);    
%     shadedErrorBar((1:nDef)'-1,PDistMode(:,2*count-1)...
%         ,PDistMode(:,2*count),'lineprops','-og','transparent',1);     
    count=count+1;
end
legend(['defect: ',num2str(mean(vNumAll),3),'|',num2str(std(vNumAll),3)])% axis equal 
% % hold off
%%
vNum = 1e4;
Rpos = rand(vNum,2);
Rpos = [rand(vNum,1),.01*rand(vNum,1)];
[~,RDist] = knnsearch(Rpos,Rpos,'k',50,'distance','euclidean');
    RM = nanmean(RDist);%mode(round(RDist));     
    RDistMode(:,1) = RM';%/M(2);
plot((1:size(RDistMode,1))-1,RDistMode(:,1)/RDistMode(2,1))
  hold on 
%% Quasi-Random number sequence
nDef = 50;
colorS = jet(12);
count  = 1;
for n=10:5:60 % scan different defect number
    nDef = 50;
    sW = 200/1500;
    NumberOfPoints = n*1/sW;
    RDist = [];
    for i=1:200
%         p = haltonset(2,'Skip',1e3*i,'Leap',1e2);
%         p = scramble(p,'RR2');
%         X = net(p,floor(NumberOfPoints)); % same as X = p(1:NumberOfPoints,:);
        X = 1500*rand(round(NumberOfPoints),2); % TO TEST RANDOM SET
        X0 = X(X(:,1)<sW*1500,:);        
        
        [~,DistTemp] = knnsearch(X0,X0,'k',nDef,'distance','euclidean');
        Temp = NaN(size(DistTemp,1),nDef);
        Temp(1:size(DistTemp,1),1:size(DistTemp,2))= DistTemp;
        RDist = vertcat(RDist,Temp);
    end
    RM = nanmedian(RDist); %;%mode(round(RDist));
    maP(count,:) = RM/RM(2);
    %     figure(1);errorbar((1:size(RM,2))-1,RM/RM(2),nanstd(RDist),'o');hold on
        figure(4); pp=plot((1:size(RM,2))-1,RM,'color',colorS(count,:)); hold on%/RM(2)
        pp.LineWidth=2;
    count = count+1;
end
y = 10:5:60; x = 1:50;
[XX,YY] = meshgrid(x,y);
figure(round(sW*1500));surf(XX,YY,maP); shading interp; view(2); colormap jet; caxis([0 20]);
set(gca,'Fontsize',18);axis tight
ylabel('Number of defects'); xlabel('Neighbor');
%% Quasi-Random number sequence
i=79;
sW = 200/1500;
NumberOfPoints = 30*1/sW;
% rng shuffle  % For reproducibility
p = haltonset(2,'Skip',1e3*i,'Leap',1e2);
% p = scramble(p,'RR2');
X = net(p,NumberOfPoints); % same as X = p(1:NumberOfPoints,:);
% X0 = [X(:,2),.2+zeros(size(X))];
% X = rand(round(NumberOfPoints),2); % TEST RANDOM SET 
X0 = X(X(:,1)<sW,:);

[~,RDist] = knnsearch(X0,X0,'k',50,'distance','euclidean');
  RM = nanmedian(RDist); %;%mode(round(RDist));  
% %   %%
%   figure(12);histogram(RDist(:,2),30 ,'Normalization','pdf');hold on
  figure(13);plot((1:size(RM,2))-1,RM/RM(2),'o-');hold on
% % plot(X0(:,1),X0(:,2),'o-')
figure(14);scatter(X0(:,1),X0(:,2),15,'filled');hold on
axis([0 1 0 1]);
% title('{\bf Quasi-Random Scatter}');axis equal; 
%% Simulation of randomly positioned finate discs
% expect to have an order at high density
nDef = 50; colorS = jet(12);
nDset = 10;%round(mean(vNumAll)); % number of defects per stripe
vortRadi = 50;
sW = 200;%ww;
iMax = 1e2;
mask0 = zeros(2*vortRadi+1,2*vortRadi+1);
[XX,YY] = meshgrid(-vortRadi:vortRadi,-vortRadi:vortRadi);
mask0 = sqrt(XX.^2+YY.^2) < vortRadi;
cn = 1;
for nDset=10:5:60
posArr = cell(iMax,1);
for i=1:iMax
    Frame = zeros(1500+2*vortRadi,sW+2*vortRadi);
    count=1;
    n=nDset;
    while n>0
        p0 = ceil(rand(1,2).*[1500,sW]);
        if sum(sum(Frame(p0(1):p0(1)+2*vortRadi,p0(2):p0(2)+2*vortRadi)))==0           
            Frame(p0(1):p0(1)+2*vortRadi,p0(2):p0(2)+2*vortRadi) = mask0;
            posArr{i}(count,:) = p0;
            count=count+1;
            n=n-1;
        end
    end
%     subplot(1,iMax,i)
%     Stripe = Frame(vortRadi:end-vortRadi,vortRadi:end-vortRadi);
% imshow(Stripe)
end
% %% GET distance to neighbor
nDef = 50;
RDist =[];
for i=1:size(posArr,1)
    [~,DistTemp] = knnsearch(posArr{i},posArr{i},'k',nDef,'distance','euclidean');
    Temp = NaN(size(DistTemp,1),nDef);
    Temp(1:size(DistTemp,1),1:size(DistTemp,2))= DistTemp;
    RDist = vertcat(RDist,Temp);
end
RM = nanmedian(RDist); %;%mode(round(RDist));
for j=1:nDef
    RS(j) = nanstd(RDist(:,j))...
        /sqrt(sum(~isnan(RDist(:,j))));
end
figure(sW+1); 
%     shadedErrorBar((1:size(RM,2))-1,RM/RM(2)...
%         ,RS/RM(2),'lineprops','-og','transparent',1); 
pp=plot((1:size(RM,2))-1,RM,'color',colorS(cn,:)); hold on %/RM(2)
pp.LineWidth=2;
cn = cn+1;
end
set(gca,'Fontsize',18);axis tight
ylabel('Normalized Distance'); xlabel('Neighbor');




















