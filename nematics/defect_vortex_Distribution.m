%%
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')
%%
clear var DistAll 
kN = 4;
s=1;
for i=1:size(defPOS,1)
    stripe = indByWidth(i);
    Stripe(i,1) = width(stripe);
    for time=1:3%size(defPOS{stripe},1)
        %         [stripe, time]
        %         [Idx, Dist] = knnsearch(defPOS{stripe}{time},defPOS{stripe}{time},'k',2,'distance','euclidean');
        [Idx,Dist] = knnsearch(defPOS{stripe}{time},defPOS{stripe}{time},'k',5,'distance','euclidean');        
        if size(Dist,2)>2
            DistAll{1}{stripe,time}(:,1) = Dist(:,2);
%             DistAll{2}{stripe,time}(:,1) = Dist(:,3);
%             DistAll{3}{stripe,time}(:,1) = Dist(:,4);
%             DistAll{4}{stripe,time}(:,1) = Dist(:,5);
            %-------- random reference------------
            clear var rXY
            defNum =  size(defPOS{stripe}{time},1);
            rXY(:,1) = Orient_width(stripe)*rand(defNum,1);
            rXY(:,2) = (Orient_area(stripe)/Orient_width(stripe))*rand(defNum,1);
            [rIdx,rDist] = knnsearch(rXY,rXY,'k',5,'distance','euclidean');
            rDistAll{1}{stripe,time}(:,1) = rDist(:,2);
%             rDistAll{2}{stripe,time}(:,1) = rDist(:,3);
%             rDistAll{3}{stripe,time}(:,1) = rDist(:,4);
%             rDistAll{4}{stripe,time}(:,1) = rDist(:,5);
        else if size(Dist,2)>2
        end
        
    end
    i
end
%%
for stripe=1:size(vortPOS,1)
    for time=1:size(vortPOS{stripe},1)
        %         [stripe, time]
        %         [Idx, Dist] = knnsearch(defPOS{stripe}{time},defPOS{stripe}{time},'k',2,'distance','euclidean');
        [vIdx,vDist] = knnsearch(vortPOS{stripe}{time},vortPOS{stripe}{time},'k',5,'distance','euclidean');
        if ~(size(vDist,2)<5)
            vortDistAll1{stripe,time}(:,1) = vDist(:,2);
            vortDistAll2{stripe,time}(:,1) = vDist(:,3);
            vortDistAll3{stripe,time}(:,1) = vDist(:,4);
            vortDistAll4{stripe,time}(:,1) = vDist(:,5);
            %-------- random reference------------
            clear var vrXY
            vortNum =  size(vortPOS{stripe}{time},1);
            vrXY(:,1) = Orient_width(stripe)*rand(vortNum,1);
            vrXY(:,2) = (Orient_area(stripe)/Orient_width(stripe))*rand(vortNum,1);
            [vrIdx,vrDist] = knnsearch(vrXY,vrXY,'k',5,'distance','euclidean');
            vrDistAll1{stripe,time}(:,1) = vrDist(:,2);
            vrDistAll2{stripe,time}(:,1) = vrDist(:,3);
            vrDistAll3{stripe,time}(:,1) = vrDist(:,4);
            vrDistAll4{stripe,time}(:,1) = vrDist(:,5);
        end
    end
    stripe
end

% figure(1);
% plot(defPOS{stripe}{time}(:,1),defPOS{stripe}{time}(:,2),'bx');
% hold on;
% plot(defPOS{stripe}{time}(Idx(:,1),1),defPOS{stripe}{time}(Idx(:,2),2),'ko','MarkerSize',10);
% axis equal; hold off;
% histogram((DistAll{1,1}));hold on
%% ORDERED ARRAY OF BLA_BLA
oN = 1e3;
% orderdArray = [(rand(1,oN))*100*oN; ones(1,oN)];
% orderdArray = [(rand(1,oN))*oN; (rand(1,oN))*oN];
orderdArray = [(0.1 * (randn(1,oN)))+(1:oN); ones(1,oN)];
[oIdx,oDist] =  knnsearch(orderdArray',orderdArray','K',100,'Distance','euclidean');
figure(35); plot(mean(oDist,1)/mean(oDist(:,2)),'o'); hold on
% for i=1:9
% figure(36); histogram(oDist(:,i),30,'Normalization','probability');hold on
% end
% hold off
%%
figure(35); plot(mean(oDist,1)/mean(oDist(:,2)),'o'); hold on
%%
xi = 1:100;
figure(35);hold on
plot(xi,2^.5*xi.^.5,'-');hold off
%% DEFECTS / VORTICIES
clear var sortDistAll RsortDistAll indTemp empties DistAll rDistAll
%------ DEFECTS----------
DistAll = DistAll4;
rDistAll = rDistAll4;
%----- VORTICIES---------
% DistAll = vortDistAll4;
% rDistAll = vrDistAll4;

ww = 100; dw = 20;% choose width and +/- delta
width1 = V_OP_mAng(:,1);
indTemp = find((width1<ww+dw & width1>ww-dw)~=0);
% indTemp = find((width1>300)~=0);% to chech all stripes above 300um
sortDistAll = DistAll(indTemp,:);
RsortDistAll = rDistAll(indTemp,:);
empties = cellfun('isempty',sortDistAll);
edges = 0:5:500;
figure(ww)
histogram(cell2mat(sortDistAll(~empties)),edges,'Normalization','probability');hold on
histogram(cell2mat(RsortDistAll(~empties)),edges,'Normalization','probability');hold on
% set(gca, 'YScale', 'log');
title(['width = ', num2str(ww)])
set(gca,'Fontsize',18);%axis([0 500 1e-5 1])
xlabel('Nearest Defect Distance (\mum)'); ylabel('PDF');


%%
axis([0 500 1e-5 1])

%%

load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation.mat')

stripe =30;time=1;
X = defPOS{stripe}{time}(1:5:end,:);
Y = X;

h = zeros(3,1);
figure(1);
h(1) = plot(X(:,1),X(:,2),'bx');
hold on;
h(2) = plot(Y(:,1),Y(:,2),'rs','MarkerSize',10);
title('Heterogenous Data')

[Idx,D] = knnsearch(X,Y,'k',2,'distance','euclidean');
% idx and D are 4-by-3 matrices.
% 
% idx(j,1) is the row index of the closest observation in X to observation j of Y, and D(j,1) is their distance.
% 
% idx(j,2) is the row index of the next closest observation in X to observation j of Y, and D(j,2) is their distance.
% 
% And so on.
% 
% Identify the nearest observations in the plot.

% for j = 1:size(D,2)
    h(3) = plot(X(Idx(:,1),1),X(Idx(:,2),2),'ko','MarkerSize',10);
% end
legend(h,{'\texttt{X}','\texttt{Y}','Nearest Neighbor'},'Interpreter','latex');
title('Heterogenous Data and Nearest Neighbors')
axis equal; hold off;