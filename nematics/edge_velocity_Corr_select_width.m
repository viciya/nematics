%% correspond PIV and Orientation names
% dirOP = dir(['C:\Users\vici\Google Drive\DATA\HT1080\Orient'  '\*.tif']);
% dirPIV = dir(['C:\Users\vici\Google Drive\DATA\HT1080\PIV_DATs'  '\*.mat']);

dirOP = dir(['C:\Users\victo\Google Drive\DATA\HT1080\Orient'  '\*.tif']);
dirPIV = dir(['C:\Users\victo\Google Drive\DATA\HT1080\PIV_DATs'  '\*.mat']);

for i = 1:size(dirOP,1)
    
    nameOP = dirOP(i).name;
    p1=strfind(nameOP,'s');
    p2=strfind(nameOP,'.tif');
    patternOP = nameOP(p1:p2);
    
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

%% sort files in directrory by width
px_sizeOP = .74*3;
for i=1:size(dirOP,1)
    % --------------------------ORIENT import ---------------------------------
    filepathOP = [dirOP(indX(i,1)).folder '\' dirOP(indX(i,1)).name];
    info = imfinfo(filepathOP); % Place path to file inside single quotes
    Nn = numel(info);
    Orient_width(i,1) = px_sizeOP*info(1).Width;
end
[~,idx] = sort(Orient_width);
Sorted_Orient_width = [idx,Orient_width(idx)];

%% SELECT FILES BY SPECIFIC WIDTH
Sw = 600; % selectd width
dw = .1*Sw; % define delta
RangeOP = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);
RangePIV = indX(RangeOP,2);
% Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>450 & Sorted_Orient_width(:,2)<1100,1);
% i = 29;
%     filepathPIV = [folderPIV '\' dirPIV(RangePIV(i)).name];
%     load(filepathPIV);
% for k=1:20
% ff = 7; filt = fspecial('gaussian',ff,ff);
% vv = imfilter(v{k}, filt);
% % figure(10)
% % imagesc(vv/norm(vv)); axis equal; axis tight
% Y = fft2(vv);
% Y1 = (abs(fftshift(Y)));
% % figure(12)
% % imagesc(Y1);axis equal; axis tight
% 
% mmH = ceil(size(Y1,1)/2);
% mmV = ceil(size(Y1,2)/2);
% xaxH = (1:size(Y1,1))-1-mmH;
% xaxV = (1:size(Y1,2))-1-mmV;
% figure(13)
% plot(xaxV,Y1(mmH,:)/norm(Y1(mmH,:)),'g'); hold on
% figure(14)
% plot(xaxH,Y1(:,mmV+1)/norm(Y1(:,mmV+1)),'g');hold on
% end
%%
folderPIV = 'C:\Users\victo\Google Drive\DATA\HT1080\PIV_DATs';

px_sizeOP = .74*3;
px_sizePIV = .74;
frame_per_hr = 4;
px2mic = px_sizeOP * frame_per_hr;
ff = 3; filt = fspecial('gaussian',ff,ff);


for i=1%:length(RangePIV)
    filepathPIV = [folderPIV '\' dirPIV(RangePIV(i)).name];
    load(filepathPIV);
    [L,W] = size(u{1,1});
    T = size(u,1);
    widthS(i,1) = x{1}(1,end)*px_sizePIV;
    clearvars u_t v_t
    dx = px_sizePIV*(x{1}(1,2)-x{1}(1,1));
    for k = 1:size(x,1)
        vv = imfilter(u{k}, filt); 
        v_R = mean(vv(:,end),2);%plot(v_R,'.-r'); hold on
        v_L = mean(vv(:,1:2),2); %plot(v_L,'.-b'); hold on
        Corr = xcorr(v_R-mean(v_R));
        mm = ceil(length(Corr)/2);
        CorrP = mean([Corr(mm:end), flip(Corr(1:mm))],2);

        CorrP = CorrP/(max(CorrP));
        CorrPP(:,k) = CorrP;
        xax = dx*((1:50)-1)';
        plot(xax,CorrP(1:50),'Color',[.6 .1 .1 .2]); hold on
%         GET AVERAGE VALUE OF THE SLOPEBY LINEAR FIT
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    end
    
%     CorrPPm = mean(CorrPP,2);
%     xax = dx*((1:length(CorrPPm))-1)';
%     plot(xax,CorrPPm); hold on
end
set(gca,'Fontsize',18);
ylabel('$Correlation\ functions\ \ v_y(x=\pm L/2) $','Interpreter','latex','FontSize',24);
xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',28); axis tight
%%

%%
x=zeros(200,100);
x(98:102,45:55)=1;

X=fft2(x);
y=zeros(200,200);
y(20:180,100)=1;
Y=fft2(y);
figure(1);imshow(x);
figure(2);imshow(y);
figure(3);imshow(fftshift(log(abs(X)+1)),[]);
figure(4);imshow(fftshift(log(abs(Y)+1)),[]);
%% SAVE DATA
% save 'C:\Users\vici\Google Drive\Curie\DESKTOP\HT1080\pDefect_angle.mat'

%%
emptyTF = cellfun(@isempty,PdefPOS_stripe); %check for empty cells
s9 = ~emptyTF;
Q = cell2mat(PdefPOS_stripe(s9));
% histogram(Q(:,1),30); 
% histogram(Q(Q(:,1)<50,3),20,'Normalization', 'PDF');  hold on
histogram(-Q(:,3),'Normalization', 'PDF');  hold off
%% LOAD DATA
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\pDefect_angle.mat')

%%
emptyTF = cellfun(@isempty,PdefPOS); %check for empty cells
fileY = ~emptyTF; %sum(~emptyTF)
PdefPOS_1 = PdefPOS(fileY);
for j = 1:length(PdefPOS_1)
    emptyTF = cellfun(@isempty,PdefPOS_1{j}); %check for empty cells
    s9 = ~emptyTF; %sum(~emptyTF)
    QQ{j,1} = cell2mat(PdefPOS_1{j}(s9));
end

Q = cell2mat(QQ);
% histogram(Q(:,1),30); 
figure(1346);
histogram(Q(Q(:,1)<50,3),60,'Normalization', 'PDF');  hold on
histogram(Q(:,3),60,'Normalization', 'PDF');  hold off

figure(1345);
polarhistogram(Q(Q(:,1)<50,3)/180*pi,'Normalization', 'PDF');  hold on
polarhistogram(Q(:,3)/180*pi,60,'Normalization', 'PDF');  hold off


%% SELECT BY WIDTH
load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new3.mat')
[widthSorted1, indByWidth] = sortrows(width,1);
%
Sw = 400; %/ px_sizeOP; RUN FOR Sw=190,Dw = 0.1  
Dw = 0.1 * Sw;
xax = (0:Sw)';
% Right tilt  (V_OP_mAng(:,2)>100 & V_OP_mAng(:,3)> 10)
% sRange = indByWidth(widthSorted1(:,1)>Sw-Dw & widthSorted1(:,1)<Sw+Dw);
sRange = indByWidth(Sorted_Orient_width(:,2)>Sw & Sorted_Orient_width(:,2)<1400);

clearvars PdefPOS_S PdefPOS_1 Q QQ s9
for j=1:length(sRange)
PdefPOS_S{j,1} = PdefPOS{sRange};
end
emptyTF = cellfun(@isempty,PdefPOS_S); %check for empty cells
fileY = ~emptyTF; %sum(~emptyTF)
PdefPOS_1 = PdefPOS_S(fileY);
for j = 1:length(PdefPOS_1)
    emptyTF = cellfun(@isempty,PdefPOS_1{j}); %check for empty cells
    s9 = ~emptyTF; %sum(~emptyTF)
    QQ{j,1} = cell2mat(PdefPOS_1{j}(s9));
end
%
Q = cell2mat(QQ);
% Q(Q(:,3)<0,3) = Q(Q(:,3)<0,3)+360;
% histogram(Q(:,1),30); 
%
figure(Sw);
edgeQ = Q(Q(:,1)<54,3);
centerQ = Q(Q(:,1)>100,3);
% h2 = histogram(centerQ,'Normalization', 'PDF'); hold on
% h2.EdgeAlpha = 0; h2.FaceColor = [1 0 0]; 
[Qn,binQ] = histcounts(centerQ,17,'Normalization', 'PDF');
plot(mean([binQ(2:end);binQ(1:end-1)])',Qn','r','LineWidth',2); hold on

% h1= histogram(edgeQ,9,'Normalization', 'PDF');  hold on
% h1.EdgeAlpha = 0; h1.FaceColor = [.1 .3 .8];
[Qn,binQ] = histcounts(edgeQ,17,'Normalization', 'PDF');  
plot(mean([binQ(2:end);binQ(1:end-1)])',Qn','b','LineWidth',2);hold on

hold off
axis([-180 180 0 inf]); title(length(sRange)); set(gca,'Fontsize',18); 
ylabel('$ PDF $','Interpreter','latex','FontSize',28);
xlabel('$Angle\ of\ + \frac{1}{2}\ defect\ (deg)$','Interpreter','latex','FontSize',28);

%
figure(Sw+1);
polarhistogram(edgeQ/180*pi,30,'Normalization', 'PDF');  hold on
polarhistogram(centerQ/180*pi,30,'Normalization', 'PDF');  hold off

