%%
PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
% PC_path = 'D:\GD\';                         % RSIP notebook

addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

[dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);


%% SELECT FILES BY SPECIFIC WIDTH
Sw = 600; % selectd width
dw = .07*Sw; % define delta
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);
% Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>1200 & Sorted_Orient_width(:,2)<2100,1);

%% 
px_sizeOP = .74*3;
px_sizePIV = .74;

OP_mid = zeros(size(Range,1),1);
OP_mid_std = OP_mid;
width = OP_mid;
Orient_area = OP_mid;
Orient_width = OP_mid;
Ang_mid = zeros(size(Range,1),1);
Ang_mid_std = Ang_mid;
defNum = cell(size(Range,1),1);
defDensity = cell(size(Range,1),1);
def_NumAv = OP_mid;
def_DensityAv = OP_mid;
PdefPOS = cell(size(Range,1),1);

% kk = 11;kkk=29;
% for i=kk:kk
%  figure
%%
% viD = VideoWriter('C:\Users\victo\Google Drive\Curie\Carles\HT1080 Turbulence Onset\Figures\stripe_pDefects.avi');
% viD.FrameRate = 5;
% open(viD)
i=0;
pp=i+1;
for i= pp:length(Range)
filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
info = imfinfo(filepathOP); % Place path to file inside single quotes
Nn = numel(info);
Orient_area(i) = px_sizeOP^2*info(1).Width*info(1).Height;
Orient_width(i) = px_sizeOP*info(1).Width;


    clearvars ps_x ps_y s_x s_y plocPsi_vec s px py Ang pPsi plocPsi
    % --------------------------ORIENT import ---------------------------------
    disp(['file: ' num2str(i) ' from: ' num2str(size(Range,1))]);    
    Ang = imread(filepathOP,1); % k
    [ll,ww] = size(Ang);
    [Xu,Yu] = meshgrid(1:ww,1:ll);
    OP_frame = zeros(Nn,1);
    mAng_frame = zeros(Nn,1);
    
    qstep = 15;
    for k=1:Nn
        try
        clearvars ps_x ps_y s_x s_y pDefect_Centroid plocPsi_vec
        k
        Ang = imread(filepathOP,k); % k
        if ~any( Ang(:)>2 ) % check if Ang is in RAD
            Ang = Ang * 180/pi;
        end
        %--  FLIP HORISONTALY
%         Ang = fliplr(-Ang);   
%         Ang = rot90(Ang); 
        
        mAng = Ang(:, floor(ww/2)-5:ceil(ww/2)+5);
        mAng(mAng<0) = mAng(mAng<0)+180;
        mAng_frame(k,1) = mean2(mAng);
        mAngVec = reshape(mAng,[size(mAng,1)*size(mAng,2) 1]);
        OP_frame(k,1) = sqrt(sum(sum(cosd(2*mAngVec)))^2 ...
            +sum(sum(sind(2*mAngVec)))^2)/(size(mAng,1)*size(mAng,2));
        %  ---------Defect counter--------
        qq = ordermatrixglissant_overlap(Ang*pi/180,qstep,3);
%         figure(1001);imagesc(qq); axis equal;axis tight; view([-90 90]);hold on
        im2 = qq < min(qq(:))+0.3;%.2; % make binary image to use regionprops
%         figure(1001);imagesc(im2); axis equal;axis tight; view([-90 90])
        s = regionprops('table', im2,'centroid');
% -------------------------------------------------------
        r_Circ = 10; dEdge = 60;
%         s_x = s.Centroid(s.Centroid(:,1)<dEdge,1);% select defects at the edge only
%         s_y = s.Centroid(s.Centroid(:,1)<dEdge,2);
        s_x = s.Centroid(:,1);
        s_y = s.Centroid(:,2);
        s_Circ = zeros(size(Ang));
        TEMP = zeros(2*r_Circ+1,2*r_Circ+1,10);
        Blank = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]);
        
        [XX1,YY1] = meshgrid(-r_Circ:r_Circ,-r_Circ:r_Circ);
        AngTEMP_vec = zeros(10,1);
        
        for j=1:10
            TEMP(:,:,j) = sqrt(XX1.^2+YY1.^2) < r_Circ ...
                & atan2(YY1, XX1)>= (j-1)*pi/5-pi ...
                & atan2(YY1, XX1)< j*pi/5-pi;
        end
%         %%
% %         imagesc(TEMP(:,:,3)); axis equal;axis tight;% hold on
%         %%
%         Ang(Ang <= 0) = 180 + Ang(Ang <= 0);
        pcount = 1;
        ncount = 1;
        
        for ii=1:length(s_x)-1
            for j=1:10
                Blank = zeros(size(Blank));
                Blank(round(s_y(ii)):round(s_y(ii))+2*r_Circ,...
                    round(s_x(ii)):round(s_x(ii))+2*r_Circ) = TEMP(:,:,j);
                sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
                
                AngTEMP  = Ang.* sBlank;
                AngTEMP(AngTEMP==0) = NaN;
        %         AngTEMP_vec1(j,1)  = nanmean(nanmean(180/pi*AngTEMP));

                TF = ~isnan(AngTEMP);
                AngTEMP_vec(j,1)  = 180/pi*circ_mean(AngTEMP(TF));                 
            end
            
%             figure(k+1);imagesc(AngTEMP); axis equal;axis tight; view([-90 90]); hold on
            
            % +/- 1/2 defect characterization
            pos_neg = (AngTEMP_vec(2:end)- AngTEMP_vec(1:end-1))>0;
            if sum(pos_neg)<= 4
                pDefect_Centroid(pcount,:) = [s_x(ii),s_y(ii)];
                pcount = pcount+1;
            elseif sum(pos_neg)>= 5
                nDefect_Centroid(ncount,:) = [s_x(ii),s_y(ii)];
                ncount = ncount+1;
            end
        end
        exist pDefect_Centroid
        % +1/2 defect angle------------------------------------------------------
        px = cosd(Ang);
        py = -sind(Ang);
        Qxx = (px.*px - 1/2);
        Qxy = (px.*py);
        Qyx = (py.*px);
        Qyy = (py.*py - 1/2);
        
        [dxQxx,~] = gradient(Qxx);
        [dxQxy,dyQxy] = gradient(Qxy);
        [~,dyQyy] = gradient(Qyy);
        pPsi = atan2((dxQxy+dyQyy),(dxQxx+dyQxy));
        
        ps_x = pDefect_Centroid(:,1);
        ps_y = pDefect_Centroid(:,2);
%         s_Circ = zeros(size(Ang));
%         TEMP = zeros(2*r_Circ+1,2*r_Circ+1);
        Blank1 = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]);
        
        [XX1,YY1] = meshgrid(-r_Circ:r_Circ,-r_Circ:r_Circ);
        TEMP1(:,:) = sqrt(XX1.^2+YY1.^2) < r_Circ;
        
%         for ii=1:length(ps_x)
%             Blank1 = zeros(size(Blank1));
%             Blank1(round(ps_y(ii)):round(ps_y(ii))+2*r_Circ,...
%                 round(ps_x(ii)):round(ps_x(ii))+2*r_Circ) = TEMP1;
%             sBlank1 = Blank1(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
%             plocPsi  = pPsi.* sBlank1;
%             %         imagesc(sBlank)
%             %         pause(.2)
%             plocPsi(plocPsi==0) = NaN;
% %             plocPsi_vec(ii,1)  = nanmean(nanmean(180/pi*plocPsi));
%             TF = ~isnan(plocPsi);
%             plocPsi_vec(ii,1)  = 180/pi*circ_mean(plocPsi(TF));
%         end

        for ii=1:length(ps_x)
            Blank1 = zeros(size(Blank1));
            Blank1(round(ps_y(ii)):round(ps_y(ii))+2*r_Circ,...
                round(ps_x(ii)):round(ps_x(ii))+2*r_Circ) = TEMP1;
            sBlank1 = Blank1(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
            plocPsi  = pPsi.* sBlank1;
            plocPsi(plocPsi==0) = NaN;

            loc_dxQxx = dxQxx.* sBlank1;
            loc_dxQxy = dxQxy.* sBlank1;
            loc_dyQxy = dyQxy.* sBlank1;
            loc_dyQyy = dyQyy.* sBlank1;
            plocPsi_vec(ii,1) = 180/pi*atan2(mean(loc_dxQxy(loc_dxQxy~=0))+mean(loc_dyQyy(loc_dyQyy~=0)),...
                mean(loc_dxQxx(loc_dxQxx~=0))+ mean(loc_dyQxy(loc_dyQxy~=0)));


        %     TF = ~isnan(plocPsi);
        %     plocPsi_vec(ii,1)  = 180/pi*circ_mean(plocPsi(TF));
        end
% -----------------------------------------------------------
        
        defNum{i}(k,1) = size(s,1);
        defDensity{i}(k,1) = size(s,1)/Orient_area(i);
        if ~isempty(s)
            defNum{i}(k,1) = size(s,1);
            defDensity{i}(k,1) = size(s,1)/Orient_area(i);
            PdefPOS{i}{k,1}(:,:) = [px_sizeOP*ps_x,px_sizeOP*ps_y, plocPsi_vec];%size(s,1);
            PdefPOS_stripe{k,1} = [ps_x, ps_y, plocPsi_vec];
        end
        %  ---------Defect counter--------
        
        %------SHOW-----------
 
%%
% figure;% 
%         %-----LIC------------
% [W,L] = size(Ang); % Width Length
% M = randn(W,L);
% sigma = 5;
% v = zeros(W,L,2);
% v(:,:,2) = cosd(Ang);% NOTE THAT COS GOES IN TO SECOND LAYER
% v(:,:,1) = -sind(Ang);% NOTE THAT -SIN GOES IN TO SECOND LAYER
% options.bound = 'sym'; % boundary handling
% v = perform_blurring(v, sigma, options);
% v = perform_vf_normalization(v);
% options.histogram = 'linear';
% options.verb = 0;
% options.dt = 1; % time steping
% options.flow_correction = 5;
% options.niter_lic = 2; % several iterations gives better results
%     options.M0 = M;
%     Mlist = perform_lic(v, 4, options);
% imageplot(Mlist,''); 
%       hold on
% %         -----END LIC------------
% step = 5;
% q6 = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
%     cosd(Ang(1:step:end,1:step:end)),-sind(Ang(1:step:end,1:step:end)),.7);
% q6.LineWidth=1;
% q6.Color = [.8 .8 .8];
% q6.ShowArrowHead='off';
%  axis equal; axis tight; hold on
% l_len = .08;
% % ALL defects
% p2 = plot(s_x,s_y,'o','MarkerFaceColor',[0 .7 .3]);hold on
% p2.MarkerSize = 10;
% p2.MarkerEdgeColor= [1 1 1];
% % +1/2 defect
% % % --- selected plot
% % ss = 30:34;
% % ps_x1 = ps_x(ss); ps_y1 = ps_y(ss); plocPsi_vec1= plocPsi_vec(ss);
% % p3 = plot(ps_x1,ps_y1,'o','MarkerFaceColor',[.8 .1 0]);hold on
% % q3 = quiver(ps_x1,ps_y1,cosd(plocPsi_vec1),sind(plocPsi_vec1),l_len);hold on
% % %--------------
% p3 = plot(ps_x(cosd(plocPsi_vec)>0),ps_y(cosd(plocPsi_vec)>0),'o','MarkerFaceColor',[.8 .1 0]);hold on
% p3.MarkerSize = 8;
% p3.MarkerEdgeColor = 'white';
% p4 = plot(ps_x(cosd(plocPsi_vec)<0),ps_y(cosd(plocPsi_vec)<0),'o','MarkerFaceColor',[0 .1 .9]);hold on
% p4.MarkerSize = 8;
% p4.MarkerEdgeColor = 'white';
% 
% q3 = quiver(ps_x,ps_y,cosd(plocPsi_vec),sind(plocPsi_vec),l_len);hold on
% q3.LineWidth=5;
% q3.Color = [.8 .1 0];
% q3.ShowArrowHead = 'off';
% hold off 
% view([90 -90]);    

%         F = getframe(gcf);
%         writeVideo(viD,F)
%         pause(.2)
%%        
        %----------------------
        catch
            k=k+1;
        end
    end
    def_NumAv(i,1) = mean(defNum{i});
    def_DensityAv(i,1) = mean(defDensity{i});
    width(i,1) = px_sizeOP*ww;
    OP_mid(i,1) = mean(OP_frame);
    OP_mid_std(i,1) = std(OP_frame)/Nn^.5;%
    Ang_mid(i,1) = mean(mAng_frame);
    Ang_mid_std(i,1) = std(mAng_frame)/Nn^.5;%
    
    % -----------------ORIENT END----------------------------------------------
end
% close(viD)
%                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
% %% SAVE DATA
% PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
% PC_path = 'D:\GD\';                         % RSIP notebook
folder_path = 'Curie\DESKTOP\HT1080\';
save_file = [PC_path, folder_path, 'pDefect_angle',num2str(Sw),'.mat'];
save(save_file)

%%
emptyTF = cellfun(@isempty,PdefPOS_stripe); %check for empty cells
s9 = ~emptyTF;
Q = cell2mat(PdefPOS_stripe(s9));
% histogram(Q(:,1),30); 
figure(190)
% histogram(Q(Q(:,1)<50,3),20,'Normalization', 'PDF');  hold on
histogram(Q(:,3),30,'Normalization', 'PDF');  hold off

%% SHOW DEFECT DISTRIBUTION LOAD DATA
PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
% PC_path = 'D:\GD\';                         % RSIP notebook
folder_path = 'Curie\DESKTOP\HT1080\';

set_name = {'pDefect_angle300',...
            'pDefect_angle400',...
            'pDefect_angle500',...
            'pDefect_angle600',...
            'pDefect_angle700',...
            'pDefect_angle800',...
            'pDefect_angle1000'};
color = lines(10);
set_number  = 7;        
load([PC_path, folder_path, set_name{set_number}, '.mat']);
% PdefPOS = PdefPOS_free;

% %%
emptyTF = cellfun(@isempty,PdefPOS); %check for empty cells
fileY = ~emptyTF; %sum(~emptyTF)
PdefPOS_1 = PdefPOS(fileY);
width_1 = width(fileY);
QQ = [];
for j = 1:length(PdefPOS_1)
    emptyTF = cellfun(@isempty,PdefPOS_1{j}); %check for empty cells
    s9 = ~emptyTF; %sum(~emptyTF)
    QQ{j,1} = cell2mat(PdefPOS_1{j}(s9));
    w = max([width_1(j),Sw]);
    if w==0
        w = Sw;
    end
    QQ{j,1}(:,1) = QQ{j,1}(:,1)/w;
end

Q = cell2mat(QQ);
% histogram(Q(:,1),30); 
dw = 60/Sw; % border width
px_sizeOP = .74*3;
max_xpos = max(Q(:,1));

% figure(floor(width(1)));
% histogram(Q(Q(:,1)<dw,3),60,'Normalization', 'PDF');  hold on
% histogram(Q(Q(:,1)>(max_xpos-dw+.1*dw),3),60,'Normalization', 'PDF');  hold on
% histogram(Q(:,3),60,'Normalization', 'PDF');  hold off
% axis([-180 180 0 inf]); set(gca,'Fontsize',18); 
% ylabel('$ PDF $','Interpreter','latex','FontSize',28);
% xlabel('$Angle\ of\ + \frac{1}{2}\ defect\ (deg)$','Interpreter','latex','FontSize',28);
% % 
% figure();
% polarhistogram(Q(Q(:,1)<dw,3)/180*pi,90,'Normalization', 'PDF');  hold on
% polarhistogram(Q(Q(:,1)>(max_xpos-dw+.1*dw),3)/180*pi,90,'Normalization', 'PDF');  hold on
% polarhistogram(Q(:,3)/180*pi,60,'Normalization', 'PDF');  hold off

% %% Q = [x,y,angle]
% show profile of defect angle

Q = sortrows(Q,1);

xpos = round(Q(:,1)*Sw)-Sw/2;
angle  = Q(:,3)*pi/180;
[U_xpos,~,idx]  = unique(10*round(xpos/10));
N = histc(xpos, U_xpos); % repetition number
U_xpos_angle = [U_xpos, accumarray(idx, angle,[],@circ_mean), accumarray(idx,angle,[],@circ_std)];% ./sqrt(N)
U_xpos_OP = [U_xpos, accumarray(idx, angle,[],@circ_r),accumarray(idx, angle,[],@circ_std)./sqrt(N)];% ./sqrt(N)

% U_xpos_OP = [U_xpos_OP(:,1),...
%     mean([U_xpos_OP(:,2),flip(U_xpos_OP(:,2))],2),...
%     mean([U_xpos_OP(:,3),flip(U_xpos_OP(:,3))],2)];

% find out how to get the magnitute output usig @circ_moment in accumarray

% figure(11); 
% errorbar(U_xpos_angle(:,1),...
%     180/pi*U_xpos_angle(:,2),...
%     180/pi*U_xpos_angle(:,3));

% figure(13); 
% plt1 = plot(U_xpos_OP(:,1),...
%     U_xpos_OP(:,2)); hold on
% plt1.LineWidth = 2; 
figure(12);
[llB, bB] = boundedline(U_xpos_OP(:,1),...
    U_xpos_OP(:,2),...
    U_xpos_OP(:,3),'alpha');
llB.LineWidth = 2; 
llB.Color = color(set_number,:);
bB.FaceColor = color(set_number,:);
bB.FaceAlpha = .2;
hold on

set(gca,'Fontsize',18);set(gcf,'color','w');
ylabel('$Order\ parameter$','Interpreter','latex','FontSize',28);
xlabel('$ Width \ (\mu m) $','Interpreter','latex','FontSize',24);
axis square tight; ylim([0 1])

% figure(round(Sw));
% [llB, bB] = boundedline(U_xpos_angle(:,1),...
%     180/pi*U_xpos_angle(:,2),...
%     180/pi*U_xpos_angle(:,3),'alpha');
% llB.LineWidth = 2; 
% llB.Color = [0.8,0.2,0.2];
% bB.FaceColor = [0.8,0.2,0.2];
% bB.FaceAlpha = .2;
% 
% set(gca,'Fontsize',18);
% ylabel('$Angle \ (deg)$','Interpreter','latex','FontSize',28);
% xlabel('$ Width \ (\mu m) $','Interpreter','latex','FontSize',24);
% title(['$ Width  $ ' regexprep(set_name{set_number}, 'pDefect_angle', ' ') ' $(\mu m) $'],'Interpreter','latex','FontSize',24);
% axis square tight

%% Get Order Parameter of defect orientation


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

