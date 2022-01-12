%% DEFECT detection and classification
% Import IMAGE SERIES
i = 69;
folderOP = dirOP(i).folder;
nameOP = dirOP(i).name;
folderPIV = dirPIV(indX(i,2)).folder;
namePIV = dirPIV(indX(i,2)).name;
%%
filepathOP = [folderOP '\' nameOP];
info = imfinfo(filepathOP); % Place path to file inside single quotes
Nn = numel(info);

filepathPIV = [folderPIV '\' namePIV];
load(filepathPIV);

t_shift = 0; % first Orientation frame
%% Check in one frame (k)
clearvars projection deltaTeta

ff = 3;
filt = fspecial('gaussian',ff,ff);
k_count =1;

kk=321;

for k=1:4%Nn-1
    k
    clearvars -except k k_count info Nn filename pathname filterindex A projection t_shift deltaTeta...
     k kStep k_count info Nn filename pathname...
        filterindex pix2mic A projection t_shift deltaTeta rot_u rot_v...
        box s_box start last total_count filt...
        sub_PIV_u sub_PIV_v sub_nx1 sub_ny1 sub_nx2 sub_ny2...
        u v x y filepathOP filepathPIV Sorted_Orient_width Xu Yu dPsi...
        indX dirOP dirPIV
    
    
    step = 15; % in pixels
    Ang = imread(filepathOP,k); % k
    [l,w] = size(Ang);
    
    if any( Ang(:)>2 ) % chek if Ang is in RAD
        Ang=Ang*pi/180;
    end
    
    [XX,YY] = meshgrid(1:step:w-step,1:step:l-step);
    
    nx = zeros(size(XX));
    ny = nx;
    
    for i=1:size(nx,1)
        %     ii = step*i;
        for j=1:size(nx,2)
            %         jj = step*j;
            nx(i,j) = cos(Ang(step*i,step*j));
            ny(i,j) = -sin(Ang(step*i,step*j));
        end
    end
    
    q = ordermatrixglissant_overlap(Ang,step,3);
    im2 = q < min(q(:))+0.45;%.2; % make binary image to use regionprops
    s = regionprops('table', im2,'centroid');
    % --------------------------------------------------------------------
    % --------------------------------------------------------------------
    % --------------------------------------------------------------------
    % %% !------- Find +/-  defects --------- in ONE FRAME  -------!
    % --------------------------------------------------------------------
    % --------------------------------------------------------------------
    % --------------------------------------------------------------------
    r_Circ = 10;
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
    
    Ang(Ang <= 0) = pi + Ang(Ang <= 0);
    pcount = 1;
    ncount = 1;
    
    for i=1:length(s_x)
        %     if s_x(1)<r_Circ & s_y(2)<r_Circ
        for j=1:10
            Blank = zeros(size(Blank));
            Blank(round(s_y(i)):round(s_y(i))+2*r_Circ,...
                round(s_x(i)):round(s_x(i))+2*r_Circ) = TEMP(:,:,j);
            sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
            
            AngTEMP  = Ang.* sBlank;
            AngTEMP_vec(j,1)  = mean(mean(180/pi*AngTEMP(AngTEMP~=0)));
            %         s_Circ = s_Circ + TEMP;
        end
        
        % +/- 1/2 defect characterization
        pos_neg = (AngTEMP_vec(2:end)- AngTEMP_vec(1:end-1))>0;
        if sum(pos_neg)< 4
            pDefect_Centroid(pcount,:)= s.Centroid(i,:);
            pcount = pcount+1;
        elseif sum(pos_neg)> 5
            nDefect_Centroid(ncount,:)= s.Centroid(i,:);
            ncount = ncount+1;
        end
        %----------------------------
    end
        
    % %%
    px = cos(Ang);
    py = -sin(Ang);
    Qxx = (px.*px - 1/2);
    Qxy = (px.*py);
    Qyx = (py.*px);
    Qyy = (py.*py - 1/2);
    
    [dxQxx,~] = gradient(Qxx);
    [dxQxy,dyQxy] = gradient(Qxy);
    [~,dyQyy] = gradient(Qyy);
    Psi = atan2((dxQxy+dyQyy),(dxQxx+dyQxy));
    
    s_x = pDefect_Centroid(:,1);
    s_y = pDefect_Centroid(:,2);
    s_Circ = zeros(size(Ang));
    TEMP = zeros(2*r_Circ+1,2*r_Circ+1);
    Blank = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]);
    
    [XX1,YY1] = meshgrid(-r_Circ:r_Circ,-r_Circ:r_Circ);
    TEMP(:,:) = sqrt(XX1.^2+YY1.^2) < r_Circ;
    
    for i=1:length(s_x)
        Blank = zeros(size(Blank));
        Blank(round(s_y(i)):round(s_y(i))+2*r_Circ,...
            round(s_x(i)):round(s_x(i))+2*r_Circ) = TEMP;
        sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
        locPsi  = Psi.* sBlank;
        %         imagesc(sBlank)
        %         pause(.2)
        locPsi_vec(i,1)  = mean(mean(180/pi*locPsi(locPsi~=0)));
    end
    
    % --------------------------PIV import ---------------------------------
    uu = imfilter(u{k}, filt);
    vv = imfilter(v{k}, filt);
    sc = size(Ang,1)/size(uu,1);
    
    [Xorig,Yorig] = meshgrid((1/sc:size(uu,1))*sc,(1/sc:size(uu,2))*sc);
    [Xu,Yu] = meshgrid(1:w,1:l);
    u_interp = interp2(Xorig,Yorig,uu,Xu,Yu,'cubic',0);
    v_interp = interp2(Xorig,Yorig,vv,Xu,Yu,'cubic',0);
    
    r_Circ = 3; % this value could be diffrent from the one used in defect classification
    s_Circ = zeros(size(Ang));
    Blank = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]); %imagesc(TEMP); axis equal
    [XX1,YY1] = meshgrid(-r_Circ:r_Circ,-r_Circ:r_Circ);
    TEMP1(:,:) = sqrt(XX1.^2+YY1.^2) < r_Circ;
    for i=1:length(s_x)
        Blank = zeros(size(Blank));
        Blank(round(s_y(i)):round(s_y(i))+2*r_Circ,...
            round(s_x(i)):round(s_x(i))+2*r_Circ) = TEMP1;
        sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
        loc_u  = u_interp.* sBlank;
        loc_v  = v_interp.* sBlank;        
        loc_u_vec(i,1)= mean2(loc_u(loc_u~=0));
        loc_v_vec(i,1)= mean2(loc_v(loc_v~=0));
    end
    
    projection{k_count} = (loc_u_vec.*cosd(locPsi_vec)+loc_v_vec.*sind(locPsi_vec))...
        ./sqrt(loc_u_vec.^2+loc_v_vec.^2);  % dot product of velocity and defect directions
    
    temp = atan2d(loc_v_vec,loc_u_vec)-locPsi_vec; %figure;histogram(temp,40)
    temp(temp<0) = temp(temp<0)+360;
    deltaTeta{k_count} = temp;    
    
    %     histogram(acosd(cell2mat(projection')),20,'Normalization','pdf'); hold off
    
%     figure(1)
%     histogram(cell2mat(deltaTeta'),20,'Normalization','pdf'); hold on;title('\Delta\Theta');axis tight;
%     figure(2)
%     polarhistogram(cell2mat(deltaTeta')*pi/180,60,'Normalization','pdf');  hold on;title('\Delta\Theta');axis tight;
    
    
    k_count=k_count+1;
end
%%
vy = 1*(rand(1000,1)-.25); vx = 1*(rand(1000,1)-.5);
% vy = 5*ones(100,1); vx = -5*ones(100,1);
temp = atan2d((vy),(vx));mean(temp); polarhistogram(temp*pi/180,60,'Normalization','pdf')%figure;
%% Save deltaTeta file  NEEDD TO BE CORRECTED (saves just one frame)
exp_name = ['C:\Users\vici\Desktop\Orient_TEST\DATA\',...
    filename,'_Velocity_vec_pDefect_', num2str(k-k_count),'_',num2str(k),'.mat'];
save(exp_name, 'deltaTeta');
%% Save DEFECT positions file
exp_name = ['C:\Users\vici\Desktop\Orient_TEST\DATA\',...
    filename,'_DefectPos_pOrient_', num2str(k-k_count),'_',num2str(k),'.mat'];
all_Defect = [s.Centroid(:,1),s.Centroid(:,2)];
save(exp_name, 'all_Defect','pDefect_Centroid','nDefect_Centroid','locPsi_vec','loc_u_vec','loc_v_vec' );
%%
figure(34)
subplot(1,3,1);  hold on
histogram(acosd(cell2mat(projection')),20,'Normalization','pdf'); hold off; title('projection');axis tight;
subplot(1,3,2);  hold on
histogram(cell2mat(deltaTeta'),20,'Normalization','pdf'); hold off;title('\Delta\Theta');axis tight;
subplot(1,3,3);  %hold on
polarhistogram(cell2mat(deltaTeta')*pi/180,60,'Normalization','pdf');  hold off;title('\Delta\Theta');axis tight;
%%

%% PLOT MEAN ANGLE FROM DAT FILE
for i=1:size(deltaTeta,2)
    plot((i+100)/12,mean(deltaTeta{1,i}(:)),'o');hold on
end
hold off
%% PLOT MEAN ANGLE DEFECT-VELOCITY DAT FILE
for i=1:size(loc_v_vec,1)
    temp = atan2d(loc_v_vec,loc_u_vec)-locPsi_vec;
    temp(temp<0) = temp(temp<0)+360;
%     deltaTeta{k_count} = temp; 

    plot(i/12,mean(temp),'o');hold on
end
hold off
%% Save deltaTeta file
save(['C:\Users\vici\Desktop\',...
    filename,'_Velocity_vec_pDefect_', num2str(k-k_count),'_',num2str(k),'.mat'], deltaTeta);
%% Load and plot
[lfile, lpathname] = uigetfile( ...
    '*.mat', 'Pick a PIV file',...
    'C:\Users\vici\Desktop\Orient_TEST\DATA');
load([lpathname,lfile],'deltaTeta');
subplot(1,2,1);
histogram(cell2mat(deltaTeta'),20,'Normalization','pdf'); hold off;title('\Delta\Theta');axis tight;
subplot(1,2,2);
polarhistogram(cell2mat(deltaTeta')*pi/180,60,'Normalization','pdf');  hold off;title('\Delta\Theta');axis tight;
%%
[dTetaCounts, dTetaAngles] = histcounts(cell2mat(deltaTeta'),60,'Normalization','pdf'); 
figure(1)
plot((dTetaAngles(1:end-1)+dTetaAngles(2:end))/2, dTetaCounts)
Angles = (dTetaAngles(1:end-1)+dTetaAngles(2:end))/2;
EXPO = [Angles', dTetaCounts'];
% findpeack(cell2mat(deltaTeta'))
%% PLOT DEFECTS
%
figure(k_count)

p1=plot(s.Centroid(:,1),s.Centroid(:,2),'o', 'MarkerFaceColor',[0 0 1]);hold on
p1.MarkerSize = 10;
p1.MarkerEdgeColor= 'none';

p2=plot(pDefect_Centroid(:,1),pDefect_Centroid(:,2),'o','MarkerFaceColor',[0 .5 .1]);
p2.MarkerSize = 10;
p2.MarkerEdgeColor= 'none';

p3=plot(nDefect_Centroid(:,1),nDefect_Centroid(:,2),'o','MarkerFaceColor',[1 0 0]);
p3.MarkerSize = 10;
p3.MarkerEdgeColor= 'none';

step = 5;
q1 = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
    cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)),.7);
q1.LineWidth=1;
q1.Color = [.4 .4 .4];
q1.ShowArrowHead='off';

q2 = quiver(s_x,s_y,cosd(locPsi_vec),sind(locPsi_vec),.2);hold on
q2.LineWidth=3;
q2.Color = [0 .5 .1];
q2.ShowArrowHead='off';

% q3 = quiver(s_x,s_y,loc_u_vec,loc_v_vec,.3);hold on
% q3.LineWidth=2;
% q3.Color = [1 0 1];
axis equal; axis tight;title(k);

hold off