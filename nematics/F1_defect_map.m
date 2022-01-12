% DEFECT detection and classification
% RUN first cell of "correlate_OP_PIV_Defect.mat" to get missing variables
i = 88;
folderOP = dirOP(i).folder;
nameOP = dirOP(i).name;
folderPIV = dirPIV(indX(i,2)).folder;
namePIV = dirPIV(indX(i,2)).name;
% %%
filepathOP = [folderOP '\' nameOP];
info = imfinfo(filepathOP); % Place path to file inside single quotes
Nn = numel(info);

% filepathPIV = [folderPIV '\' namePIV];
% load(filepathPIV);

% %% Check in one frame (k)

ff = 3;
filt = fspecial('gaussian',ff,ff);

pix2mic = .74;
k_count =1;
total_count = 0;

box = 100;
s_box = floor(sqrt(box^2/2));disp(['box width in um: ',num2str(s_box* pix2mic *2,3)]);
sub_PIV_u = zeros(2*s_box+1);
sub_PIV_v =sub_PIV_u;
sub_nx1 = sub_PIV_u;
sub_ny1 = sub_nx1;
sub_nx2 = sub_nx1;
sub_ny2 = sub_nx1;

% for k=1:Nn-dt
start= 1; kStep = 1; last= Nn-1;%+40;
for k=25%start:kStep:last
%     try
    k
    clearvars -except k kStep k_count info Nn filename pathname...
        filterindex pix2mic A projection t_shift deltaTeta rot_u rot_v...
        box s_box start last total_count filt...
        sub_PIV_u sub_PIV_v sub_nx1 sub_ny1 sub_nx2 sub_ny2...
        u v x y filepathOP filepathPIV Sorted_Orient_width Xu Yu dPsi...
        indX dirOP dirPIV
    
    qstep = 15;%6; %mine 10 in pixels
    Ang = imread(filepathOP,k); % k
    [l,w] = size(Ang);
    
    if any( Ang(:)>2 ) % chek if Ang is in RAD
        Ang = Ang*pi/180;
    end
   
    q = ordermatrixglissant_overlap(Ang,qstep,3);
    im2 = q < min(q(:))+.45;%0.6;%.2; % make binary image to use regionprops
    s = regionprops('table', im2,'centroid');
    % --------------------------------------------------------------------
    % %% !------- Find +/-  defects --------- in ONE FRAME  -------!
    % --------------------------------------------------------------------
    r_Circ = 10;
    ps_x = s.Centroid(:,1);
    ps_y = s.Centroid(:,2);
    s_x = ps_x;
    s_y = ps_y;
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
    
    for i=1:length(ps_x)-1
        for j=1:10
            Blank = zeros(size(Blank));
            Blank(round(ps_y(i)):round(ps_y(i))+2*r_Circ,...
                round(ps_x(i)):round(ps_x(i))+2*r_Circ) = TEMP(:,:,j);
            sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
            
            AngTEMP  = Ang.* sBlank;
            AngTEMP(AngTEMP==0) = NaN;
            AngTEMP_vec(j,1)  = nanmean(nanmean(180/pi*AngTEMP));
        end
        
        % +/- 1/2 defect characterization
        pos_neg = (AngTEMP_vec(2:end)- AngTEMP_vec(1:end-1))>0;
        if sum(pos_neg)<= 4
            pDefect_Centroid(pcount,:)=s.Centroid(i,:);
            pcount = pcount+1;
        elseif sum(pos_neg)>= 5
            nDefect_Centroid(ncount,:)=s.Centroid(i,:);
            ncount = ncount+1;
        end
    end
     exist pDefect_Centroid   
% +1/2 defect angle------------------------------------------------------
    px = cos(Ang);
    py = -sin(Ang);
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
    s_Circ = zeros(size(Ang));
    TEMP = zeros(2*r_Circ+1,2*r_Circ+1);
    Blank = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]);
    
    [XX1,YY1] = meshgrid(-r_Circ:r_Circ,-r_Circ:r_Circ);
    TEMP(:,:) = sqrt(XX1.^2+YY1.^2) < r_Circ;
    
    for i=1:length(ps_x)
        Blank = zeros(size(Blank));
        Blank(round(ps_y(i)):round(ps_y(i))+2*r_Circ,...
            round(ps_x(i)):round(ps_x(i))+2*r_Circ) = TEMP;
        sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
        plocPsi  = pPsi.* sBlank;
        %         imagesc(sBlank)
        %         pause(.2)
        plocPsi(plocPsi==0) = NaN;
        plocPsi_vec(i,1)  = nanmean(nanmean(180/pi*plocPsi));
    end
    
% -1/2 defect angle-------------------------------------------------------
    py = sin(Ang); % sin(Ang) will make -1/2 to 1/2
    Qxx = (px.*px - 1/2);
    Qxy = (px.*py);
    Qyx = (py.*px);
    Qyy = (py.*py - 1/2);
    
    [dxQxx,~] = gradient(Qxx);
    [dxQxy,dyQxy] = gradient(Qxy);
    [~,dyQyy] = gradient(Qyy);
    nPsi = atan2((dxQxy+dyQyy),(dxQxx+dyQxy));
    
    ns_x = nDefect_Centroid(:,1);
    ns_y = nDefect_Centroid(:,2);
    
    for i=1:length(ns_x)
        Blank = zeros(size(Blank));
        Blank(round(ns_y(i)):round(ns_y(i))+2*r_Circ,...
            round(ns_x(i)):round(ns_x(i))+2*r_Circ) = TEMP;
        sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
        nlocPsi  = nPsi.* sBlank;
        %         imagesc(sBlank)
        %         pause(.2)
     end
       nlocPsi(plocPsi==0) = NaN;
        nlocPsi_vec(i,1)  = nanmean(nanmean(180/pi*nlocPsi));
    nlocPsi_vec = -nlocPsi_vec/3;
    
    % --------------------------PIV import ---------------------------------    

    % ----- averaging of velocity fields (dt=1: cancels averaging) -----
    [Xu,Yu] = meshgrid(1:w,1:l);
   
end
%% PLOT DEFECTS
%
% tt = ii;
figure(k)
% subplot(1,2,1)
% p1=plot(s.Centroid(:,1),s.Centroid(:,2),'o', 'MarkerFaceColor',[0 0 1]);hold on
% p1.MarkerSize = 10;
% p1.MarkerEdgeColor= 'none';
% figure

l_len = .05;
MarkerS = 8;
step = 8;
O_len = 0.8;
% Orientation

q6 = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
    cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)),O_len);
q6.LineWidth=1;
q6.Color = [.4 .4 .4];
q6.ShowArrowHead='off'; hold on


% ALL defects
p2 = plot(s_x,s_y,'o','MarkerFaceColor',[0 .5 .1]);hold on
p2.MarkerSize = MarkerS;
p2.MarkerEdgeColor= 'none';
% +1/2 defect
p2 = plot(ps_x,ps_y,'o','MarkerFaceColor',[.8 .1 0]);hold on
p2.MarkerSize = MarkerS;
p2.MarkerEdgeColor= 'none';

q2 = quiver(ps_x,ps_y,cosd(plocPsi_vec),sind(plocPsi_vec),l_len);hold on
q2.LineWidth=3;
q2.Color = [.8 .1 0];
q2.ShowArrowHead = 'off';

% -1/2 defect
p3 = plot(ns_x,ns_y,'o','MarkerFaceColor',[.1 .1 .9]);hold on
p3.MarkerSize = MarkerS;
p3.MarkerEdgeColor= 'none';

q3 = quiver(ns_x,ns_y,cosd(nlocPsi_vec),sind(nlocPsi_vec),l_len);hold on
q3.LineWidth=3;
q3.Color = [.1 .1 .9];
q3.ShowArrowHead = 'off';
q4 = quiver(ns_x,ns_y,cosd(nlocPsi_vec+120),sind(nlocPsi_vec+120),l_len);hold on
q4.LineWidth=3;
q4.Color = [.1 .1 .9];
q4.ShowArrowHead = 'off';
q5 = quiver(ns_x,ns_y,cosd(nlocPsi_vec+240),sind(nlocPsi_vec+240),l_len);hold on
q5.LineWidth=3;
q5.Color = [.1 .1 .9];
q5.ShowArrowHead = 'off';

% % Velocity
% vstep = 1;hold on
% q0 = quiver(Xorig(1:vstep:end,1:vstep:end),Yorig(1:vstep:end,1:vstep:end),...
%     uu(1:vstep:end,1:vstep:end),vv(1:vstep:end,1:vstep:end),2);
% q0.LineWidth=.5;
% q0.Color = [0 0 0];
% 
% % Vorticity
% vstep = 1;
% [u_x,u_y] = gradient(uu);%/dx gradient need to be corrected for the dx
% [v_x,v_y] = gradient(vv);%/dx
% vorticity = (v_x - u_y);
% ff = 7;
% filt = fspecial('gaussian',ff,ff);
% u1 = imfilter(vorticity, filt);
% surf(Xorig(1:vstep:end,1:vstep:end),Yorig(1:vstep:end,1:vstep:end),...
%     u1(1:vstep:end,1:vstep:end)-200);

% Divergence
% vstep = 1;%
% divV =  divergence(uu,vv);
% ff = 5;
% filt = fspecial('gaussian',ff,ff);
% u2 = imfilter(divV, filt);
% surf(Xorig(1:vstep:end,1:vstep:end),Yorig(1:vstep:end,1:vstep:end),...
%     u2(1:vstep:end,1:vstep:end)-10);
% 
% load('mycbar.mat')
% set(gcf,'Colormap',mycbar); %this works
% 
% % colormap jet
% view(2);shading interp;
axis equal;axis tight;axis off; hold off
set(gca,'View',[-90 90]);
%

%%
folder = 'C:\Users\victo\Google Drive\Curie\Carles\HT1080 Turbulence Onset\Figures';
file = 'HT1080_10x_15min_2_s331_t86_ORIENT.tif';
Ang = imread([folder,'\',file]);
[l,w] = size(Ang);
[Xu,Yu] = meshgrid(1:w,1:l);
%%
% Orientation
step = 30;
q6 = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
    cosd(Ang(1:step:end,1:step:end)),-sind(Ang(1:step:end,1:step:end)),.7);
q6.LineWidth=1;
q6.Color = [.4 .4 .4];
q6.ShowArrowHead='off';
axis equal; axis tight