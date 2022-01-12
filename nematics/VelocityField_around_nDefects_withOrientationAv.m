%%
% PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
PC_path = 'D:\GD\';                         % RSIP notebook

addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

[dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);
%%
% DEFECT detection and classification
% RUN first cell of "correlate_OP_PIV_Defect.mat" to get missing variables
i = 1;

Sw = 1500; % selectd width
dw = .1*Sw; % define delta
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);

filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];

info = imfinfo(filepathOP); % Place path to file inside single quotes
Nn = numel(info);
load(filepathPIV);

t_shift = 0; % first Orientation frame

% %% Check in one frame (k)
clearvars projection deltaTeta

ff = 3;
filt = fspecial('gaussian',ff,ff);

pix2mic = 3*.74;
k_count =1;
total_count = 0;

box = 70;
s_box = floor(sqrt(box^2/2));
disp(['box width in um: ',num2str(s_box* pix2mic *2,3)]);
sub_PIV_u = zeros(2*s_box+1);
sub_PIV_v =sub_PIV_u;
sub_nx1 = sub_PIV_u;
sub_ny1 = sub_nx1;
sub_nx2 = sub_nx1;
sub_ny2 = sub_nx1;

% for k=1:Nn-dt
start= 1; kStep = 1; last= Nn-1;%+40;
%
% pp = i+1
for k=1:Nn-1%start:kStep:last
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
        Ang=Ang*pi/180;
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
        if sum(pos_neg)< 4
            pDefect_Centroid(pcount,:)=s.Centroid(i,:);
            pcount = pcount+1;
        elseif sum(pos_neg)> 5
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
    
    for i= 1:length(ns_x)
        Blank = zeros(size(Blank));
        Blank(round(ns_y(i)):round(ns_y(i))+2*r_Circ,...
            round(ns_x(i)):round(ns_x(i))+2*r_Circ) = TEMP;
        sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
        nlocPsi  = nPsi.* sBlank;
        %         imagesc(sBlank)
        %         pause(.2)
        nlocPsi(nlocPsi==0) = NaN;
%         nlocPsi_vec(i,1)  = nanmean(nanmean(180/pi*nlocPsi));
        TF = ~isnan(nlocPsi);
        nlocPsi_vec(i,1)  = 180/pi*circ_median(nlocPsi(TF));
    end
    nlocPsi_vec = -nlocPsi_vec/3;
    
    % --------------------------PIV import ---------------------------------    

    % ----- averaging of velocity fields (dt=1: cancels averaging) -----
 
    uu = imfilter(u{k}, filt);
    vv = imfilter(v{k}, filt);
    sc = size(Ang,1)/size(uu,1);
    
    [Xorig,Yorig] = meshgrid((1/sc:size(uu,2))*sc,(1/sc:size(uu,1))*sc);
    [Xu,Yu] = meshgrid(1:w,1:l);
    u_interp = interp2(Xorig,Yorig,uu,Xu,Yu,'cubic',0);
    v_interp = interp2(Xorig,Yorig,vv,Xu,Yu,'cubic',0);
    
% figure(5); 
% % subplot(1,2,1);imagesc(q);axis equal; axis tight;hold on
% % % Orientation
% % 
% %  hold off
% % subplot(1,2,2);
% imagesc(q); axis equal; axis tight;hold on
% p1=plot(s.Centroid(:,1),s.Centroid(:,2),'o', 'MarkerFaceColor',[0 0 0]);
% p1.MarkerSize = 6;
% p1.MarkerEdgeColor= 'none';
% step = 6;
% q1 = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
%     cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)),.7);
% q1.LineWidth=1;
% q1.Color = [.4 .4 .4];
% q1.ShowArrowHead='off';
% axis equal; axis tight; 
% 
% l_len = .15;
% % +1/2 defect
% p2 = plot(ps_x,ps_y,'o','MarkerFaceColor',[0 .5 .1]);hold on
% p2.MarkerSize = 5;
% p2.MarkerEdgeColor= 'none';
% 
% q2 = quiver(ps_x,ps_y,cosd(plocPsi_vec),sind(plocPsi_vec),l_len);hold on
% q2.LineWidth=3;
% q2.Color = [0 .5 .1];
% q2.ShowArrowHead = 'off';
% set(gca,'View',[-90 90])
% hold off
%
%------ replace P with N for negative defects ----------
ps_y = ns_y; ps_x = ns_x;
plocPsi_vec = nlocPsi_vec;
%-------------------------------------------------------
%%
%  Calculate the difference between velocity and depect angle   
    r_Circ = 3; % this value could be diffrent from the one used in defect classification
    s_Circ = zeros(size(Ang));
    Blank = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]); %imagesc(TEMP); axis equal
    [XX1,YY1] = meshgrid(-r_Circ:r_Circ,-r_Circ:r_Circ);
    TEMP1(:,:) = sqrt(XX1.^2+YY1.^2) < r_Circ;
    for i=1:length(ps_x)
        Blank = zeros(size(Blank));
        Blank(round(ps_y(i)):round(ps_y(i))+2*r_Circ,...
            round(ps_x(i)):round(ps_x(i))+2*r_Circ) = TEMP1;
        sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
        loc_u  = u_interp.* sBlank;
        loc_v  = v_interp.* sBlank;
        loc_u(loc_u==0) = NaN;
        loc_v(loc_v==0) = NaN;
        loc_u_vec(i,1)= nanmean(nanmean(loc_u)); %figure(1);imagesc(loc_u)
        loc_v_vec(i,1)= nanmean(nanmean(loc_v)); %figure(2);imagesc(loc_v)       
    end  
    projection{k_count} = (loc_u_vec.*cosd(plocPsi_vec)+loc_v_vec.*sind(plocPsi_vec))...
        ./sqrt(loc_u_vec.^2+loc_v_vec.^2);  % dot product of velocity and defect directions
    
    temp = atan2d(loc_v_vec,loc_u_vec)-plocPsi_vec;
    temp(temp<0) = temp(temp<0)+360;
    deltaTeta{k_count,1} = temp;
%%    
% histogram(m(~isnan(m)), 50, 'Normalization','PDF')
% histogram(vel_dir, 50, 'Normalization','PDF'); hold on
% histogram(plocPsi_vec, 50, 'Normalization','PDF'); hold off
% ///////////////   ROTATION ////////////////////////////////////// 
    rot_u = zeros(2*s_box+1);
    rot_v = rot_u;
    rot_nx1 = rot_u;
    rot_ny1 = rot_u;
    rot_nx2 = rot_u;
    rot_ny2 = rot_u;    
%%     
    i_count = 0;
ii=41;   
%     for i=1:length(ps_x)
      parfor i=1:length(ps_x)
% ---  check if box around the defects fits in the PIV field  -----
        if (ps_x(i)-box)>1 && (ps_y(i)-box)>1 && (ps_x(i)+box)<w && (ps_y(i)+box)<l 
%             take PIV sub-window around the defect
            su = u_interp(round(ps_y(i))-box:round(ps_y(i))+box,...
                round(ps_x(i))-box:round(ps_x(i))+box);
            sv = v_interp(round(ps_y(i))-box:round(ps_y(i))+box,...
                round(ps_x(i))-box:round(ps_x(i)+box));
            %             take ORIENT sub-window around the defect
            sAng = Ang(round(ps_y(i))-box:round(ps_y(i))+box,...
                round(ps_x(i))-box:round(ps_x(i))+box);
            snx = cos(sAng);
            sny = -sin(sAng);
%%  PLOT defect subwindow before rotation         
% step = 6; 
% [xl,yl] = size(su);
% [xxx,yyy] = meshgrid(1:xl,1:yl);
% figure(1234); %subplot(1,2,1);
% q1 = quiver(xxx(1:step:end,1:step:end),yyy(1:step:end,1:step:end),...
%     snx(1:step:end,1:step:end),sny(1:step:end,1:step:end),.7);
% q1.LineWidth=1;
% q1.Color = [.4 .4 .4];
% q1.ShowArrowHead='off';
% hold on
% q2 = quiver(xxx(1:step:end,1:step:end),yyy(1:step:end,1:step:end),...
%     su(1:step:end,1:step:end),sv(1:step:end,1:step:end),.9);
% q2.LineWidth=1;
% q2.Color = [1 0 0];
% 
% p2 = plot3(xl/2,yl/2,40,'o','MarkerFaceColor',[0 0 0]);
% p2.MarkerSize = 10;
% p2.MarkerEdgeColor= 'none';
% 
% l_len = 25;
% q3 = quiver(xl/2,yl/2,cosd(plocPsi_vec(i,1)),sind(plocPsi_vec(i,1)),l_len);hold on
% q3.LineWidth=3;
% q3.Color = [0 .5 .1];
% q3.ShowArrowHead = 'off';
% axis equal; axis tight; 
% r1 = double(plocPsi_vec(i,1));
% % set(gca,'View',[r1, 90]); 
% title([deltaTeta{k_count,1}(ii)]);
% hold off
%%        
            
            %             rotate each PIV vector by angle of the defect
            suR = cosd(plocPsi_vec(i,1))*su + sind(plocPsi_vec(i,1))*sv;
            svR = -sind(plocPsi_vec(i,1))*su + cosd(plocPsi_vec(i,1))*sv;
            %             rotate each ORIENT vector by angle of the defect
            snxR = cosd(plocPsi_vec(i,1))*snx + sind(plocPsi_vec(i,1))*sny;
            snyR = -sind(plocPsi_vec(i,1))*snx + cosd(plocPsi_vec(i,1))*sny;          

%             rotate whole PIV field by angle of the defect               
            suRR = imrotate(suR,plocPsi_vec(i,1),'bilinear','crop');
            svRR = imrotate(svR,plocPsi_vec(i,1),'bilinear','crop');%,'crop'
%             rotate whole ORIENT field by angle of the defect              
            snxRR = imrotate(snxR,plocPsi_vec(i,1),'bilinear','crop');
            snyRR = imrotate(snyR,plocPsi_vec(i,1),'bilinear','crop');%,'crop'
            
%%  PLOT defect subwindow after rotation         
% step = 6; 
% [xl,yl] = size(suRR);
% [xxx,yyy] = meshgrid(1:xl,1:yl);
% figure(1234); subplot(1,2,2);
% q1 = quiver(xxx(1:step:end,1:step:end),yyy(1:step:end,1:step:end),...
%     snxRR(1:step:end,1:step:end),snyRR(1:step:end,1:step:end),.7);
% q1.LineWidth=1;
% q1.Color = [.4 .4 .4];
% q1.ShowArrowHead='off';
% hold on
% q2 = quiver(xxx(1:step:end,1:step:end),yyy(1:step:end,1:step:end),...
%     suRR(1:step:end,1:step:end),svRR(1:step:end,1:step:end),.9);
% q2.LineWidth=1;
% q2.Color = [1 0 0];
% 
% p2 = plot3(xl/2,yl/2,40,'o','MarkerFaceColor',[0 0 0]);
% p2.MarkerSize = 10;
% p2.MarkerEdgeColor= 'none';
% 
% l_len = 25;
% q3 = quiver(xl/2,yl/2,cosd(0),sind(0),l_len);hold on
% q3.LineWidth=3;
% q3.Color = [.5 .1 .1];
% q3.ShowArrowHead = 'off';
% title([num2str(plocPsi_vec(i,1)),' | ']);%, num2str(vel_dir(i,1))
% axis equal; axis tight; hold off
%%          
          
%           Average PIV fields          
            rot_u  = rot_u + suRR(box-s_box:box+s_box,box-s_box:box+s_box);
            rot_v  = rot_v + svRR(box-s_box:box+s_box,box-s_box:box+s_box);
%           Average ORIENT fields (averaging of nematic field involves flipping, see plot of sub_nx/y1 VS sub_nx/y2)
            nx_temp = snxRR(box-s_box:box+s_box,box-s_box:box+s_box);
            ny_temp = snyRR(box-s_box:box+s_box,box-s_box:box+s_box);
            nx_temp1=nx_temp;nx_temp2=nx_temp;
            ny_temp1=ny_temp;ny_temp2=ny_temp;
            nx_temp1(nx_temp<0) = -nx_temp(nx_temp<0);
            ny_temp1(nx_temp<0) = -ny_temp(nx_temp<0);
            nx_temp2(ny_temp<0) = -nx_temp(ny_temp<0);
            ny_temp2(ny_temp<0) = -ny_temp(ny_temp<0);
            
            rot_nx1  = rot_nx1 + nx_temp1;
            rot_ny1  = rot_ny1 + ny_temp1;
            rot_nx2  = rot_nx2 + nx_temp2;
            rot_ny2  = rot_ny2 + ny_temp2;            

            i_count = i_count+1;
        end
    end
    %%
disp(['analysed defects: ' num2str(i_count) ' from: ' num2str(length(ps_x))]);
if i_count~=0
%     sub_PIV_u = sub_PIV_u + rot_u/mean(sqrt(uu(:).^2+vv(:).^2))/i_count;
%     sub_PIV_v = sub_PIV_v + rot_v/mean(sqrt(uu(:).^2+vv(:).^2))/i_count;
%     disp(['NaN check: ' num2str(sum(sum(isnan(sub_PIV_u))))]);
    v_norm = mean2(sqrt(.5*(uu.^2+vv.^2)));
    sub_PIV_u = sub_PIV_u + rot_u/i_count/v_norm;
    sub_PIV_v = sub_PIV_v + rot_v/i_count/v_norm;
    disp(['NaN check: ' num2str(sum(sum(isnan(sub_PIV_u))))]);
end
% if sum(sum(isnan(sub_PIV_u)))~=0
%     disp(['!!! you have NaNs !!!']);
%     break
% end
% sub_PIV_u =  rot_u/mean(sqrt(uu(:).^2+vv(:).^2));
% sub_PIV_v =  rot_v/mean(sqrt(uu(:).^2+vv(:).^2));
sub_nx1 = sub_nx1+ rot_nx1;
sub_ny1 = sub_ny1+ rot_ny1;
sub_nx2 = sub_nx2+ rot_nx2;
sub_ny2 = sub_ny2+ rot_ny2;

k_count=k_count+1;
total_count = total_count + i_count;  
    
%     catch ME
%         disp(['skipped:', num2str(k)]);
%         disp(['Message: ', ME.message]);
%     end
end

mdPsi = cell2mat(deltaTeta);
% mdPsi = dPsi{k,1};
figure(111); polarhistogram(mdPsi*pi/180,90, 'Normalization','PDF');hold on
figure(222); histogram(mdPsi,90, 'Normalization','PDF'); hold on
mdPsi = cell2mat(projection');
figure(333); histogram(mdPsi,90, 'Normalization','PDF'); hold on
%% ---------------------------------------------------
sub_nx1(sub_ny1<0)=-sub_nx1(sub_ny1<0);
sub_nx = sub_nx1+sub_nx2;
sub_ny = sub_ny1+sub_ny2;
figure(start+121)
% subplot(ceil(((last-start)/kStep).^.5),floor(((last-start)/kStep).^.5),k_count-1)
bin = round(5*box/100);
ff=2*s_box+1;
% quiver(Xu(1:bin:ff,1:bin:ff),Yu(1:bin:ff,1:bin:ff),...
%     sub_PIV_u(1:bin:end,1:bin:end),sub_PIV_v(1:bin:end,1:bin:end),2);axis equal;axis tight; hold on
% Orientation
binO = 4;
q = quiver(Xu(1:binO:ff,1:binO:ff),Yu(1:binO:ff,1:binO:ff),...
    sub_nx(1:binO:end,1:binO:end),sub_ny(1:binO:end,1:binO:end),1);axis equal;axis tight; hold on
q.LineWidth=.5;
q.Color = [0 0 0];
q.ShowArrowHead='off';
p1 = plot(ff/2,ff/2,'o','MarkerFaceColor',[0 .5 .1]);
p1.MarkerSize = 10;
p1.MarkerEdgeColor= 'none';
axis off; title(k); hold off
% 
figure(start+123)
% subplot(ceil(((last-start)/kStep).^.5),floor(((last-start)/kStep).^.5),k_count-1)
[u_x,u_y] = gradient(sub_PIV_u);%/dx gradient need to be corrected for the dx
[v_x,v_y] = gradient(sub_PIV_v);%/dx
vorticity = (v_x - u_y);%------------------- OPTION1
divV = (u_x + v_y);%----- OPTION2
filtN = 30;
filt = fspecial('gaussian',filtN,filtN);
u1 = imfilter(vorticity, filt);
surf(Xu(1:1:ff,1:1:ff),Yu(1:1:ff,1:1:ff),u1-10);view(2);shading interp;colormap jet;axis equal;axis tight;hold on
load('mycbar.mat')
set(gcf,'Colormap',mycbar); %this works
% caxis([-max(u1(:))/2-10, max(u1(:))/2-10]);
caxis([-10-.5, -10+.5]);

q=quiver(Xu(1:bin:ff,1:bin:ff),Yu(1:bin:ff,1:bin:ff),...
    sub_PIV_u(1:bin:end,1:bin:end),sub_PIV_v(1:bin:end,1:bin:end),1);axis equal;axis tight; hold on
q.LineWidth=1;
q.Color = [0 0 0];

p2 = plot3(ff/2,ff/2,40,'o','MarkerFaceColor',[0 0 0]);
p2.MarkerSize = 10;
p2.MarkerEdgeColor= 'none';
% axis off; title([num2str(k),'(',num2str(i_count),')']); hold off   

axis off; title([num2str(k),'(',num2str(total_count),') w_{stripe}= ',...
    num2str(pix2mic*size(Ang,2)),'  w_{box}= ',num2str(s_box* pix2mic *2,3)]); hold off
% 
% % uisave({'sub_PIV_u','sub_PIV_v'},['box_size_um_',num2str(s_box*1*2,3)]);

%% PLOT DEFECTS
%
% tt = ii;
figure(3)
% subplot(1,2,1)
% p1=plot(s.Centroid(:,1),s.Centroid(:,2),'o', 'MarkerFaceColor',[0 0 1]);hold on
% p1.MarkerSize = 10;
% p1.MarkerEdgeColor= 'none';
% figure
l_len = .15;
% +1/2 defect
p2 = plot(ps_x,ps_y,'o','MarkerFaceColor',[0 .5 .1]);hold on
p2.MarkerSize = 5;
p2.MarkerEdgeColor= 'none';

q2 = quiver(ps_x,ps_y,cosd(plocPsi_vec),sind(plocPsi_vec),l_len);hold on
q2.LineWidth=3;
q2.Color = [0 .5 .1];
q2.ShowArrowHead = 'off';

% -1/2 defect
p3 = plot(ns_x,ns_y,'o','MarkerFaceColor',[.8 .1 0]);hold on
p3.MarkerSize = 5;
p3.MarkerEdgeColor= 'none';

q3 = quiver(ns_x,ns_y,cosd(nlocPsi_vec),sind(nlocPsi_vec),l_len);hold on
q3.LineWidth=3;
q3.Color = [.8 .1 0];
q3.ShowArrowHead = 'off';
q4 = quiver(ns_x,ns_y,cosd(nlocPsi_vec+120),sind(nlocPsi_vec+120),l_len);hold on
q4.LineWidth=3;
q4.Color = [.8 .1 0];
q4.ShowArrowHead = 'off';
q5 = quiver(ns_x,ns_y,cosd(nlocPsi_vec+240),sind(nlocPsi_vec+240),l_len);hold on
q5.LineWidth=3;
q5.Color = [.8 .1 0];
q5.ShowArrowHead = 'off';

% Orientation
step = 6;
q6 = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
    cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)),.7);
q6.LineWidth=1;
q6.Color = [.4 .4 .4];
q6.ShowArrowHead='off';

% Velocity
vstep = 1;hold on
q0 = quiver(Xorig(1:vstep:end,1:vstep:end),Yorig(1:vstep:end,1:vstep:end),...
    uu(1:vstep:end,1:vstep:end),vv(1:vstep:end,1:vstep:end),2);
q0.LineWidth=.5;
q0.Color = [0 0 0];

% Vorticity
vstep = 1;
[u_x,u_y] = gradient(uu);%/dx gradient need to be corrected for the dx
[v_x,v_y] = gradient(vv);%/dx
vorticity = (v_x - u_y);
ff = 7;
filt = fspecial('gaussian',ff,ff);
u1 = imfilter(vorticity, filt);
surf(Xorig(1:vstep:end,1:vstep:end),Yorig(1:vstep:end,1:vstep:end),...
    u1(1:vstep:end,1:vstep:end)-200);

% Divergence
% vstep = 1;%
% divV =  divergence(uu,vv);
% ff = 5;
% filt = fspecial('gaussian',ff,ff);
% u2 = imfilter(divV, filt);
% surf(Xorig(1:vstep:end,1:vstep:end),Yorig(1:vstep:end,1:vstep:end),...
%     u2(1:vstep:end,1:vstep:end)-10);

load('mycbar.mat')
set(gcf,'Colormap',mycbar); %this works

% colormap jet
view(2);shading interp;axis equal;axis tight;axis off
set(gca,'View',[-90 90]);
hold off