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
i = 8;

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
k_count = 1;
total_count = 0;

r2_Circ = 20;
box = 50;
s_box = floor(sqrt(box^2/2));
disp(['box width in um: ',num2str(s_box* pix2mic *2,3)]);
sub_PIV_u = zeros(2*s_box+1);
sub_PIV_v = sub_PIV_u;
sub_nx1 = sub_PIV_u;
sub_ny1 = sub_nx1;
sub_nx2 = sub_nx1;
sub_ny2 = sub_nx1;

% for k=1:Nn-dt
start= 1; kStep = 1; last= Nn-1;%+40;
%
% pp = i+1
for k=1:2%Nn-1%start:kStep:last
    try
    k
    clearvars -except k kStep k_count info Nn filename pathname...
        filterindex pix2mic A projection t_shift deltaTeta rot_u rot_v...
        box s_box start last total_count filt...
        sub_PIV_u sub_PIV_v sub_nx1 sub_ny1 sub_nx2 sub_ny2...
        u v x y filepathOP filepathPIV Sorted_Orient_width Xu Yu dPsi...
        indX dirOP dirPIV r2_Circ
    
    qstep = 15;%6; %mine 10 in pixels
    Ang = imread(filepathOP,k); % k
    [l,w] = size(Ang);
    
%     [ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec]
    [ps_xA, ps_yA, plocPsi_vecA,   ~,    ~,    ~] = fun_get_pn_Defects(Ang, r2_Circ);
% -----    SELECT EDGES ONLY ------
%     edge_select = ps_xA<1/4*w;
%     edge_select = ps_xA>size(Ang,2)-65;%3/4*w;
%     ps_x = ps_xA(edge_select);
%     ps_y = ps_yA(edge_select);
%     plocPsi_vec = plocPsi_vecA(edge_select);
% -----    SELECT ALL ------
    ps_x = ps_xA;
    ps_y = ps_yA;
    plocPsi_vec = plocPsi_vecA;
    % --------------------------PIV import ---------------------------------    

    uu = imfilter(u{k}, filt);
    vv = imfilter(v{k}, filt);
%     [u_x,u_y] = gradient(uu);%/dx gradient need to be corrected for the dx
%     [v_x,v_y] = gradient(vv);
%     Enstrophy = mean2(0.5*(v_x - u_y).^2);
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
% % imagesc(q); axis equal; axis tight;hold on
% p1=plot(s.Centroid(:,1),s.Centroid(:,2),'o', 'MarkerFaceColor',[0 0 0]);
% p1.MarkerSize = 6;
% p1.MarkerEdgeColor= 'none';
% step = 10;
% q1 = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
%     cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)),.7);
% q1.LineWidth=1;
% q1.Color = [.4 .4 .4];
% q1.ShowArrowHead='off';
% axis equal; axis tight; hold on
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

%  Calculate the difference between velocity and depect angle   
    r_Circ = 5; % this value could be diffrent from the one used in defect classification
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
        loc_u_vec(i,1)= nanmean(nanmean(loc_u));
        loc_v_vec(i,1)= nanmean(nanmean(loc_v));        
    end  
    projection{k_count} = (loc_u_vec.*cosd(plocPsi_vec)+loc_v_vec.*sind(plocPsi_vec))...
        ./sqrt(loc_u_vec.^2+loc_v_vec.^2);  % dot product of velocity and defect directions
    
    temp = atan2d(loc_v_vec,loc_u_vec)-plocPsi_vec;
    temp(temp<0) = temp(temp<0)+360;
    deltaTeta{k_count,1} = temp;
    
% ///////////////   ROTATION ////////////////////////////////////// 
    rot_u = zeros(2*s_box+1);
    rot_v = rot_u;
    rot_nx1 = rot_u;
    rot_ny1 = rot_u;
    rot_nx2 = rot_u;
    rot_ny2 = rot_u;    
%%     
    i_count = 0;
    ii=23;
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
    
    catch ME
        disp(['skipped:', num2str(k)]);
        disp(['Message: ', ME.message]);
    end
end

% mdPsi = cell2mat(deltaTeta);
% mdPsi = dPsi{k,1};
% figure(111); polarhistogram(mdPsi*pi/180,90, 'Normalization','PDF');hold on
% figure(222); histogram(mdPsi,90, 'Normalization','PDF'); hold on
% mdPsi = cell2mat(projection');
% figure(333); histogram(mdPsi,90, 'Normalization','PDF'); hold on

% %%
sub_nx11 = sub_nx1;
sub_ny11 = sub_ny1;
sub_nx22 = sub_nx2;
sub_ny22 = sub_ny2;

sub_nx11(sub_ny11<0)=-sub_nx11(sub_ny11<0);
sub_ny11(sub_nx11<0)=-sub_ny11(sub_nx11<0);
sub_nx = sub_nx11+sub_nx22;
sub_ny = sub_ny11+sub_ny22;
figure(start+121)
% subplot(ceil(((last-start)/kStep).^.5),floor(((last-start)/kStep).^.5),k_count-1)
bin = round(5*box/100);
ff=2*s_box+1;
% quiver(Xu(1:bin:ff,1:bin:ff),Yu(1:bin:ff,1:bin:ff),...
%     sub_PIV_u(1:bin:end,1:bin:end),sub_PIV_v(1:bin:end,1:bin:end),2);axis equal;axis tight; hold on
% Orientation
binO = 3;
q = quiver(Xu(1:binO:ff,1:binO:ff),Yu(1:binO:ff,1:binO:ff),...
    sub_nx(1:binO:end,1:binO:end),sub_ny(1:binO:end,1:binO:end),1);axis equal;axis tight;hold on
q.LineWidth=.5;
q.Color = [0 0 0];
q.ShowArrowHead='off';
p1 = plot(round(ff/2),round(ff/2),'o','MarkerFaceColor',[0 .5 .1]);
p1.MarkerSize = 10;
p1.MarkerEdgeColor= 'none';
axis off; title(k); hold off
% 
figure(start+124)
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

p2 = plot3(round(ff/2),round(ff/2),40,'o','MarkerFaceColor',[0 0 0]);
p2.MarkerSize = 10;
p2.MarkerEdgeColor= 'none';
% axis off; title([num2str(k),'(',num2str(i_count),')']); hold off   

axis off; title([num2str(k),'(',num2str(total_count),') w_{stripe}= ',...
    num2str(pix2mic*size(Ang,2)),'  w_{box}= ',num2str(s_box* pix2mic *2,3)]); hold off
% 
% % uisave({'sub_PIV_u','sub_PIV_v'},['box_size_um_',num2str(s_box*1*2,3)]);