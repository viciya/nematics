%% LOAD FILES
PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
% PC_path = 'D:\GD\';                         % RSIP notebook

addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

% [dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);

folder_path = 'Curie\DESKTOP\HT1080\';
name_template = 'symm_pDefect_x_y_angle_frame_idx_2_';
width_set = [300,400,500,600,700,800,1000,1500];
for i = length(width_set)
    set_name{i} = [name_template, num2str(width_set(i)),'.txt'];
end
%% SELECT WIDTH AND PARAMETERS
clearvars -except dirOP  dirPIV  Sorted_Orient_width  indX PC_path pathOP pathPIV ...
    folder_path name_template set_name width_set px_size

for n = length(width_set)

Sw = width_set(n); % selectd width
dw = .05*Sw; % define delta
box = 80;
s_box = floor(sqrt(box^2/2));
px_size = 0.74;
pix2mic = 3 * px_size;

edge = 60;%70;

Ltot=0;
Rtot=0;
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);
% %%

T = table2array(readtable([PC_path, folder_path, set_name{n}]));
Lvel_angle = [];
Rvel_angle = [];
%%

for i = 1:max(T(:,5))%:numel(Range)
    iT  = T(T(:,5)==i,:);
    disp('.............................................');
    disp(['File ',num2str(i), ' from ',num2str(max(T(:,5)))]);
    disp('.............................................');
    filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
    filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];
    
    % [sub_PIV_u,sub_PIV_v, sub_nx1,sub_ny1,sub_nx2,sub_ny2,total_count] = PIV_Orient_pDef(filepathPIV,filepathOP);
    [Lsub_PIV_u,Lsub_PIV_v, Lsub_nx1,Lsub_ny1,Lsub_nx2,Lsub_ny2,Ltotal_count,...
        Rsub_PIV_u,Rsub_PIV_v, Rsub_nx1,Rsub_ny1,Rsub_nx2,Rsub_ny2,Rtotal_count,...
        Lvel_angle_tmp,Rvel_angle_tmp] = ...
        PIV_Orient_pDef(filepathPIV,filepathOP,box,edge,iT);
    
    Lvel_angle = [Lvel_angle, Lvel_angle_tmp];
    Rvel_angle = [Rvel_angle, Rvel_angle_tmp];
    Ltot = Ltot + Ltotal_count;
    Rtot = Rtot + Rtotal_count;
    
    if exist('Lav_PIV_u', 'var')
        %        NaNs should be replaced with zeros
        Lav_PIV_u = Lav_PIV_u + Lsub_PIV_u;
        Lav_PIV_v = Lav_PIV_v + Lsub_PIV_v;
        Lav_nx1 = Lav_nx1 + Lsub_nx1;
        Lav_nx2 = Lav_nx2 + Lsub_nx2;
        Lav_ny1 = Lav_ny1 + Lsub_ny1;
        Lav_ny2 = Lav_ny2 + Lsub_ny2;
        
        Rav_PIV_u = Rav_PIV_u + Rsub_PIV_u;
        Rav_PIV_v = Rav_PIV_v + Rsub_PIV_v;
        Rav_nx1 = Rav_nx1 + Rsub_nx1;
        Rav_nx2 = Rav_nx2 + Rsub_nx2;
        Rav_ny1 = Rav_ny1 + Rsub_ny1;
        Rav_ny2 = Rav_ny2 + Rsub_ny2;
    else
        Lav_PIV_u = Lsub_PIV_u;
        Lav_PIV_v = Lsub_PIV_v;
        Lav_nx1 = Lsub_nx1;
        Lav_nx2 = Lsub_nx2;
        Lav_ny1 = Lsub_ny1;
        Lav_ny2 = Lsub_ny2;
        
        Rav_PIV_u = Rsub_PIV_u;
        Rav_PIV_v = Rsub_PIV_v;
        Rav_nx1 = Rsub_nx1;
        Rav_nx2 = Rsub_nx2;
        Rav_ny1 = Rsub_ny1;
        Rav_ny2 = Rsub_ny2;
    end
end
Lav_PIV_u = Lav_PIV_u / i;
Lav_PIV_v = Lav_PIV_v / i;


% Lav_nx1(Lav_ny1<0) = -Lav_nx1(Lav_ny1<0);
% Lav_ny1(Lav_nx1<0) = -Lav_ny1(Lav_nx1<0);
% Lnx = Lav_nx1 + Lav_nx2;
% Lny = Lav_ny1 + Lav_ny2;
L_theta = .5 * atan2(Lav_ny1, Lav_nx1);
Lnx = cos(L_theta);
Lny = sin(L_theta);



Rav_PIV_u = Rav_PIV_u / i;
Rav_PIV_v = Rav_PIV_v / i;
% Rav_nx1(Rav_ny1<0) = -Rav_nx1(Rav_ny1<0);
% Rav_ny1(Rav_nx1<0) = -Rav_ny1(Rav_nx1<0);
% Rnx = Rav_nx1 + Rav_nx2;
% Rny = Rav_ny1 + Rav_ny2;
R_theta = .5 * atan2(Rav_ny1, Rav_nx1);
Rnx = cos(R_theta);
Rny = sin(R_theta);

%
%% Save all experiments
FOLDER = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\average_flows-av_core-box6-3\';
SAVE = false;
figure(Sw);
vort_filt = 10;
% space=5;
% quiver(Lnx(1:space:end,1:space:end),Lny(1:space:end,1:space:end));axis equal
show_PIV_Orent(Lav_PIV_u,Lav_PIV_v,Lnx,Lny,1,vort_filt);
show_PIV_Orent(Rav_PIV_u,Rav_PIV_v,Rnx,Rny,2,vort_filt);

vrms_L = sqrt(.5*(Lav_PIV_u.^2 + Lav_PIV_v.^2));
vrms_R = sqrt(.5*(Rav_PIV_u.^2 + Rav_PIV_v.^2));
cent = (size(vrms_L,1) + 1)/2; d = 3;

vrms_L_cent = mean2(vrms_L(cent-d:cent+d, cent-d:cent+d));
vrms_R_cent = mean2(vrms_R(cent-d:cent+d, cent-d:cent+d));

subplot(2,2,2);title({['Edge def: ', num2str(Ltot)],...
                      ['vrms@core: ', num2str(vrms_L_cent,3),' / ',...
                                   '*']})
subplot(2,2,1);title(['box width in um: ',num2str(s_box* pix2mic *2,3)])
subplot(2,2,4);title({['Non-edge def: ', num2str(Rtot)],...
                      ['vrms@core: ', num2str(vrms_R_cent,3),' / ',...
                                   '*']})                                       
      
subplot(2,2,3);title(['Edge in um:', num2str(edge*pix2mic)])
set(gcf,'Renderer', 'painters', 'Position', [10 10 800 600])
set(gcf,'Color',[1 1 1]);
    if SAVE
        saveas(gcf,[FOLDER, num2str(Sw),'box80.png'])
    end

figure(Sw+1);
polarhistogram(Rvel_angle,25); hold on
polarhistogram(Lvel_angle,25); hold off
title({['Edge: ', num2str(circ_mean(Lvel_angle(:))*180/pi,3),' | ',...
    num2str(circ_std(Lvel_angle(:))*180/pi,3),]...
    ['Non-edge: ',num2str(circ_mean(Rvel_angle(:))*180/pi,3),' | ',...
    num2str(circ_std(Rvel_angle(:))*180/pi,3)],...
    ['(mean | std)']})
set(gcf,'Color',[1 1 1]);
    if SAVE
        saveas(gcf,[FOLDER, num2str(Sw),'box80_ang.png'])
        save_file = [FOLDER, 'pdef_av_piv_orient_box80pad_w_v_ang_',num2str(Sw),'.mat'];
        save(save_file,'Lav_PIV_u','Lav_PIV_v','Lnx','Lny',...
            'Rav_PIV_u','Rav_PIV_v','Rnx','Rny','Sw',...
            'Ltot', 'Rtot', 's_box', 'pix2mic', 'edge', 'Lvel_angle', 'Rvel_angle');
    end
end
%%
function [Lsub_PIV_u,Lsub_PIV_v, Lsub_nx1,Lsub_ny1,Lsub_nx2,Lsub_ny2,Ltotal_count,...
    Rsub_PIV_u,Rsub_PIV_v, Rsub_nx1,Rsub_ny1,Rsub_nx2,Rsub_ny2,Rtotal_count,...
    Lvel_angle, Rvel_angle] = ...
    PIV_Orient_pDef(filepathPIV,filepathOP,boxA,edge,defect_list)

px_size = 0.74;
px = defect_list(:,1);
py = defect_list(:,2);
theta = defect_list(:,3);
frame = defect_list(:,4);

info = imfinfo(filepathOP); % Place path to file inside single quotes
Nn = numel(info);
load(filepathPIV);

ff = 3;
filt = fspecial('gaussian',ff,ff);

pix2mic = 3*.74;
k_count = 0;
Ltotal_count = 0;
Rtotal_count = 0;

box = boxA;
s_box = floor(sqrt(box^2/2));
disp(['box width in um: ',num2str(s_box* pix2mic *2,3)]);
Lsub_PIV_u = zeros(2*s_box+1);
Lsub_PIV_v = Lsub_PIV_u;
Lsub_nx1 = Lsub_PIV_u;
Lsub_ny1 = Lsub_PIV_u;
Lsub_nx2 = Lsub_PIV_u;
Lsub_ny2 = Lsub_PIV_u;
Rsub_PIV_u = Lsub_PIV_u;
Rsub_PIV_v = Lsub_PIV_u;
Rsub_nx1 = Lsub_PIV_u;
Rsub_ny1 = Lsub_PIV_u;
Rsub_nx2 = Lsub_PIV_u;
Rsub_ny2 = Lsub_PIV_u;
Lvel_angle = [];
Rvel_angle = [];

% pp = i+1
for k=1:Nn-1%start:kStep:last
    %     try
    %     k
%     clearvars -except k kStep k_count info Nn filename pathname...
%         filterindex pix2mic A projection t_shift deltaTeta rot_u rot_v...
%         box s_box start last Ltotal_count Rtotal_count filt...
%         Lsub_PIV_u Lsub_PIV_v Lsub_nx1 Lsub_ny1 Lsub_nx2 Lsub_ny2...
%         Rsub_PIV_u Rsub_PIV_v Rsub_nx1 Rsub_ny1 Rsub_nx2 Rsub_ny2...
%         Lrot_u Lrot_v Lrot_nx1 Lrot_ny1 Lrot_nx2 Lrot_ny2 Li_count...
%         Rrot_u Rrot_v  Rrot_nx1 Rrot_ny1 Rrot_nx2 Rrot_ny2 Ri_count...
%         u v x y filepathOP filepathPIV Sorted_Orient_width Xu Yu dPsi...
%         indX dirOP dirPIV Edge px py theta frame edge
    
    qstep = 15;%6; %mine 10 in pixels
    Ang = imread(filepathOP,k); % k
    [l,w] = size(Ang);
    
    if any( Ang(:)>4 ) % check if Ang is in RAD
        Ang = Ang * pi/180;
    end
    
    % --------------------------PIV import ---------------------------------
    uu = px_size * (imfilter(u{k}, filt));
    uu = uu - mean2(uu);
    vv = px_size * (imfilter(v{k}, filt));
    vv = vv - mean2(vv);
    sc = size(Ang,1)/size(uu,1);
    
    %     [ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec]
%     [ps_xA, ps_yA, plocPsi_vecA, ns_xA, ns_yA, nlocPsi_vecA] = ...
%         fun_get_pn_Defects_newDefectAngle_blockproc(Ang);
    %         fun_get_pn_Defects_newDefectAngle(Ang);
    
    ps_xA  = px(frame==k & (px<edge | px>size(Ang,2)-edge));
    ps_yA  = py(frame==k);
    plocPsi_vecA  = theta(frame==k);
    
    ns_xA  = px(frame==k  & (px>edge | px<size(Ang,2)-edge));
    ns_yA  = py(frame==k);
    nlocPsi_vecA  = theta(frame==k);

    
    
    % ---------- SHOW ORIENTATION FIELD AND DEFECTS  ----------
    %     show_defects(Ang,ps_xA, ps_yA, plocPsi_vecA, ns_xA, ns_yA, nlocPsi_vecA)
    %----------------------------------------------------------
    
    Lps_x = ps_xA;
    Lps_y = ps_yA;
    LplocPsi_vec = plocPsi_vecA;
    Rps_x = ns_xA;
    Rps_y = ns_yA;
    RplocPsi_vec = nlocPsi_vecA;
    
    [Xorig,Yorig] = meshgrid((1/sc:size(uu,2))*sc,(1/sc:size(uu,1))*sc);
    [Xu,Yu] = meshgrid(1:w,1:l);
    u_interp = interp2(Xorig,Yorig,uu,Xu,Yu,'cubic',0);
    v_interp = interp2(Xorig,Yorig,vv,Xu,Yu,'cubic',0);
    
    [Lrot_u,Lrot_v, Lrot_nx1,Lrot_ny1,Lrot_nx2,Lrot_ny2,~,Li_count,Lvel_angle_temp] = ...
        fun_get_local_velocity_and_orientation_padded(Ang, u_interp,v_interp,Lps_x,Lps_y,LplocPsi_vec,box,1500,1500);
    [Rrot_u,Rrot_v, Rrot_nx1,Rrot_ny1,Rrot_nx2,Rrot_ny2,~,Ri_count,Rvel_angle_temp] = ...
        fun_get_local_velocity_and_orientation_padded(Ang, u_interp,v_interp,Rps_x,Rps_y,RplocPsi_vec,box,1500,1500);
    % !!!!!!!!!!!!!!!!!!!!!!! REPLACED "getLocalPIVandOrient"
    Lvel_angle = [Lvel_angle, Lvel_angle_temp];
    Rvel_angle = [Rvel_angle, Rvel_angle_temp];
    
    disp(['Frame: ' num2str(k) ' from: ' num2str(Nn)]);
    disp(['Left, analysed defects: ' num2str(Li_count) ' from: ' num2str(length(Lps_x))]);
    disp(['Right, analysed defects: ' num2str(Ri_count) ' from: ' num2str(length(Rps_x))]);
    
    
    %v_norm = mean2(sqrt(.5*(uu.^2+vv.^2)));
    if Li_count~=0
        Lsub_PIV_u = Lsub_PIV_u + Lrot_u;%/Li_count/v_norm;
        Lsub_PIV_v = Lsub_PIV_v + Lrot_v;%/Li_count;%/v_norm;
    end
    if Ri_count~=0
        Rsub_PIV_u = Rsub_PIV_u + Rrot_u;%/Ri_count;%/v_norm;
        Rsub_PIV_v = Rsub_PIV_v + Rrot_v;%/Ri_count;%/v_norm;
        %     disp(['NaN check: ' num2str(sum(sum(isnan(sub_PIV_u))))]);
    end
    
    vrms_Lrot = mean2(sqrt(.5*(Lrot_u.^2 + Lrot_v.^2)));
    vrms_Rrot = mean2(sqrt(.5*(Rrot_u.^2 + Rrot_v.^2)));
    disp(['NEW:' num2str(vrms_Lrot,3),'/',...
                 num2str(vrms_Rrot,3)]);   
    disp(['ALL:' num2str(mean(abs(Rsub_PIV_u(:))),3),'/',...
                 num2str(mean(abs(Rsub_PIV_v(:))),3),'/',...
                 num2str(mean(abs(Lsub_PIV_u(:))),3),'/',...
                 num2str(mean(abs(Lsub_PIV_v(:))),3)]);
    
    Lsub_nx1 = Lsub_nx1+ Lrot_nx1;
    Lsub_ny1 = Lsub_ny1+ Lrot_ny1;
    Lsub_nx2 = Lsub_nx2+ Lrot_nx2;
    Lsub_ny2 = Lsub_ny2+ Lrot_ny2;
    
    Rsub_nx1 = Rsub_nx1+ Rrot_nx1;
    Rsub_ny1 = Rsub_ny1+ Rrot_ny1;
    Rsub_nx2 = Rsub_nx2+ Rrot_nx2;
    Rsub_ny2 = Rsub_ny2+ Rrot_ny2;
    
    k_count=k_count+1;
    Ltotal_count = Ltotal_count + Li_count;
    Rtotal_count = Rtotal_count + Ri_count;
    
    %     catch ME
    %         disp(['skipped:', num2str(k)]);
    %         disp(['Message: ', ME.message]);
    %     end
    
end

Lsub_PIV_u = Lsub_PIV_u/k_count;
Lsub_PIV_v = Lsub_PIV_v/k_count;

Rsub_PIV_u = Rsub_PIV_u/k_count;
Rsub_PIV_v = Rsub_PIV_v/k_count;
disp('********************************************')
vrms_Lsub = sqrt(.5*(Lsub_PIV_u.^2+Lsub_PIV_v.^2));
vrms_Rsub = sqrt(.5*(Rsub_PIV_u.^2+Rsub_PIV_v.^2));
disp(['STRIPE-ALL:' num2str(max(vrms_Lsub(:)),3),'/',...
             num2str(max(vrms_Rsub(:)),3)]);
end


function [rot_u,rot_v, rot_nx1,rot_ny1,rot_nx2,rot_ny2,i_count] = ...
    getLocalPIVandOrient(Ang, u_interp,v_interp,ps_x,ps_y,plocPsi_vec,box)
s_box = floor(sqrt(box^2/2));
[l,w] = size(u_interp);
% ///////////////   ROTATION //////////////////////////////////////
rot_u = zeros(2*s_box+1);
rot_v = rot_u;
rot_nx1 = rot_u;
rot_ny1 = rot_u;
rot_nx2 = rot_u;
rot_ny2 = rot_u;
%%
i_count = 0;

for i=1:length(ps_x)
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
end

% function show_PIV_Orent(u,v,nx1,nx2,ny1,ny2,LR)
function show_PIV_Orent(u,v,nx,ny,LR,filtN)
u = u - mean(u);
v = v - mean(u);
[l, w] = size(u);
[Xu,Yu] = meshgrid(1:w,1:l);
ff = l;

% nx1(ny1<0)=-nx1(ny1<0);
% ny1(nx1<0)=-ny1(nx1<0);
% nx = nx1+nx2;
% ny = ny1+ny2;

bin = round(l/25);
binO = bin+2;

% figure();
subplot(2,2,2*LR-1);
q = quiver(Xu(1:binO:ff,1:binO:ff),Yu(1:binO:ff,1:binO:ff),...
    nx(1:binO:end,1:binO:end),ny(1:binO:end,1:binO:end),1);axis equal;axis tight;hold on
q.LineWidth=.5;
q.Color = [0 0 0];
q.ShowArrowHead='off';
axis off; %title(total_count);
hold off

% figure(FIG+1)
subplot(2,2,2*LR);
[u_x,u_y] = gradient(u);%/dx gradient need to be corrected for the dx
[v_x,v_y] = gradient(v);%/dx
vorticity = (v_x - u_y);%------------------- OPTION1
divV = (u_x + v_y);%----- OPTION2

filt = fspecial('gaussian',filtN,filtN);
u1 = imfilter(vorticity, filt);
surf(Xu(1:1:ff,1:1:ff),Yu(1:1:ff,1:1:ff),u1-10);view(2);shading interp;colormap jet;axis equal;axis tight;hold on
load('mycbar.mat')
set(gcf,'Colormap',mycbar); %this works
caxis([-10-max(u1(:))/2, -10+max(u1(:))/2]);

q=quiver(Xu(1:bin:ff,1:bin:ff),Yu(1:bin:ff,1:bin:ff),...
    u(1:bin:end,1:bin:end),v(1:bin:end,1:bin:end),1);axis equal;axis tight; hold on
q.LineWidth=1;
q.Color = [0 0 0];

p2 = plot3(round(ff/2),round(ff/2),40,'o','MarkerFaceColor',[0 0 0]);
p2.MarkerSize = 5;
p2.MarkerEdgeColor= 'none';
axis off;
hold off
end
