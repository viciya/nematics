%% LOAD FILES
% PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
PC_path = 'D:\GD\';                         % RSIP notebook

addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

[dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);
%% SELECT WIDTH AND PARAMETERS
clearvars -except dirOP  dirPIV  Sorted_Orient_width  indX PC_path pathOP pathPIV

i = 7;
Sw = 1500; % selectd width
dw = .05*Sw; % define delta
box = 80;
s_box = floor(sqrt(box^2/2));
pix2mic = 3*.74;
Edge = Sw;%70;

Ltot=0;
Rtot=0;
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);
% %%


for i = 1:numel(Range)
    disp(['File ',num2str(i), ' from ',num2str(numel(Range))]);
    filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
    filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];
    
    % [sub_PIV_u,sub_PIV_v, sub_nx1,sub_ny1,sub_nx2,sub_ny2,total_count] = PIV_Orient_pDef(filepathPIV,filepathOP);
    [Lsub_PIV_u,Lsub_PIV_v, Lsub_nx1,Lsub_ny1,Lsub_nx2,Lsub_ny2,Ltotal_count,...
        Rsub_PIV_u,Rsub_PIV_v, Rsub_nx1,Rsub_ny1,Rsub_nx2,Rsub_ny2,Rtotal_count] = ...
        PIV_Orient_pDef(filepathPIV,filepathOP,box,Edge);
    Ltot = Ltot+Ltotal_count;
    Rtot = Rtot+Rtotal_count;
    %% Save single experiments
    % show_PIV_Orent(Lsub_PIV_u,Lsub_PIV_v,Lsub_nx1,Lsub_ny1,Lsub_nx2,Lsub_ny2,1);
    % show_PIV_Orent(Rsub_PIV_u,Rsub_PIV_v,Rsub_nx1,Rsub_ny1,Rsub_nx2,Rsub_ny2,2)
    % subplot(2,2,1);title(['Left def: ', num2str(Ltotal_count)])
    % subplot(2,2,2);title(['box width in um: ',num2str(s_box* pix2mic *2,3)])
    % subplot(2,2,3);title(['Right def: ', num2str(Rtotal_count)])
    % subplot(2,2,4);title(['Edge in um:', num2str(Edge*pix2mic)])
    % set(gcf,'Renderer', 'painters', 'Position', [20 20 900 800])
    % saveas(gcf,['D:\GD\Curie\Carles\HT1080 Turbulence Onset\Figures\edge defects\', num2str(Sw),'-',num2str(i),'.png'])
    
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
Lav_nx1(Lav_ny1<0) = -Lav_nx1(Lav_ny1<0);
Lav_ny1(Lav_nx1<0) = -Lav_ny1(Lav_nx1<0);
Lnx = Lav_nx1 + Lav_nx2;
Lny = Lav_ny1 + Lav_ny2;

        
Rav_PIV_u = Rav_PIV_u / i;
Rav_PIV_v = Rav_PIV_v / i;
Rav_nx1(Rav_ny1<0) = -Rav_nx1(Rav_ny1<0);
Rav_ny1(Rav_nx1<0) = -Rav_ny1(Rav_nx1<0);
Rnx = Rav_nx1 + Rav_nx2;
Rny = Rav_ny1 + Rav_ny2;

%%
% Save all experiments
show_PIV_Orent(Lav_PIV_u,Lav_PIV_v,Lnx,Lny,1); 
show_PIV_Orent(Rav_PIV_u,Rav_PIV_v,Rnx,Rny,2); 
    subplot(2,2,2);title(['Left def: ', num2str(Ltot)])
    subplot(2,2,1);title(['box width in um: ',num2str(s_box* pix2mic *2,3)])
    subplot(2,2,4);title(['Right def: ', num2str(Rtot)])
    subplot(2,2,3);title(['Edge in um:', num2str(Edge*pix2mic)])
    set(gcf,'Renderer', 'painters', 'Position', [40 40 600 600])
% saveas(gcf,[PC_path,'Curie\Carles\HT1080 Turbulence Onset\Figures\edge defects\', num2str(Sw),'-','AVERAGE.png'])
%%
function [Lsub_PIV_u,Lsub_PIV_v, Lsub_nx1,Lsub_ny1,Lsub_nx2,Lsub_ny2,Ltotal_count,...
          Rsub_PIV_u,Rsub_PIV_v, Rsub_nx1,Rsub_ny1,Rsub_nx2,Rsub_ny2,Rtotal_count] = ...
          PIV_Orient_pDef(filepathPIV,filepathOP,boxA,Edge)
      
info = imfinfo(filepathOP); % Place path to file inside single quotes
Nn = numel(info);
load(filepathPIV);

ff = 3;
filt = fspecial('gaussian',ff,ff);

pix2mic = 3*.74;
k_count = 1;
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

% pp = i+1
for k=1:Nn-1%start:kStep:last
%     try
%     k
    clearvars -except k kStep k_count info Nn filename pathname...
        filterindex pix2mic A projection t_shift deltaTeta rot_u rot_v...
        box s_box start last Ltotal_count Rtotal_count filt...
        Lsub_PIV_u Lsub_PIV_v Lsub_nx1 Lsub_ny1 Lsub_nx2 Lsub_ny2...
        Rsub_PIV_u Rsub_PIV_v Rsub_nx1 Rsub_ny1 Rsub_nx2 Rsub_ny2...
        Lrot_u Lrot_v Lrot_nx1 Lrot_ny1 Lrot_nx2 Lrot_ny2 Li_count...
        Rrot_u Rrot_v  Rrot_nx1 Rrot_ny1 Rrot_nx2 Rrot_ny2 Ri_count...
        u v x y filepathOP filepathPIV Sorted_Orient_width Xu Yu dPsi...
        indX dirOP dirPIV Edge
    
    qstep = 15;%6; %mine 10 in pixels
    Ang = imread(filepathOP,k); % k
    [l,w] = size(Ang);
    
    % --------------------------PIV import ---------------------------------    
    uu = imfilter(u{k}, filt);
    vv = imfilter(v{k}, filt);
    sc = size(Ang,1)/size(uu,1);    
    
%     [ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec]
    [ps_xA, ps_yA, plocPsi_vecA, ~,  ~,  ~] = fun_get_pn_Defects(Ang);
    Ledge_select = ps_xA < Edge;   %1/4*w;
    Redge_select = ps_xA > size(Ang,2)- Edge;     %3/4*w;
    Lps_x = ps_xA(Ledge_select);
    Lps_y = ps_yA(Ledge_select);
    LplocPsi_vec = plocPsi_vecA(Ledge_select);
    Rps_x = ps_xA(Redge_select);
    Rps_y = ps_yA(Redge_select);
    RplocPsi_vec = plocPsi_vecA(Redge_select);
    
    [Xorig,Yorig] = meshgrid((1/sc:size(uu,2))*sc,(1/sc:size(uu,1))*sc);
    [Xu,Yu] = meshgrid(1:w,1:l);
    u_interp = interp2(Xorig,Yorig,uu,Xu,Yu,'cubic',0);
    v_interp = interp2(Xorig,Yorig,vv,Xu,Yu,'cubic',0);
    
[Lrot_u,Lrot_v, Lrot_nx1,Lrot_ny1,Lrot_nx2,Lrot_ny2,Li_count] = getLocalPIVandOrient(Ang, u_interp,v_interp,Lps_x,Lps_y,LplocPsi_vec,box);
[Rrot_u,Rrot_v, Rrot_nx1,Rrot_ny1,Rrot_nx2,Rrot_ny2,Ri_count] = getLocalPIVandOrient(Ang, u_interp,v_interp,Rps_x,Rps_y,RplocPsi_vec,box);
 
disp(['Frame: ' num2str(k) ' from: ' num2str(Nn)]);
disp(['Left, analysed defects: ' num2str(Li_count) ' from: ' num2str(length(Lps_x))]);
disp(['Right, analysed defects: ' num2str(Ri_count) ' from: ' num2str(length(Lps_x))]);


v_norm = mean2(sqrt(.5*(uu.^2+vv.^2)));
if Li_count~=0    
    Lsub_PIV_u = Lsub_PIV_u + Lrot_u/Li_count/v_norm;
    Lsub_PIV_v = Lsub_PIV_v + Lrot_v/Li_count/v_norm;
end
if Ri_count~=0  
    Rsub_PIV_u = Rsub_PIV_u + Rrot_u/Ri_count/v_norm;
    Rsub_PIV_v = Rsub_PIV_v + Rrot_v/Ri_count/v_norm;
%     disp(['NaN check: ' num2str(sum(sum(isnan(sub_PIV_u))))]);
end

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

end


function [rot_u,rot_v, rot_nx1,rot_ny1,rot_nx2,rot_ny2,i_count] = getLocalPIVandOrient(Ang, u_interp,v_interp,ps_x,ps_y,plocPsi_vec,box)
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
end

% function show_PIV_Orent(u,v,nx1,nx2,ny1,ny2,LR)
function show_PIV_Orent(u,v,nx,ny,LR)
[l, w] = size(u);
[Xu,Yu] = meshgrid(1:w,1:l);
ff = l;

% nx1(ny1<0)=-nx1(ny1<0);
% ny1(nx1<0)=-ny1(nx1<0);
% nx = nx1+nx2;
% ny = ny1+ny2;

bin = round(l/25);
binO = bin+2;

figure(1);
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
filtN = 30;
filt = fspecial('gaussian',filtN,filtN);
u1 = imfilter(vorticity, filt);
surf(Xu(1:1:ff,1:1:ff),Yu(1:1:ff,1:1:ff),u1-10);view(2);shading interp;colormap jet;axis equal;axis tight;hold on
load('mycbar.mat')
set(gcf,'Colormap',mycbar); %this works
caxis([-10-1, -10+1]);

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
