%% make list of Orient files
PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
% PC_path = 'D:\GD\';                         % RSIP notebook

addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

[dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);

PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
folder_path = 'Curie\DESKTOP\HT1080\';
name_template = 'symm_pDefect_x_y_angle_frame_idx_2_';
width_set = [300,400,500,600,700,800,1000,1500];
for i = 1:length(width_set)
    set_name{i} = [name_template, num2str(width_set(i)),'.txt'];
end
%%
px_sizeOP = 3* .74;
edge = 80;
box = 60;

wset = [300,400,500,600,700,800,1000];
for n=4%:length(wset)

Sw = wset(n); % selectd width
dw = .05*Sw; % define delta
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);

ff = 3;
filt = fspecial('gaussian',ff,ff);

[loc_u, loc_v, x1, y1, x2, y2, count] = empty_set(box);
for i = 1%:min(length(Range),100/(2*n))
    
        T = table2array(readtable([PC_path, folder_path, set_name{n}]));
        px = T(:,1);
        py = T(:,2);
        theta = T(:,3);
        frame = T(:,4);
        file_in_range = T(:,5);
        
    disp([num2str(i),' **** from: ', num2str(length(Range))])
    disp('----------------------------------------------');
    % get path orient and piv
    filepathOP = [dirOP(Range(i)).folder, '\', dirOP(indX(Range(i),1)).name];
    info = imfinfo(filepathOP); % Place path to file inside single quotes
    iwidth = px_sizeOP*info(1).Width;
    
    filepathPIV = [dirPIV(Range(i)).folder, '\', dirPIV(indX(Range(i),2)).name];
    piv = load(filepathPIV);
    
    [iloc_u, iloc_v, ix1, iy1, ix2, iy2, icount] = empty_set(box);
    kcount = 1;
    for k=1:numel(info)-1
       
        %         get defect positions and angles
        ang = imread(filepathOP,k);
%         [ps_x,ps_y,ptheta] = get_p_defect_symm(ang);
            ps_x  = px(file_in_range==i & frame==k);
            ps_y  = py(file_in_range==i & frame==k);
            ptheta  = theta(file_in_range==i & frame==k);
        
%         uu = imfilter(piv.u{k}, filt);
%         vv = imfilter(piv.v{k}, filt);
        [ang_x,ang_y] = meshgrid(1:size(ang,2),1:size(ang,1));
        u_interp = interp2(piv.x{k},piv.y{k},piv.u{k},ang_x,ang_y,'cubic',0);
        v_interp = interp2(piv.x{k},piv.y{k},piv.v{k},ang_x,ang_y,'cubic',0);
        
        
        [kloc_u, kloc_v, kloc_nx1,kloc_ny1,kloc_nx2,kloc_ny2,kx_pos,ki_count] = ...
            fun_get_local_velocity_and_orientation...
            (ang, u_interp, v_interp, ps_x, ps_y, ptheta, box, edge);
        
        icount = [icount;ki_count];
        
        if exist('kloc_u', 'var')
            iloc_u = iloc_u + ki_count*kloc_u;
            iloc_v = iloc_v + ki_count*kloc_v;
            ix1 = ix1 + kloc_nx1;
            iy1 = iy1 + kloc_ny1;
            ix2 = ix2 + kloc_nx2;
            iy2 = iy2 + kloc_ny2; 
        end        
        disp(['Done -- ',num2str(k),'(',num2str(ki_count),')'])
        
    end
    if sum(icount)~=0
        iloc_u = iloc_u/sum(icount);
        iloc_v = iloc_v/sum(icount);
        
        loc_u = loc_u + iloc_u;
        loc_v = loc_v + iloc_v;
        
        x1 = x1 + ix1;
        y1 = y1 + iy1;
        x2 = x2 + ix2;
        y2 = y2 + iy2;
    end
    
    count = [count;icount]; 
end

loc_u = loc_u/i;
loc_v = loc_v/i;    
x1(y1<0) = -x1(y1<0);
y1(x1<0) = -y1(x1<0);
nx = x1 + x2;
ny = y1 + y2; 

PC_path = 'C:\Users\victo\Google Drive\'; 
folder_path = 'Curie\DESKTOP\HT1080\';
save_file = [PC_path, folder_path, 'edge_pdef_av_piv_orient_',num2str(Sw),'.mat'];
% save(save_file);
end
% %%
show_PIV_Orent(loc_u,loc_u,nx,ny)
title(sum(count))
%%
function [x,y,q] = get_p_defect_symm(ang)
x = [];
y = [];
q = [];
[ix, iy, iq, ~, ~, ~] = ...
    fun_get_pn_Defects_newDefectAngle(ang);
x = [x;ix];
y = [y;iy];
q = [q;iq];

[ix, iy, iq, ~, ~, ~] = ...
    fun_get_pn_Defects_newDefectAngle(fliplr(ang));
x = [x;size(ang,2)-ix];
y = [y;iy];
q = [q;iq];
end

function [u, v, x1,y1,x2,y2,count] = empty_set(box)
s_box = floor(sqrt(box^2/2));
u=zeros(2*s_box+1);
v=u;
x1=u;
y1=u;
x2=u;
y2=u;
count=[];
end

function show_PIV_Orent(u,v,nx,ny)
[l, w] = size(u);
[Xu,Yu] = meshgrid(1:w,1:l);
ff = l;

% nx1(ny1<0)=-nx1(ny1<0);
% ny1(nx1<0)=-ny1(nx1<0);
% nx = nx1+nx2;
% ny = ny1+ny2;

bin = round(l/30);
binO = bin+2;

figure();
subplot(1,2,1);
q = quiver(Xu(1:binO:ff,1:binO:ff),Yu(1:binO:ff,1:binO:ff),...
    nx(1:binO:end,1:binO:end),ny(1:binO:end,1:binO:end),1);axis equal;axis tight;hold on
q.LineWidth=.5;
q.Color = [0 0 0];
q.ShowArrowHead='off';
axis off; %title(total_count); 
hold off

% figure(FIG+1)
subplot(1,2,2);
[u_x,u_y] = gradient(u);%/dx gradient need to be corrected for the dx
[v_x,v_y] = gradient(v);%/dx
vorticity = (v_x - u_y);%------------------- OPTION1
divV = (u_x + v_y);%----- OPTION2
filtN = 20;
filt = fspecial('gaussian',filtN,filtN);
u1 = imfilter(vorticity, filt);
surf(Xu(1:1:ff,1:1:ff),Yu(1:1:ff,1:1:ff),u1-10);view(2);shading interp;colormap jet;axis equal;axis tight;hold on
load('mycbar.mat')
set(gcf,'Colormap',mycbar); %this works
caxis([-10-max(u1(:)), -10+max(u1(:))]);

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