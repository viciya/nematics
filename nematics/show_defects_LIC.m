%%
% this code loads external file with defect positions
% and calculates local velocity field around the defects
% note, velocity field is not alighned

PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

[dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);



PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
folder_path = 'Curie\DESKTOP\HT1080\';
set_name = {
    'symm_pDefect_x_y_angle_frame_idx_300.txt',...
    'symm_pDefect_x_y_angle_frame_idx_400.txt',...
    'symm_pDefect_x_y_angle_frame_idx_500.txt',...
    'symm_pDefect_x_y_angle_frame_idx_600.txt',...
    'symm_pDefect_x_y_angle_frame_idx_700.txt',...
    'symm_pDefect_x_y_angle_frame_idx_800.txt',...
    'symm_pDefect_x_y_angle_frame_idx_1000.txt'};
%%
px_sizeOP = 3* .74;
edge = 200;
box = 40;

wset = [300,400,500,600,700,800,1000];
x_pos = [];
n = 5;
i = 10;
k = 50;

Sw = wset(n); % selectd width
dw = .05*Sw; % define delta
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);



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

%         filepathPIV = [dirPIV(Range(i)).folder, '\', dirPIV(indX(Range(i),2)).name];
%         piv = load(filepathPIV);

[iloc_u, iloc_v, ix1, iy1, ix2, iy2, icount] = empty_set(box);

%         get defect positions and angles
ang = imread(filepathOP,k);
if any( ang(:)>2 ) % check if Ang is in RAD
    ang = ang * pi/180;
end
%             [ps_x,ps_y,ptheta] = get_p_defect_symm(ang);
%             [ps_x, ps_y, ptheta,ns_x, ns_y, ntheta] = fun_get_pn_Defects_newDefectAngle(ang);
            [ps_x, ps_y, ptheta,ns_x, ns_y, ntheta] = fun_get_pn_Defects_newDefectAngle_blockproc(ang);
% ps_x  = px(file_in_range==i & frame==k);
% ps_y  = py(file_in_range==i & frame==k);
% ptheta  = theta(file_in_range==i & frame==k);

% [ang_x,ang_y] = meshgrid(1:size(ang,2),1:size(ang,1));
%             u_interp = interp2(piv.x{k},piv.y{k},piv.u{k},ang_x,ang_y,'cubic',0);
%             v_interp = interp2(piv.x{k},piv.y{k},piv.v{k},ang_x,ang_y,'cubic',0);
% %%
figure;
imageplot(fun_getLIC(cos(ang),-sin(ang)),''); hold on
% fun_defectDraw(ps_x, ps_y, ptheta);
fun_defectDraw(ps_x, ps_y, ptheta,ns_x, ns_y, ntheta);

% colormap(ax1,gray)
view([-90 90]);
%%



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

[ix, iy, iq,~, ~, ~] = ...
    fun_get_pn_Defects_newDefectAngle(rot90(ang,2));
x = [x;size(ang,2)-ix];
y = [y;size(ang,1)-iy];
q = [q;iq+180];
end

function [u, v, x1,y1,x2,y2,count] = empty_set(box)
s_box = floor(sqrt(box^2/2));
u=zeros(2*box+1);%zeros(2*s_box+1);
v=u;
x1=u;
y1=u;
x2=u;
y2=u;
count=[];
end

function show_PIV(u,v,filt_v,bin)
u = u - mean2(u);
v = v - mean2(v);
[l, w] = size(u);
[Xu,Yu] = meshgrid(1:w,1:l);
ff = l;

figure()
% subplot(1,2,2);
[u_x,u_y] = gradient(u);%/dx gradient need to be corrected for the dx
[v_x,v_y] = gradient(v);%/dx
vorticity = (v_x - u_y);%------------------- OPTION1
divV = (u_x + v_y);%----- OPTION2
filtN = filt_v;
filt = fspecial('gaussian',filtN,filtN);
u1 = imfilter(vorticity, filt);
surf(Xu,Yu,u1-10);view(2);shading interp;colormap jet;axis equal;axis tight;hold on
load('mycbar.mat')
set(gcf,'Colormap',mycbar); %this works
caxis([-10-max(u1(:)/2), -10+max(u1(:))/2]);

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

function show_PIV_Orent(u,v,nx,ny,filt_v)
% u = u - mean2(u);
% v = v - mean2(v);
[l, w] = size(u);
[Xu,Yu] = meshgrid(1:w,1:l);
ff = l;

% nx1(ny1<0)=-nx1(ny1<0);
% ny1(nx1<0)=-ny1(nx1<0);
% nx = nx1+nx2;
% ny = ny1+ny2;

bin = round(l/20);
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
filtN = filt_v;
filt = fspecial('gaussian',filtN,filtN);
u1 = imfilter(vorticity, filt);
surf(Xu(1:1:ff,1:1:ff),Yu(1:1:ff,1:1:ff),u1-10);view(2);shading interp;colormap jet;axis equal;axis tight;hold on
load('mycbar.mat')
set(gcf,'Colormap',mycbar); %this works
caxis([-10-max(u1(:)/2), -10+max(u1(:))/2]);

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