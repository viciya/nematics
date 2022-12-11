  %%
folder = 'C:\Users\USER\Downloads\BEER\March 1st 100fps 40X 50-50 1um gap\';
dir_info_orient  = dir([folder,'\orient\*.tif']);
[~, reindex] = sort_nat({dir_info_orient.name});
dir_info_orient = dir_info_orient(reindex);

% 'C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\Orient\Orient_1_X1.tif'

load([folder, '\SUMMARY.mat'])
defNum = defNum(reindex,:);


% vel_filepath = 'C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\OptFlow\test.mat'
%%
box = 150;
s_box = floor(sqrt(box^2/2));
rot_u = zeros(2*s_box+1);

sub_PIV_u = zeros(2*s_box+1);
sub_PIV_v = zeros(2*s_box+1);

total_count = 0;
for n = 1:100%500
    orient_filepath = [dir_info_orient(n).folder '\' dir_info_orient(n).name];
    ang = imread(orient_filepath);
    [l,w] = size(ang);

    [f,name,ext] = fileparts(orient_filepath);
    dir_info_flow  = dir([folder, '\*\',name(8:end),'.mat']);
    vel_filepath = [dir_info_flow.folder '\' dir_info_flow.name];
    load(vel_filepath);

%     [ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec] = fun_get_pn_Defects_newDefectAngle(ang);
    [ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec] = deal(defNum{n,:});
%%
plus = true;
if plus
    s_x = ps_x;
    s_y = ps_y;
    locPsi_vec = plocPsi_vec;
else
    s_x = ns_x;
    s_y = ns_y;
    locPsi_vec = nlocPsi_vec;
end

% box = 200;
% s_box = floor(sqrt(box^2/2));
% i = 10;
% rectangle('Position', [ps_x(i) ps_x(i) s_box s_box])

rot_u = zeros(2*s_box+1);
rot_v = zeros(2*s_box+1);
rot_nx1 = zeros(2*s_box+1);
rot_ny1 = zeros(2*s_box+1);
rot_nx2 = zeros(2*s_box+1);
rot_ny2 = zeros(2*s_box+1);
sub_nx1 = rot_nx1;
sub_ny1 = sub_nx1;
sub_nx2 = sub_nx1;
sub_ny2 = sub_nx1;

i_count = 0;
for i = 1:length(s_x)
    % ---  check if box around the defects fits in the PIV field  -----
    if (s_x(i)-box)>1 && (s_y(i)-box)>1 && (s_x(i)+box)<w && (s_y(i)+box)<l
        %             take PIV sub-window around the defect
        su = u(round(s_y(i))-box:round(s_y(i))+box,...
            round(s_x(i))-box:round(s_x(i))+box);
        sv = v(round(s_y(i))-box:round(s_y(i))+box,...
            round(s_x(i))-box:round(s_x(i)+box));
        %             take ORIENT sub-window around the defect
        sang = ang(round(s_y(i))-box:round(s_y(i))+box,...
            round(s_x(i))-box:round(s_x(i))+box);
        snx = cos(sang);
        sny = -sin(sang);

        %             rotate each PIV vector by angle of the defect
        suR = cosd(locPsi_vec(i,1))*su + sind(locPsi_vec(i,1))*sv;
        svR = -sind(locPsi_vec(i,1))*su + cosd(locPsi_vec(i,1))*sv;
        %             rotate each ORIENT vector by angle of the defect
        snxR = cosd(locPsi_vec(i,1))*snx + sind(locPsi_vec(i,1))*sny;
        snyR = -sind(locPsi_vec(i,1))*snx + cosd(locPsi_vec(i,1))*sny;

        %             rotate whole PIV field by angle of the defect
        suRR = imrotate(suR,locPsi_vec(i,1),'bilinear','crop');
        svRR = imrotate(svR,locPsi_vec(i,1),'bilinear','crop');%,'crop'
        %             rotate whole ORIENT field by angle of the defect
        snxRR = imrotate(snxR,locPsi_vec(i,1),'bilinear','crop');
        snyRR = imrotate(snyR,locPsi_vec(i,1),'bilinear','crop');%,'crop'

        %           Average PIV fields
        rot_u  = rot_u + suRR(box-s_box:box+s_box,box-s_box:box+s_box);
        rot_v  = rot_v + svRR(box-s_box:box+s_box,box-s_box:box+s_box);

        % Average ORIENT fields (averaging of nematic field involves flipping, see plot of sub_nx/y1 VS sub_nx/y2)
        nx_temp = snxRR(box-s_box:box+s_box,box-s_box:box+s_box);
        ny_temp = snyRR(box-s_box:box+s_box,box-s_box:box+s_box);
%         nx_temp1=nx_temp;nx_temp2=nx_temp;
%         ny_temp1=ny_temp;ny_temp2=ny_temp;
%         nx_temp1(nx_temp<0) = -nx_temp(nx_temp<0);
%         ny_temp1(nx_temp<0) = -ny_temp(nx_temp<0);
%         nx_temp2(ny_temp<0) = -nx_temp(ny_temp<0);
%         ny_temp2(ny_temp<0) = -ny_temp(ny_temp<0);
%         
%         rot_nx1  = rot_nx1 + nx_temp1;
%         rot_ny1  = rot_ny1 + ny_temp1;
%         rot_nx2  = rot_nx2 + nx_temp2;
%         rot_ny2  = rot_ny2 + ny_temp2;            
% ////////////////////////////////////////////////////////
        qxx = nx_temp .* nx_temp - ny_temp .* ny_temp;
        qxy = 2* (nx_temp .* ny_temp);
        qyy = -qxx;
        qyx = qxy;

        rot_nx1 = rot_nx1 + qxx;
        rot_ny1 = rot_ny1 + qxy;
        rot_nx2 = rot_nx2 + qyx;
        rot_ny2 = rot_ny2 + qyy;
% //////////////////////////////////////////////////////// 
PLOT_ORIENT = true;
if PLOT_ORIENT
    plot_orientation(snx,sny)
    plot_orientation(snxRR,snyRR)
    title(locPsi_vec(i,1));
end

        i_count = i_count+1;
    end
end


sub_nx1 = sub_nx1 + rot_nx1;
sub_ny1 = sub_ny1 + rot_ny1;
sub_nx2 = sub_nx2 + rot_nx2;
sub_ny2 = sub_ny2 + rot_ny2;
sub_PIV_u = sub_PIV_u + rot_u/i_count;
sub_PIV_v = sub_PIV_v + rot_v/i_count;

total_count = total_count + i_count;
end
[Xu,Yu] = meshgrid(1:w,1:l);
%%
sub_nx11 = sub_nx1;
sub_ny11 = sub_ny1;
sub_nx22 = sub_nx2;
sub_ny22 = sub_ny2;

sub_nx11(sub_ny11<0) = -sub_nx11(sub_ny11<0);
sub_ny11(sub_nx11<0) = -sub_ny11(sub_nx11<0);

sub_nx = sub_nx11 + sub_nx22;
sub_ny = sub_ny11 + sub_ny22;

figure(121)
% subplot(ceil(((last-start)/kStep).^.5),floor(((last-start)/kStep).^.5),k_count-1)

ff=2*s_box+1;
% quiver(Xu(1:bin:ff,1:bin:ff),Yu(1:bin:ff,1:bin:ff),...
%     sub_PIV_u(1:bin:end,1:bin:end),sub_PIV_v(1:bin:end,1:bin:end),2);axis equal;axis tight; hold on
% Orientation
binO = 5;
q = quiver(Xu(1:binO:ff,1:binO:ff),Yu(1:binO:ff,1:binO:ff),...
    sub_nx(1:binO:end,1:binO:end),sub_ny(1:binO:end,1:binO:end),1);axis equal;axis tight;hold on
q.LineWidth=.5;
q.Color = [0 0 0];
q.ShowArrowHead='off';
p1 = plot(round(ff/2),round(ff/2),'o','MarkerFaceColor',[0 .5 .1]);
p1.MarkerSize = 10;
p1.MarkerEdgeColor= 'none';
axis off;
% title(k);
hold off


figure(222);
[u_x,u_y] = gradient(sub_PIV_u);%/dx gradient need to be corrected for the dx
[v_x,v_y] = gradient(sub_PIV_v);%/dx
vorticity = (v_x - u_y);%------------------- OPTION1
divV = (u_x + v_y);%----- OPTION2
filtN = 10;
filt = fspecial('gaussian',filtN,filtN);
u1 = imfilter(vorticity, filt);
surf(Xu(1:1:ff,1:1:ff),Yu(1:1:ff,1:1:ff),u1-10);
view(2); shading interp; colormap jet; axis equal; axis tight; hold on
load('mycbar.mat')
set(gcf,'Colormap',mycbar); %this works
caxis([-max(u1(:))/2-10, max(u1(:))/2-10]);


bin = round(5*box/100);
q = quiver(Xu(1:bin:ff,1:bin:ff),Yu(1:bin:ff,1:bin:ff),...
    sub_PIV_u(1:bin:end,1:bin:end),sub_PIV_v(1:bin:end,1:bin:end),1);
axis equal; axis tight; hold on
q.LineWidth=1;
q.Color = [0 0 0];

p2 = plot3(round(ff/2),round(ff/2),40,'o','MarkerFaceColor',[0 0 0]);
p2.MarkerSize = 10;
p2.MarkerEdgeColor= 'none';

figure(333);
[u_x,u_y] = gradient(sub_PIV_u);%/dx gradient need to be corrected for the dx
[v_x,v_y] = gradient(sub_PIV_v);%/dx
vorticity = (v_x - u_y);%------------------- OPTION1
divV = (u_x + v_y);%----- OPTION2
filtN = filtN;
filt = fspecial('gaussian',filtN,filtN);
u1 = imfilter(divV, filt);
surf(Xu(1:1:ff,1:1:ff),Yu(1:1:ff,1:1:ff),u1-10);
view(2); shading interp; colormap jet; axis equal; axis tight; hold on
load('mycbar.mat')
set(gcf,'Colormap',mycbar); %this works
caxis([-max(u1(:))/2-10, max(u1(:))/2-10]);


bin = round(5*box/100);
q = quiver(Xu(1:bin:ff,1:bin:ff),Yu(1:bin:ff,1:bin:ff),...
    sub_PIV_u(1:bin:end,1:bin:end),sub_PIV_v(1:bin:end,1:bin:end),1);
axis equal; axis tight; hold on
q.LineWidth=1;
q.Color = [0 0 0];

p2 = plot3(round(ff/2),round(ff/2),40,'o','MarkerFaceColor',[0 0 0]);
p2.MarkerSize = 10;
p2.MarkerEdgeColor= 'none';

figure(444);
u1 = imfilter(sub_PIV_u, 10);
surf(u1);
view(2); shading interp; colormap jet; axis equal; axis tight; hold on
load('mycbar.mat')
set(gcf,'Colormap',mycbar); %this works
caxis([-max(u1(:)), max(u1(:))]);

%%

% r_circ = 20; % this value could be diffrent from the one used in defect classification
% s_circ = zeros(size(ang));
% blank = zeros(size(ang)+[2*r_circ, 2*r_circ]); %imagesc(blank); axis equal
% [xx,yy] = meshgrid(-r_circ:r_circ, -r_circ:r_circ);
% temp = sqrt(xx.^2+yy.^2) < r_circ; % white circle in black rectangle
%
% loc_u_vec = zeros(length(ps_x),1);
% loc_v_vec = zeros(length(ps_x),1);
% for i=1:length(ps_x)
%     blank = zeros(size(blank));
%
%     blank(round(ps_y(i)):round(ps_y(i))+2*r_circ,...
%         round(ps_x(i)):round(ps_x(i))+2*r_circ) = temp;
%     s_blank = blank(r_circ+1:end-r_circ, r_circ+1:end-r_circ);
%
%     imshow(s_blank);
%
%     loc_u  = u.* s_blank;
%     loc_v  = v.* s_blank;
%     loc_u(loc_u==0) = NaN;
%     loc_v(loc_v==0) = NaN;
%     loc_u_vec(i,1)= nanmean(nanmean(loc_u));
%     loc_v_vec(i,1)= nanmean(nanmean(loc_v));
% end

