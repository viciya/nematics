%% LOAD FILES
% --- Description ----
% This code finds shows orientation field for specific stripe
% "get_highest_piv_shear" finds fastest shear flow 
% for a stripes in given width

PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
% PC_path = 'D:\GD\';                         % RSIP notebook

addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

[dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);

%%
[~, filepathPIV, filepathOP] = get_highest_piv_shear(dirPIV, dirOP, indX, Sorted_Orient_width, 400);
px_sizePIV = 0.748;
frame_per_hr = 4;
px2mic = px_sizePIV * frame_per_hr;
%%
for i = 60 %1:numel(imfinfo(filepathOP))
Ang = imread(filepathOP,i); % read Angle file
[Xu,Yu] = meshgrid(1:size(Ang,2),1:size(Ang,1));

if any( Ang(:)>4 ) % check if Ang is in RAD
    Ang = Ang * pi/180;
end

[px, py, ptheta, nx, ny, ntheta] = ...
    fun_get_pn_Defects_newDefectAngle_blockproc(Ang);

figure(22);
% ax2 = subplot(1,2,1);

% plot LIC
LIC = fun_getLIC(cos(Ang),-sin(Ang)); % get LIC from vector matrix
imageplot(LIC,''); hold on

% plot defects
plot_pdefect_ndefect(px, py, ptheta,...
    nx, ny, ntheta,...
    .15,.12,15,7);
hold on

% plot quiver
quiver_orientation(Ang,7)

% plot Schlieren 
schlieren = abs(pi/4-mod(Ang,pi/2));
imshow(schlieren)

view([90 -90])
[filepath,name,ext] = fileparts(filepathOP);
title([name,' / frame: ',num2str(i)], 'Interpreter', 'none')
% save png image 300 dpi resolution
% print('LIC_defects-89_3_60','-dpng','-r300')
%%
% % //////////////////////////////////////////////
% load(filepathPIV);
% dx = px_sizePIV*(x{1}(1,2)-x{1}(1,1));
% ff = 5;
% filt = fspecial('gaussian',ff,ff);
% uu = px2mic*imfilter(u{i}, filt);
% vv = px2mic*imfilter(v{i}, filt);
% [u_x,u_y] = gradient(uu,dx);%/dx gradient need to be corrected for the dx
% [v_x,v_y] = gradient(vv,dx);%/dx
% vorticity = (v_x - u_y);
% filtN = 5;
% filt = fspecial('gaussian',filtN,filtN);
% u1 = imfilter(vorticity, filt)+10;
% sc = size(Ang,1)/size(uu,1);
% step = 2;
% O_len = 1.5;
% 
% % PIV quiver
% % figure(22);
% % ax2 = subplot(1,2,2);
% [Xu,Yu] = meshgrid(1:size(uu,2),1:size(uu,1));
% s1 = surf(sc*Xu(1:step:end,1:step:end),sc*Yu(1:step:end,1:step:end),u1(1:step:end,1:step:end));
% view(2);
% shading interp;axis equal;axis tight;hold on
% 
%     load('mycbar.mat'); caxis([min(u1(:)),max(u1(:))])
% %     colormap(ax2,mycbar)
%     colormap(mycbar)
%     
% q6 = quiver(sc*Xu(1:step:end,1:step:end),sc*Yu(1:step:end,1:step:end),...
%     uu(1:step:end,1:step:end),vv(1:step:end,1:step:end),O_len);
% q6.LineWidth=1; q6.Color = [.0 0 0];
% view([90 -90])
% axis equal;axis tight;
% % /////////////////////////////////////////////
% 
% hold off
% set(gcf,'Color',[1 1 1]);
% % saveas(gcf,['C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\images800\',num2str(i),'.png'])
end 