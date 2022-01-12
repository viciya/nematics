
%% perform LIC in vector field
[W,L] = size(Ang); % Width Length

if any( Ang(:)>2 ) % chek if Ang is in RAD
    Ang = Ang * pi/180;
end

w=4;
% -----  non random
% M = ones(W); 
% d = 1;
% gap = 2;
% M(1:gap:end,1:gap:end) = -1;
% ------  random
M = randn(W,L);

sigma = 5;
kj=1;
first = 17;
last = first;
for i=2%first:1:last
% w = ; is the length of the convolution (in pixels)

v = zeros(W,L,2);
v(:,:,2) = cos(Ang);% NOTE THAT COS GOES IN TO SECOND LAYER
v(:,:,1) = -sin(Ang);% NOTE THAT -SIN GOES IN TO SECOND LAYER

 % regularity of the vector field
options.bound = 'sym'; % boundary handling
v = perform_blurring(v, sigma, options);
v = perform_vf_normalization(v);
% Now we perform the LIC of an initial noise image along the flow.

% parameters for the LIC
% options.histogram = 'linear'; % keep contrast fixed
options.verb = 0;
options.dt = 1; % time steping
% size of the features
options.flow_correction = 5;
options.niter_lic = 2; % several iterations gives better results
% iterated lic
    options.M0 = M;
    Mlist = perform_lic(v, w, options);
figure(i); subplot(1,last-first+1,kj);
imageplot(Mlist,'');
title(i);
kj=kj+1;
% M = Mlist;    
% for i=1:4
%     options.M0 = M;
%     Mlist{end+1} = perform_lic(v, i*4, options);
% end
% display
% clf; imageplot(Mlist,'',2,2);
end
%%
[Xu,Yu] = meshgrid(1:L,1:W);
hold on
l_len = .05;
MarkerS = 8;
step = 6;
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
% p2 = plot(ps_x,ps_y,'o','MarkerFaceColor',[.8 .1 0]);hold on
% p2.MarkerSize = 5;
% p2.MarkerEdgeColor= 'none';
% 
% q2 = quiver(ps_x,ps_y,cosd(plocPsi_vec),sind(plocPsi_vec),l_len);hold on
% q2.LineWidth=3;
% q2.Color = [.8 .1 0];
% q2.ShowArrowHead = 'off';
% view([-90 90])
hold off
%% make movie of vortexes
v = VideoWriter('C:\Users\vici\Desktop\Carles\HT1080_s16_1to127 with quiver.avi');
v.FrameRate = 30/12*frame_per_hr;
open(v)
z_max = max(abs(vorticity(:)))/30;

i = 100;
for i=1:127

quiver(u_filtered{i,1},v_filtered{i,1},3, 'Color',[0,0, 0]);hold on;

filt_size = 7;
filt = fspecial('gaussian',filt_size,filt_size);
u1 = imfilter(vorticity(:,:,i), filt);

u1 = perform_vf_normalization(u1);
u1 = u1- 3*ones(size(u1(:,:,1)));
% imagesc(u1);colormap jet;hold on
% contour(u1);hold on;
surf(u1);view(2);colormap jet;hold on;shading interp;%colorbar


axis([0 W 0 W -inf inf]);%axis equal;
% caxis([-z_max z_max])
title(['time= ',num2str(i/frame_per_hr,3),' hrs']);
hold off



F = getframe(gcf);
writeVideo(v,F)
end
close(v)
%% make movie of vortexes

okubo_weiss = size((vorticity(:,:,1)));

i = 7;
% for i=i:799
% subplot(1,2,1)
%quiver(u_filtered{i,1},v_filtered{i,1},3, 'Color',[0,0, 0]);hold on;

filt_size = 3;
filt = fspecial('gaussian',filt_size,filt_size);

okubo_weiss = strain(:,:,i)+shear(:,:,i)- vorticity(:,:,i).^2;
u1 = imfilter(okubo_weiss, filt);
u2 = imfilter(vorticity(:,:,i), filt);
% u1 = -1*perform_vf_normalization(u1);
% imagesc(u1);colormap jet;hold on
% contour(u1);hold on;
% subplot(1,2,1)
% surf(u1);view(2);colormap jet;hold on;shading interp
% axis equal;axis tight;hold off
% subplot(1,2,2)
quiver(u_filtered{i,1},v_filtered{i,1},3, 'Color',[0,0, 0]);hold on;
surf(u2);view(2);colormap jet;hold on;shading interp
axis equal;axis tight; colorbar;hold off

%axis([0 W 0 W -z_max z_max]);%axis equal;
title(['time= ',num2str(i/12,3),' hrs']);

% end

%%
m_vort = mean(vorticity(:,:,:),3);
filt_size = 7;
filt = fspecial('gaussian',filt_size,filt_size);

okubo_weiss = strain(:,:,i)+shear(:,:,i)- vorticity(:,:,i).^2;
u1 = imfilter(m_vort, filt);
surf(u1);view(2);colormap jet;hold on;shading interp
mean(m_vort(:))
std(m_vort(:))
