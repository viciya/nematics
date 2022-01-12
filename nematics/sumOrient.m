function [av_Ang, av_Std_circ] = sumOrient(filepathOP)
%% -------------------------------------------------------
% i = 4;
% 
% Sw = 600; % selectd width
% dw = .1*Sw; % define delta
% Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);
% FIG = 100
% 
% ff = 3;
% filt = fspecial('gaussian',ff,ff);
% 
% filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
% filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];
% ------------------------------------------------------

info = imfinfo(filepathOP); % Place path to file inside single quotes
Nn = numel(info);

Ang3 = zeros(info(1).Height,info(1).Width,Nn);

for k=1:Nn
    Ang = imread(filepathOP,k); % k
    [l,w] = size(Ang);
    
    if any( Ang(:)>2 ) % chek if Ang is in RAD
        Ang=Ang*pi/180;
    end
    
    Ang3(:,:,k)=Ang;
end
Ang3(Ang3<0) = Ang3(Ang3<0)+pi;
[av_Ang, ~, ~] = circ_mean(Ang3, [], 3);
[av_Std_ang, av_Std_circ] = circ_std(Ang3,[], [], 3);

% ----------  plot profile -------------
% figure(FIG);subplot(1, 2, 1);%cla 
% % errorbar(linspace(-Sw/2,Sw/2,size(av_Ang,2)),circ_mean(av_Ang)*180/pi,circ_mean(av_Std_circ)*180/pi); hold on
% [hl, hp] = boundedline(...
%     linspace(-Sw/2,Sw/2,size(av_Ang,2)),...
%     circ_mean(av_Ang)*180/pi,...
%     circ_mean(av_Std_circ)*180/pi,'alpha'); %/(size(Ang3,3))
% set(gca,'Fontsize',18);
% ylabel('$ Angle\ (deg) $','Interpreter','latex','FontSize',24);
% xlabel('$ Width\ (\mu m) $','Interpreter','latex','FontSize',24);
% axis square; axis tight; 
% % %%
% % ----------  plot STD map -------------
% figure(FIG);subplot(1, 2, 2);cla
% imagesc(av_Std_circ);
% axis equal; axis tight; 
% load('mycbar.mat')
% set(gcf,'Colormap',mycbar); %this works
% hold on
% 
% % ----------  plot directors -------------
% [l,w] = size(av_Ang);
% [X,Y] = meshgrid(1:w,1:l);
% step = 10;
% len = .6;
% q = quiver(X(1:step:end,1:step:end),Y(1:step:end,1:step:end),...
%         cos(av_Ang(1:step:end,1:step:end)),...
%         sin(av_Ang(1:step:end,1:step:end)),len);
% q.Color = [0 0 0];
% q.ShowArrowHead='off';
% q.LineWidth=1;
% hold off
% % set(gca,'View',[-90 90])
end
