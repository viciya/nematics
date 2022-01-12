% DEFECT detection and classification

% -----------------  READ ORIENT
[filename, pathname,filterindex] = uigetfile( ...
    '*.tif', 'Pick a file',...
    'C:\Users\vici\Desktop\HT1080\stripe_Orient');

info = imfinfo([pathname,filename]); % Place path to file inside single quotes
Nn = numel(info);
% -----------------  READ PIV
[pfilename, ppathname] = uigetfile( ...
    '*.mat', 'Pick a PIV file',...
    'C:\Users\vici\Desktop\HT1080\stripe_PIV');

load([ppathname,pfilename],'resultslist');
A = resultslist;  % Assign it to a new variable with different name.
clear('resultslist'); % If it's really not needed any longer.

t_shift = 0; % first Orientation frame
px_size = .74;
X = px_size*A{1,1};
Y = px_size*A{2,1};
%% Check in one frame (k)

kk = 100;
dt = 2; % velocity avearage

for k=kk:kk
    k
    clearvars -except k kStep k_count info Nn filename pathname...
        filterindex A projection t_shift deltaTeta rot_u rot_v...
        box s_box dt v X Y
    
    
    qstep = 20; %mine 10 in pixels
    Ang = imread([pathname,filename],k); % k
    Ang = pi/180*Ang;
    [l,w] = size(Ang);
    
    qq = ordermatrixglissant_overlap(Ang,qstep,6);
    im2 = qq < min(qq(:))+0.6;%.2; % make binary image to use regionprops
    s = regionprops('table', im2,'centroid');
    %     figure(5); surf(qq);view(2);view([-90 90]);
    %     shading interp;axis tight;colormap jet;axis equal;
    % --------------------------------------------------------------------
    % %% !------- Find +/-  defects --------- in ONE FRAME  -------!
    % --------------------------------------------------------------------
    r_Circ = 17;
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
    
    for i=1:length(ps_x)
        for j=1:10
            Blank = zeros(size(Blank));
            Blank(round(ps_y(i)):round(ps_y(i))+2*r_Circ,...
                round(ps_x(i)):round(ps_x(i))+2*r_Circ) = TEMP(:,:,j);
            sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
            
            AngTEMP  = Ang.* sBlank;
            AngTEMP_vec(j,1)  = mean(mean(180/pi*AngTEMP(AngTEMP~=0)));
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
        
        plocPsi_vec(i,1)  = mean(mean(180/pi*plocPsi(plocPsi~=0)));
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
    
    for i=1:length(ns_x)
        Blank = zeros(size(Blank));
        Blank(round(ns_y(i)):round(ns_y(i))+2*r_Circ,...
            round(ns_x(i)):round(ns_x(i))+2*r_Circ) = TEMP;
        sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
        nlocPsi  = nPsi.* sBlank;
        
        nlocPsi_vec(i,1)  = mean(mean(180/pi*nlocPsi(nlocPsi~=0)));
    end
    nlocPsi_vec = -nlocPsi_vec/3;
    
    % %% PLOT DEFECTS
    figure(101)
    
    l_len = .1;
    % +1/2 defect
    q2 = quiver(ps_x,ps_y,cosd(plocPsi_vec),sind(plocPsi_vec),l_len);hold on
    q2.LineWidth=3;
    q2.Color = [.8 .8 .6];
    q2.ShowArrowHead = 'off';
    
    % -1/2 defect
    q3 = quiver(ns_x,ns_y,cosd(nlocPsi_vec),sind(nlocPsi_vec),l_len);hold on
    q3.LineWidth=3;
    q3.Color = [.8 .8 .8];
    q3.ShowArrowHead = 'off';
    q4 = quiver(ns_x,ns_y,cosd(nlocPsi_vec+120),sind(nlocPsi_vec+120),l_len);hold on
    q4.LineWidth=3;
    q4.Color = [.8 .8 .8];
    q4.ShowArrowHead = 'off';
    q5 = quiver(ns_x,ns_y,cosd(nlocPsi_vec+240),sind(nlocPsi_vec+240),l_len);hold on
    q5.LineWidth=3;
    q5.Color = [.8 .8 .8];
    q5.ShowArrowHead = 'off';
    % All defects
    p1=plot(s.Centroid(:,1),s.Centroid(:,2),'o', 'MarkerFaceColor',[0 0 1]);hold on
    p1.MarkerSize = 5;
    p1.MarkerEdgeColor= 'none';
    % +1/2 defect
    p2 = plot(ps_x,ps_y,'o','MarkerFaceColor',[0 .5 .1]);hold on
    p2.MarkerSize = 8;
    p2.MarkerEdgeColor= 'none';
    % -1/2 defect
    p3 = plot(ns_x,ns_y,'o','MarkerFaceColor',[.8 .1 0]);hold on
    p3.MarkerSize = 8;
    p3.MarkerEdgeColor= 'none';
    
    % % Orientation
    [Xu,Yu] = meshgrid(1:w,1:l);
    step = 15;
    q = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
        cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)),.7);
    q.LineWidth=1;
    q.Color = [.4 .4 .4];
    q.ShowArrowHead='off';
    
    axis equal;%axis on;%axis tight
    axis([0 size(Ang,2) 0 size(Ang,1)])
    set(gca,'xtick',[]); set(gca,'ytick',[]) 
    
    % % % % % % % % % % % % % % % % % % % % % % % % % %
    % % % % % % % % % % %   PIV   % % % % % % % % % % %
    % % % % % % % % % % % % % % % % % % % % % % % % % %
    ff = 5;
    filt = fspecial('gaussian',ff,ff);
    uu = zeros(size(A{7,k}));
    vv = uu;
    uu = uu + imfilter(A{7,k}, filt);
    vv = vv + imfilter(A{8,k}, filt);
    vstep = 1;
    sc = size(Ang,1)/size(uu,1);
    [X,Y] = meshgrid((1/sc:size(uu,2))*sc,(1/sc:size(uu,1))*sc);
    figure(101)
    q0 = quiver(X(1:vstep:end,1:vstep:end),Y(1:vstep:end,1:vstep:end),...
               uu(1:vstep:end,1:vstep:end),vv(1:vstep:end,1:vstep:end),1.5);
    q0.LineWidth=.5;
    q0.Color = [0 0 0];hold on
    
    [u_x,u_y] = gradient(uu);%/dx gradient need to be corrected for the dx
    [v_x,v_y] = gradient(vv);%/dx
    vorticity = (v_x - u_y);%------------------- OPTION1
    divV = (u_x + v_y);%----- OPTION2
    filtN = 5;
    filt = fspecial('gaussian',filtN,filtN);
    u1 = imfilter(vorticity, filt);
    surf(X(1:vstep:end,1:vstep:end),Y(1:vstep:end,1:vstep:end),u1-10);view(2);shading interp;colormap jet;axis equal;axis tight;hold on
    caxis([-.7*max(u1(:))-10, .7*max(u1(:))-10]);
    axis equal; axis tight;
% %     view([-90 90]); %hold off
%  -----------------------------------------------------------    
   
    hold off
    
%     view([-90 90])
    %     F = getframe(gca);
    %     writeVideo(v,F)
end

% close(v)