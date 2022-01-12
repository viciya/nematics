% DEFECT detection and classification
% RUN first cell of "correlate_OP_PIV_Defect.mat" to get missing variables
i = 73;
folderOP = dirOP(i).folder;
nameOP = dirOP(i).name;
folderPIV = dirPIV(indX(i,2)).folder;
namePIV = dirPIV(indX(i,2)).name;
%%
filepathOP = [folderOP '\' nameOP];
info = imfinfo(filepathOP); % Place path to file inside single quotes
Nn = numel(info);

filepathPIV = [folderPIV '\' namePIV];
load(filepathPIV);

t_shift = 0; % first Orientation frame
%% Check in one frame (k)
clearvars projection deltaTeta
ff = 5;
filt = fspecial('gaussian',ff,ff);

pix2mic = .74;
k_count =1;
total_count = 0;

dt = 1; % velocity avearage

start= 50; kStep = 1; last=start+5%Nn-1;%+40;
for k=start:kStep:last
    try
    k
    clearvars -except k kStep k_count info Nn filename pathname...
        filterindex pix2mic A projection t_shift deltaTeta rot_u rot_v...
        box s_box dt start last total_count filt...
        sub_PIV_u sub_PIV_v sub_nx1 sub_ny1 sub_nx2 sub_ny2...
        u v x y filepathOP filepathPIV Sorted_Orient_width Xu Yu dPsi...
        indX dirOP dirPIV
    
    qstep = 15;%6; %mine 10 in pixels
    Ang = imread(filepathOP,k); % k
    [l,w] = size(Ang);
    
    if any( Ang(:)>2 ) % chek if Ang is in RAD
        Ang=Ang*pi/180;
    end
   
    q = ordermatrixglissant_overlap(Ang,qstep,3);
    im2 = q < min(q(:))+.45;%0.6;%.2; % make binary image to use regionprops
    s = regionprops('table', im2,'centroid');
    % --------------------------------------------------------------------
    % %% !------- Find +/-  defects --------- in ONE FRAME  -------!
    % --------------------------------------------------------------------
    r_Circ = 10;
    ps_x = s.Centroid(:,1);
    ps_y = s.Centroid(:,2);
    s_Circ = zeros(size(Ang));
    TEMP = zeros(2*r_Circ+1,2*r_Circ+1,10);
    Blank = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]); %imagesc(TEMP); axis equal
    
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
    
    for i=1:length(ps_x)-1
        for j=1:10
            Blank = zeros(size(Blank));
            Blank(round(ps_y(i)):round(ps_y(i))+2*r_Circ,...
                round(ps_x(i)):round(ps_x(i))+2*r_Circ) = TEMP(:,:,j);
            sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
            
            AngTEMP  = Ang.* sBlank;  %imagesc(AngTEMP); axis equal 
            AngTEMP(AngTEMP==0) = NaN;
            AngTEMP_vec(j,1)  = nanmean(nanmean(180/pi*AngTEMP));
        end
        
        % +/- 1/2 defect characterization
        pos_neg = (AngTEMP_vec(2:end)- AngTEMP_vec(1:end-1))>0;
        if sum(pos_neg)<= 4
            pDefect_Centroid(pcount,:)=s.Centroid(i,:);
            pcount = pcount+1;
        elseif sum(pos_neg)>= 5
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
        %         imagesc(plocPsi); axis equal
        %         pause(.2)
        plocPsi(plocPsi==0) = NaN;
        plocPsi_vec(i,1)  = nanmean(nanmean(180/pi*plocPsi));
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
        %         imagesc(sBlank)
        %         pause(.2)
        nlocPsi(plocPsi==0) = NaN;
        nlocPsi_vec(i,1)  = nanmean(nanmean(180/pi*nlocPsi));
    end
    nlocPsi_vec = -nlocPsi_vec/3;
    
    % --------------------------PIV import ---------------------------------    
    u_filtered = u(k);
    v_filtered = v(k);    
    uu = zeros(size(u_filtered{1,1}));
    vv = uu;
    % ----- averaging of velocity fields (dt=1: cancels averaging) -----
    for t=1:dt
        uu = uu + imfilter(u_filtered{t,1}, filt);
        vv = vv + imfilter(v_filtered{t,1}, filt);
    end
    
    uu = uu/dt;
    vv = vv/dt;
    sc = size(Ang,1)/size(uu,1);
    
    [Xorig,Yorig] = meshgrid((1/sc:size(uu,2))*sc,(1/sc:size(uu,1))*sc);
    [Xu,Yu] = meshgrid(1:w,1:l);
    u_interp = interp2(Xorig,Yorig,uu,Xu,Yu,'cubic',0);
    v_interp = interp2(Xorig,Yorig,vv,Xu,Yu,'cubic',0);
    
%  Calculate the difference between velocity and depect angle   
    r_Circ = 6; % this value could be diffrent from the one used in defect classification
    s_Circ = zeros(size(Ang));
    Blank = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]); %imagesc(TEMP); axis equal
    [XX1,YY1] = meshgrid(-r_Circ:r_Circ,-r_Circ:r_Circ);
    TEMP1(:,:) = sqrt(XX1.^2+YY1.^2) < r_Circ;
    for i=1:length(ps_x)
        Blank = zeros(size(Blank));
        Blank(round(ps_y(i)):round(ps_y(i))+2*r_Circ,...
            round(ps_x(i)):round(ps_x(i))+2*r_Circ) = TEMP1;
        sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
        su  = u_interp.* sBlank;
        sv  = v_interp.* sBlank;
        su(su==0) = NaN; sv(sv==0) = NaN;
        vel_dir = atan(nanmean(nanmean(su))/nanmean(nanmean(sv)))*180/pi;
     dPsi{k,1}(i,1) = plocPsi_vec(i,1)- vel_dir;
%      dPsi{k,1}(i,1) = vel_dir;
    end    
    
% figure(5); 
% % subplot(1,2,1);imagesc(q);axis equal; axis tight;hold on
% % % Orientation
% % 
% %  hold off
% % subplot(1,2,2);
% imagesc(q); axis equal; axis tight;hold on
% p1=plot(s.Centroid(:,1),s.Centroid(:,2),'o', 'MarkerFaceColor',[0 0 0]);
% p1.MarkerSize = 6;
% p1.MarkerEdgeColor= 'none';
% step = 6;
% q1 = quiver(Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
%     cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)),.7);
% q1.LineWidth=1;
% q1.Color = [.4 .4 .4];
% q1.ShowArrowHead='off';
% axis equal; axis tight; 
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
%     


k_count=k_count+1;    
    catch ME
        disp(['skipped:', num2str(k)]);
        disp(['Massege: ', ME.message]);
    end
end

dPsi = dPsi(~cellfun('isempty',dPsi));
mdPsi = cell2mat(dPsi(start:k,1));
% mdPsi = dPsi{k,1};
figure(111); polarhistogram(mdPsi*pi/180,180);hold on
figure(222); histogram(mdPsi,180); hold on