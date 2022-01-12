function [ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec] = fun_get_pn_Defects_newDefectAngle_blockproc(Ang, r2_Circ)
if nargin<2
    r2_Circ = 10; % same asr r_Circ    
end

if any( Ang(:)>4 ) % check if Ang is in RAD
    Ang = Ang * pi/180;
end

qstep = 15;

fun = @(B) ...
    sqrt((sum(sum(cos(2*B.data)))/(size(B.data,1)*size(B.data,2)))^2 ...
    +(sum(sum(sin(2*B.data)))/(size(B.data,1)*size(B.data,2)))^2);

% qq = blockproc(Ang,[5 5],fun,'PadMethod','replicate','UseParallel',true);
% use [7,7] for HBEC  %use [5,5] for HT1080 
qq = blockproc(Ang,[7,7],fun,'PadMethod','replicate'); 
scale = size(Ang,1)/size(qq,1);
im2 = qq < .91;%.78;%.2; % make binary image to use regionprops

s = regionprops('table', im2,'centroid');
r_Circ = 10;
s_x = scale * s.Centroid(:,1);
s_y = scale * s.Centroid(:,2);

TEMP = zeros(2*r_Circ+1, 2*r_Circ+1,10);
Blank = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]);

[XX1,YY1] = meshgrid(-r_Circ:r_Circ, -r_Circ:r_Circ);
AngTEMP_vec = zeros(10,1);

for j=1:10
    TEMP(:,:,j) = sqrt(XX1.^2+YY1.^2) < r_Circ ...
        & atan2(YY1, XX1)>= (j-1)*pi/5-pi ...
        & atan2(YY1, XX1)< j*pi/5-pi;
end
pcount = 1;
ncount = 1;

for ii=1:length(s_x)%-1
    try    
    for j=1:10
        Blank = zeros(size(Blank));
        Blank(round(s_y(ii)):round(s_y(ii))+2*r_Circ,...
            round(s_x(ii)):round(s_x(ii))+2*r_Circ) = TEMP(:,:,j);
        sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
%         disp([ii,j]);
        AngTEMP  = Ang.* sBlank;
        AngTEMP(AngTEMP==0) = NaN;
%         AngTEMP_vec1(j,1)  = nanmean(nanmean(180/pi*AngTEMP));
        
        TF = ~isnan(AngTEMP);
        AngTEMP_vec(j,1)  = 180/pi*circ_mean(AngTEMP(TF));        
    end
    % +/- 1/2 defect characterization
    pos_neg = (AngTEMP_vec(2:end)- AngTEMP_vec(1:end-1))>0;
    if sum(pos_neg)<=4
        pDefect_Centroid(pcount,:) = [s_x(ii),s_y(ii)];
        pcount = pcount+1;
    elseif sum(pos_neg)>=5
        nDefect_Centroid(ncount,:) = [s_x(ii),s_y(ii)];
        ncount = ncount+1;
    end
    catch
    end
end
% +1/2 defect angle------------------------------------------------------
if pcount>1
    px = cos(Ang);
    py = -sin(Ang);
    Qxx = (px.*px - 1/2);
    Qxy = (px.*py);
    Qyx = (py.*px);
    Qyy = (py.*py - 1/2);

    [dxQxx,~] = gradient(Qxx);
    [dxQxy,dyQxy] = gradient(Qxy);
    [~,dyQyy] = gradient(Qyy);

    pPsi = atan2(dxQxy+dyQyy, dxQxx+dyQxy);


    ps_x = pDefect_Centroid(:,1);
    ps_y = pDefect_Centroid(:,2);
    
    r_Circ = r2_Circ; %Defect angle
    
    Blank1 = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]);
    
    [XX1,YY1] = meshgrid(-r_Circ:r_Circ,-r_Circ:r_Circ);
    TEMP1(:,:) = sqrt(XX1.^2+YY1.^2) < r_Circ;
    
    for ii=1:length(ps_x)
        Blank1 = zeros(size(Blank1));
        Blank1(round(ps_y(ii)):round(ps_y(ii))+2*r_Circ,...
            round(ps_x(ii)):round(ps_x(ii))+2*r_Circ) = TEMP1;
        sBlank1 = Blank1(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
        plocPsi  = pPsi.* sBlank1;
        plocPsi(plocPsi==0) = NaN;
        
        loc_dxQxx = dxQxx.* sBlank1;
        loc_dxQxy = dxQxy.* sBlank1;
        loc_dyQxy = dyQxy.* sBlank1;
        loc_dyQyy = dyQyy.* sBlank1;
        plocPsi_vec(ii,1) = 180/pi*atan2(mean(loc_dxQxy(loc_dxQxy~=0))+mean(loc_dyQyy(loc_dyQyy~=0)),...
            mean(loc_dxQxx(loc_dxQxx~=0))+ mean(loc_dyQxy(loc_dyQxy~=0)));
        
        
        %     TF = ~isnan(plocPsi);
        %     plocPsi_vec(ii,1)  = 180/pi*circ_mean(plocPsi(TF));
    end
    else
        ps_x=[];
        ps_y=[];
        plocPsi_vec=[];    
end


% -1/2 defect angle------------------------------------------------------
if ncount>1
    
    px = cos(Ang);
    py = sin(Ang);
    Qxx = (px.*px - 1/2);
    Qxy = (px.*py);
    Qyx = (py.*px);
    Qyy = (py.*py - 1/2);
    
    [dxQxx,~] = gradient(Qxx);
    [dxQxy,dyQxy] = gradient(Qxy);
    [~,dyQyy] = gradient(Qyy);
    pPsi = atan2((dxQxy+dyQyy),(dxQxx+dyQxy));
    
    ns_x = nDefect_Centroid(:,1);
    ns_y = nDefect_Centroid(:,2);
    
    Blank1 = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]);

    [XX1,YY1] = meshgrid(-r_Circ:r_Circ,-r_Circ:r_Circ);
    TEMP1(:,:) = sqrt(XX1.^2+YY1.^2) < r_Circ;    
    
    for ii=1:length(ns_x)
        Blank1 = zeros(size(Blank1));
        Blank1(round(ns_y(ii)):round(ns_y(ii))+2*r_Circ,...
            round(ns_x(ii)):round(ns_x(ii))+2*r_Circ) = TEMP1;
        sBlank1 = Blank1(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
        plocPsi  = pPsi.* sBlank1;
        plocPsi(plocPsi==0) = NaN;
        %     nlocPsi_vec1(ii,1)  = nanmean(nanmean(180/pi*plocPsi));
        
        TF = ~isnan(plocPsi);
        nlocPsi_vec(ii,1)  = 180/pi*circ_mean(plocPsi(TF));
    end
    nlocPsi_vec = -nlocPsi_vec/3;
else
    ns_x=[];
    ns_y=[];
    nlocPsi_vec=[];
end
end