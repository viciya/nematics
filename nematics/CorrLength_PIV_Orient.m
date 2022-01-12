%% NUMBER OF DEFECTS & CORR LENGTH
% run first two cells of correlate_OP_PIV_Defect.m
clearvars -except dirOP dirPIV filepathOP filepathPIV...
    u v x y indX Sorted_Orient_width
i = 56;
folderOP = dirOP(i).folder;
nameOP = dirOP(i).name;
folderPIV = dirPIV(indX(i,2)).folder;
namePIV = dirPIV(indX(i,2)).name;

filepathOP = [folderOP '\' nameOP];
info = imfinfo(filepathOP); % Place path to file inside single quotes
Nn = numel(info);

filepathPIV = [folderPIV '\' namePIV];
load(filepathPIV);
%%


info(1).Width
kk=50;    step = 15; %mine 10 in pixels
dt = 4;
px_size = .748;
pix2mic = px_size*dt;
% Lc_fit = fittype('exp(-x/a)','dependent',{'y'},'independent',{'x'},'coefficients',{'a'});

for k=1:Nn-1
    %%
    %---------------------------------------------------
    % ----------------ORIENT----------------------------
    %---------------------------------------------------
%     k=80
    Ang = imread(filepathOP,k); % k
    [l,w] = size(Ang);
    if any( Ang(:)>2 ) % chek if Ang is in RAD
        Ang=Ang*pi/180;
    end
    AngS = imresize(Ang,.3);
    %     In = .5*(normxcorr2(cos(AngS),cos(AngS))+...
    %         normxcorr2(sin(AngS),sin(AngS)));
    In = normxcorr2(AngS-mean2(AngS),AngS-mean2(AngS));
    %     In = xcorr2(ones(size(AngS)));
    [ymax, xmax] = size (In);
    [XX,YY] = meshgrid (-xmax/2+1/2: xmax/2-1/2, -ymax/2+1/2: ymax/2-1/2);
    r = (0:round(xmax/10))';
    filterAll=zeros(size(XX));
    image_integral=zeros(length(r),1);
    
    %------RADIAL INTEGRATION----------
    for i=1:max(r)
        filterIn = sqrt(XX.^2+YY.^2) < r(i+1); % The filter is 1 inside radius, 0 outside
        filterOut = sqrt(XX.^2+YY.^2) >= r(i); % The filter is 0 inside radius, 1 outside
        running_filter = double(filterIn & filterOut);
        filterAll = running_filter+filterAll;
        running_image = In.*running_filter;
        running_integral = trapz(trapz(running_image))/sum(sum(running_filter));
        image_integral(i) = running_integral;
    end
    corr_Ang(:,k) = image_integral;
    %  subplot(2,1,1)
    r_scaled = 1532/size(AngS,1)*r;
%     pp1 = plot(r_scaled,image_integral,'o','Color', [1-k/Nn, 0, 1-k/Nn]); hold on
%     
% %     ----------LINIAR FITING-----------
%     fl = 10; % number of points for liniar fit
%     % f = fit(r_micron(1:fl),image_integral(1:fl),Lc_fit,'StartPoint',[50],'Lower',[5],'Upper',[500]);Lc_R(k) = f.a;
%     ff = polyfit(r_scaled(1:fl),image_integral(1:fl),1);
%     plot(r_scaled(1:fl),polyval(ff,r_scaled(1:fl)));
%     Lc_R(k,1) = -ff(2)/ff(1);
    
    %---------------------------------------------------
    %--------------PIV----------------------------------
    %---------------------------------------------------
    ff = 7; filt = fspecial('gaussian',ff,ff);
    dx = px_size*(x{k}(1,2)- x{k}(1,1));
    uu = zeros(size(u{k})); vv = uu;
    uu = pix2mic*imfilter(u{k}, filt);
    vv = pix2mic*imfilter(v{k}, filt);
    
    % Vorticity
    [u_x,u_y] = gradient(uu,dx);%/dx gradient need to be corrected for the dx
    [v_x,v_y] = gradient(vv,dx);%/dx
    vorticity = (v_x - u_y);
    ff = 5;
    filt = fspecial('gaussian',ff,ff);
    u1 = imfilter(vorticity, filt); %imagesc(vv)
%  1)   vorticity
    u1 = u1-mean2(u1);  InW = normxcorr2(u1,u1);
%  2)  velocity    
    InV = .5*(normxcorr2(uu,uu)+...
        normxcorr2(vv,vv));

    [ymax, xmax] = size (InV);
    [XX,YY] = meshgrid (-xmax/2+1/2: xmax/2-1/2, -ymax/2+1/2: ymax/2-1/2);
    r = (0:round(xmax/10))';
    filterAll=zeros(size(XX));
    image_integral=zeros(length(r),1);
    
    %------RADIAL INTEGRATION----------
    for i=1:max(r)
        filterIn = sqrt(XX.^2+YY.^2) < r(i+1); % The filter is 1 inside radius, 0 outside
        filterOut = sqrt(XX.^2+YY.^2) >= r(i); % The filter is 0 inside radius, 1 outside
        running_filter = double(filterIn & filterOut);
        filterAll = running_filter+filterAll;
        
        running_image = InV.*running_filter;
        running_integral = trapz(trapz(running_image))/sum(sum(running_filter));
        image_integralV(i) = running_integral;
        
        running_image = InW.*running_filter;
        running_integral = trapz(trapz(running_image))/sum(sum(running_filter));
        image_integralW(i) = running_integral;
    end
corr_V(:,k) = image_integralV;
corr_W(:,k) = image_integralW;
    r_scaled_v = dx*r;
%     pp1=plot(r_scaled_v,image_integral,'o','Color', [1-k/Nn, 1-k/Nn, 0]); hold on
%     
%     %----------LINIAR FITING-----------
%     fl = 5; % number of points for liniar fit
%     % f = fit(r_micron(1:fl),image_integral(1:fl),Lc_fit,'StartPoint',[50],'Lower',[5],'Upper',[500]);Lc_R(k) = f.a;
%     ff = polyfit(r_scaled(1:fl),image_integral(1:fl),1);
%     plot(r_scaled_v(1:fl),polyval(ff,r_scaled_v(1:fl)));
%     Lc_R(k,2) = -ff(2)/ff(1);
end


%%


errorbar(r_scaled, mean(corr_Ang,2), std(corr_Ang,0,2)); hold on
errorbar(r_scaled_v(1:end-1), mean(corr_V,2), std(corr_V,0,2)); 
errorbar(r_scaled_v(1:end-1), mean(corr_W,2), std(corr_W,0,2)); hold off
ylabel('C (r)'); xlabel('r (\mum)');set(gca,'Fontsize',18)
legend('director', 'velocity', 'vorticity')






