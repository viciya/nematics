%%
% note that vortex positions (vertPos) has wrong units scalling

% dirPIV = dir(['V:\HT1080_small_stripes_glass_22112017\CROPPED\PIV_DATs'  '\*.mat']);
dirPIV = dir(['C:\Users\victo\Google Drive\DATA\HT1080\PIV_DATs'  '\*.mat']);
px_sizePIV = .74;
mVorticity = [];
CorrArray = [];
fiG =30;
count = 1;
ww = 90;
dw = 8;
for ww=200:50:200
for i = 1:1%size(dirPIV,1)
    % -------------------------------------------------------------------------
    %--------------------PIV---------------------------------------------------
    ff = 1; filt = fspecial('gaussian',ff,ff);
    filepathPIV = [dirPIV(i).folder '\' dirPIV(i).name];
    load(filepathPIV);
    X = px_sizePIV*x{1,1};
    
%     if X(end)<ww+dw && X(end)>ww-dw
            if X(end)>250
        Y = px_sizePIV*y{2,1};
        u_profile = zeros(1,size(X,2));
        v_profile = zeros(1,size(X,2));
        u_std = zeros(1,size(X,2));
        v_std = zeros(1,size(X,2));
        
        for k=1:1%size(x,1)
            u_profile = u_profile + mean(u{k});
            v_profile = v_profile + mean(v{k});
            uu = zeros(size(u(k))); vv = uu;
            uu = imfilter(u{k}, filt);
            vv = imfilter(v{k}, filt);
%             uu = 100*imfilter(rand(size(x{k}))-.5, filt);
%             vv = 100*imfilter(rand(size(x{k}))-.5, filt);            
            dx = px_sizePIV*(x{1}(1,2)-x{1}(1,1));
            [u_x,u_y] = gradient(uu,dx);%/dx gradient need to be corrected for the dx
            [v_x,v_y] = gradient(vv,dx);%/dx
%             vorticity = (uu.^2 + vv.^2);            
            vorticity = (v_x - u_y);
            vorticityX = xcorr2(vorticity- mean2(vorticity),vorticity- mean2(vorticity))...
                /xcorr2(ones(size(vorticity)),ones(size(vorticity)));
%             mVorticity = mean(vorticity',1)- mean2(vorticity);
%             CorrArray = vertcat(CorrArray,autocorr(mVorticity,30,[],3));
            count = count+1;
        end
    end
end
figure(fiG+2)
% plot(Y(1:size(CorrArray,2),1)-Y(1,1), nanmean(CorrArray),'g'); hold on
% plot(mean(vorticityX',1)); 
plot(vorticityX(:,round(end/2))/(max(vorticityX(:)))); hold on
plot(vorticityX(round(end/2),:)/(max(vorticityX(:)))); 
% hold on
% figure(fiG+1)
% surf(vorticityX);
% shading interp;colormap jet;axis tight;hold on%axis equal;
% view(2); colorbar
% caxis([min(u1(:))/1-10, -min(u1(:))/1-10]);
end
% plot(median(CorrArray)); hold off
% axis([0 60 -inf inf])
%%
% Mdl = arima('AR',{0.75,0.15},'SAR',{0.9,-0.5,0.5},...
%     'SARLags',[12,24,36],'MA',-0.5,'Constant',2,...
%     'Variance',1);
% rng(1); % For reproducibility
% y = rand(1,100);
% 
% 
% figure
% autocorr(y)

figure(fiG)
yy = rand(1,100);
yyCorr = autocorr(yy,50,[],3);
plot(Y(1:size(yyCorr,2),1)-Y(1,1),yyCorr)




