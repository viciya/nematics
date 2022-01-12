% run first two cells of correlate_OP_PIV_Defect.m

widthChoice = 1514; dw = .05*widthChoice;

px_size = .748;

jjS = find(Sorted_Orient_width(:,2)<widthChoice+dw &...
    Sorted_Orient_width(:,2)>widthChoice-dw);

% for i = 784:789%1:size(Ddir,1)
for i = 1:length(jjS)
    i
jj = jjS(1);
ii = Sorted_Orient_width(jj,1);
Sorted_Orient_width(jj,2)
folderPIV = dirPIV(indX(ii,2)).folder;
namePIV = dirPIV(indX(ii,2)).name;
filepathPIV = [folderPIV '\' namePIV];    
    
    %         i=4
    clear var Ok_Wi
    load(filepathPIV); % Assign it to a new variable with different name.
    
    vort_area(i) = px_size^2*y{1}(end,end)*x{1}(end,end);
    vort_width(i) = px_size*x{1}(end,end);    
    
    for k = 1:size(x,1)
        %             %%
        %             k=23;
        ff = 7; filt = fspecial('gaussian',ff,ff);
        % --------------------------PIV import ---------------------------------
        uu = zeros(size(u{k})); vv = uu;
        uu = imfilter(u{k}, filt);
        vv = imfilter(v{k}, filt);
        
        % Vorticity
        [u_x,u_y] = gradient(uu);%/dx gradient need to be corrected for the dx
        [v_x,v_y] = gradient(vv);%/dx
        vorticity = (v_x - u_y);
        ff = 5;
        filt = fspecial('gaussian',ff,ff);
        u1 = imfilter(vorticity, filt);
        
        Ok_Wi(:,:)  = (u_x+v_y).^2-4*(u_x.*v_y-v_x.*u_y);
        ff1 = 5; ffilt = fspecial('gaussian',ff1,ff1);
        u2 = imfilter(Ok_Wi, ffilt);
        u2_1 = u2 < min(u2(:))/500;
        %         se = strel('disk',3);  u2_1 = imclose(u2_1,se);
        %         u2_1 = bwareaopen(u2_1, 4, 4);
        se = strel('disk',1);  u2_1 = imopen(u2_1,se);%u2_1 = imerode(u2_1,se);
        WS = bwconncomp(u2_1);
        ccw_vort = (u1.*u2_1)>0;
        cw_vort = (u1.*u2_1)<0;
        
%                 figure(33);
%         %         subplot(1,2,1);
%         %         imagesc(l_vort);colormap jet;axis equal;axis tight;
%                 subplot(1,2,2);
%                 imagesc(u1);colormap jet;axis equal;axis tight;%view(2);shading interp;
%                 figure(33); subplot(1,2,1);
%                 C = imfuse(ccw_vort, cw_vort,'falsecolor','Scaling','joint','ColorChannels',[1 0 2]);
%                 imagesc(C); axis equal;axis tight;

        
        ccw_s = regionprops('table', ccw_vort, vorticity,'centroid','Area','MeanIntensity');
        vortNum{i,1}(k,1) = size(ccw_s,1);
        vortDensity{i,1}(k,1) = size(ccw_s,1)/vort_area(i);
%         vortArea{i,1}(k,:) = px_size^2*(x{k}(1,2)-x{k}(1,1))^2*(rs.Area);
        vortAreaCCW{k,i} = px_size^2*(x{k}(1,2)-x{k}(1,1))^2*(ccw_s.Area);
        
        cw_s = regionprops('table', cw_vort, vorticity,'centroid','Area','MeanIntensity');
        vortNum{i,2}(k,1) = size(cw_s,1);
        vortDensity{i,2}(k,i) = size(cw_s,1)/vort_area(i);
%         vortArea{i,2}(k,:) = px_size^2*(x{k}(1,2)-x{k}(1,1))^2*(ls.Area);
        vortAreaCW{k,i} = px_size^2*(x{k}(1,2)-x{k}(1,1))^2*(cw_s.Area);
        % ------------------VORTEX AREA DISTRIBUTION--------------------------------------
        % --------------------------------------------------------------------------------
        % ------------------VORTEX AREA DISTRIBUTION--------------------------------------
        % --------------------------------------------------------------------------------
        
        %             % Velocity
        %             figure(23);
        %             vstep = 1;
        %             q0 = quiver(X(1:vstep:end,1:vstep:end),Y(1:vstep:end,1:vstep:end),...
        %                 uu(1:vstep:end,1:vstep:end),vv(1:vstep:end,1:vstep:end),3);
        %             q0.LineWidth=.5;
        %             q0.Color = [0 0 0];hold on
        %
        %             figure(223);
        %             imagesc(WS); hold on
        %             scatter(s.Centroid(:,1),s.Centroid(:,2),20,[1 1 1],'filled');
        %             hold off
        %             view(2);shading interp;colormap jet;axis equal;axis tight;%axis off
        
    end
end
disp("****************DONE************");
%%
vortAreaCW_m = cell2mat(vortAreaCW);
vortAreaCCW_m = cell2mat(vortAreaCCW);

%%
h1= histogram(vortAreaCW_m,30); hold on
h2= histogram(vortAreaCCW_m,30); %hold off
h1.Normalization = 'probability';
% h1.BinWidth = 300;
h2.Normalization = 'probability';
% h2.BinWidth = 300;
 xE = 1:12000;
plot(xE,exp(-xE/1500)); hold on
set(gca, 'YScale', 'log')
%%
nBins = 32;
[CW_pdf, edges] = histcounts(vortAreaCW_m, nBins, 'Normalization', 'probability');
edges1 = .5*(edges(1:end-1)+edges(2:end));
plot(edges1,CW_pdf); hold on
[CCW_pdf, edges] = histcounts(vortAreaCCW_m ,nBins,'Normalization', 'probability');
edges1 = .5*(edges(1:end-1)+edges(2:end));
plot(edges1,CCW_pdf);
hold off
set(gca, 'YScale', 'log')
AREA_str = num2str(AREA);
legend('CW','CCW','e^{-A/1500}');xlabel('Area (\mum)^{2}'); ylabel('PDF');set(gca,'Fontsize',18)
axis([0 inf 0 inf]);
%%
i=54;
len = 50;
for i=3:size(vort_area,2)
meanVortDensity(i) = mean(vortDensity{i}(end-len:end));
stdVortDensity(i) = std(vortDensity{i}(end-len:end))/len^.5;

meanVortNum(i) = mean(vortNum{i}(end-len:end));
stdVortNum(i) = std(vortNum{i}(end-50:end))/len^.5;

meanVortArea(i) = mean(vortArea{i}(end-len:end));
stdVortArea(i) = std(vortArea{i}(end-len:end))/len^.5;
end

[sortVort_width, ind] = sort(vort_width);
sortVort_area = vort_area(ind);
meanVortDensity = meanVortDensity(ind);
stdVortDensity = stdVortDensity(ind);
meanVortNum = meanVortNum(ind);
stdVortNum = stdVortNum(ind);
figure(1)
errorbar(sortVort_width,meanVortDensity,stdVortDensity);hold on
figure(2)
errorbar(sortVort_width,meanVortNum,stdVortNum);hold on
figure(3)
errorbar(sortVort_width,meanVortArea,stdVortArea);hold on
