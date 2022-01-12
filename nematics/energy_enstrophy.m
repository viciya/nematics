%%
% !!! RUN 'correlate_OP_PIV_Defect.m' !!!

% Ddir = dir(['C:\Users\vici\Google Drive\DATA\HT1080\PIV_DATs'  '\*.mat']);
% Ddir = dir(['\\Nausicaa\Victor\HT1080_small_stripes_glass_22112017\CROPPED\PIV_DATs'  '\*.mat']);
Ddir = dir(['C:\Users\victo\Google Drive\DATA\HT1080\PIV_DATs'  '\*.mat']);
% Ddir = dir([PC_path, 'DATA\HT1080\PIV_DATs'  '\*.mat']);
folder_main = Ddir(1).folder;
filesInFolder = size(Ddir,1);
px_size = .74;
frame_per_hr = 4;
px2mic = px_size * frame_per_hr;

width = zeros(size(Ddir,1),1);
kEnergy = cell(size(Ddir,1),1);
vEnergy = cell(size(Ddir,1),1);
%%
% !!! RUN 'correlate_OP_PIV_Defect.m' first two cells !!!
px_size = .74;
frame_per_hr = 4;
px2mic = px_size * frame_per_hr;
% for i=1:size(Ddir,1)
for i=1:size(dirOP,1)
%     if contains(Ddir(i).name, '.mat' )
%         Ddir(i).name
 disp(['file: ' num2str(indX(i,1)) ' from: ' num2str(size(dirOP,1))]);
% dirPIV(indX(i,2)).name
% dirOP(indX(i,1)).name
%         filepath = [Ddir(i).folder '\' Ddir(i).name];
%         load(filepath);
    filepathPIV = [dirPIV(indX(i,2)).folder '\' dirPIV(indX(i,2)).name];
    load(filepathPIV); 
        X = x{1}; Y = y{1};
        dx = px_size*(x{1}(1,2)-x{1}(1,1));        
        vEnergy_temp = 0;kEnergy_temp = 0; vLt=0;vRt=0;
        for k=1:size(x,1)            
            % --------------------------PIV import ---------------------------------
            uu = zeros(size(u(k))); vv = uu; 
            uu = px_size*u{k};
            vv = px_size*v{k};
            vNett(k,1) = mean2(vv);
            vLt(k,1) = mean(vv(:,1)) - vNett(k,1);
            vRt(k,1) = mean(vv(:,end))- vNett(k,1);
            
%             %filter
%             ff = 1; filt = fspecial('gaussian',ff,ff);
%             uu = px2mic*imfilter(u{k}, filt);
%             vv = px2mic*imfilter(v{k}, filt);
            
            % Vorticity
            [u_x,u_y] = gradient(uu,dx);%/dx gradient need to be corrected for the dx
            [v_x,v_y] = gradient(vv,dx);%/dx
            vEnergy_temp = vEnergy_temp + 0.5*(v_x - u_y).^2;
            kEnergy_temp = kEnergy_temp + 0.5*(uu.^2 + vv.^2);
        end
%         plot(1:k,vLt,1:k,vRt)
        width(i,1) = X(end);
        vEnergy{i,1} = vEnergy_temp/k;
        kEnergy{i,1} = kEnergy_temp/k;
        vNet{i,1} = vNett;
        vL(i,1) = mean(vLt);
        vR(i,1) = mean(vRt);
%     end
end

%%
emptyTF = cellfun(@isempty,kEnergy); %check for empty cells
cd=150;% shift jet colors
c1=jet(size(kEnergy,1)+cd);
cc = c1(1:end-cd,:);
count = 1;
for i=1:size(kEnergy,1)
    if emptyTF(i)==0 %&& px_size * width(i)<400
        wEW(count,1) = px_size * width(i);
        wEW(count,2) = mean2(kEnergy{i,1});
        wEW(count,3) = mean2(vEnergy{i,1});
    count = count + 1;
    end
end
%%
load("C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_Ek_Ev_unfiltered.mat");
[wEW,ind] = sortrows(wEW,1);
dv = vR(ind)-vL(ind);
cc=jet(size(wEW,1));
figure(1); scatter(wEW(:,1),wEW(:,2),20,cc,'filled');xlabel('Width (\mum)'); ylabel('E_k (\mum/hr)^{2}');set(gca,'Fontsize',18);
figure(2);scatter(wEW(:,1),wEW(:,3),20,cc,'filled');xlabel('Width (\mum)'); ylabel('\Omega (1/hr)^{2}');set(gca,'Fontsize',18);
figure(3);scatter(wEW(:,3),wEW(:,2),20,cc,'filled');ylabel('E_k (\mum/hr)^{2}'); xlabel('\Omega (1/hr)^{2}');set(gca,'Fontsize',18)
figure(4);scatter(wEW(:,1),(wEW(:,2)./wEW(:,3)).^.5,25,cc,'filled');xlabel('Width (\mum)'); ylabel('(E_k/\Omega)^{-1/2} (\mum)');set(gca,'Fontsize',18);
axis([0 1040 0 20]);%axis tight;%hold off

% figure(5);scatter(wEW(1:500,3),dv(1:500),20,cc(1:500,:),'filled');set(gca,'Fontsize',18);ylabel('\Deltav'); xlabel('\Omega (1/hr)^{2}');
%%
[wEW,ind] = sortrows(wEW,1);
w_set = ([50,120,200,600,700,800,1000,1500]);
cc=jet(length(w_set));

for i=1:length(w_set)
Sw = w_set(i);
dw = .2*Sw;
% choose width range
wEWs = wEW(wEW(:,1)>Sw-dw & wEW(:,1)<Sw+dw,:);

figure(2); 
sc1 = scatter(wEWs(:,3),wEWs(:,2),40,cc(i,:),'filled');
sc1.MarkerFaceAlpha = .2;
hold on

b1 = wEWs(:,3)\wEWs(:,2);
yCalc1 = b1*[0;wEWs(:,3);60];
p1 = plot([0;wEWs(:,3);60],yCalc1,'Color',[cc(i,:), .6]);
p1.LineWidth = 2;
end
set(gca,'Fontsize',18);
ylabel('$ E_{k}\ (\mu m/h)^{2} $','Interpreter','latex','FontSize',28);
xlabel('$ \Omega\ (hr^{-2}) $','Interpreter','latex','FontSize',28);
axis([0 60 0 14000]);hold off
ax = gca;
text(5,ax.YLim(2)/1.6,compose('%d',flip(w_set)),'Fontsize',18);
compose('%d',w_set);
%%
mm = 1346;
figure(6);scatter(wEW(1:mm,1),dv(1:mm),20,cc(1:mm,:),'filled');set(gca,'Fontsize',18);ylabel('\Deltav (\mum/hr)'); xlabel('Width (\mum)');
hold on
%%
Width = wEW(:,1);
av_kEnergy = wEW(:,2);
av_vEnergy = wEW(:,3);
%%
bwidth = 1:10:1600;
[UWidth,~,idx]  = unique(Width);
N = histc(Width, UWidth); % repetition number
UWidth_kE = [UWidth, accumarray(idx, av_kEnergy,[],@mean), accumarray(idx,av_kEnergy,[],@std)./sqrt(N)];
UWidth_vE = [UWidth, accumarray(idx, av_vEnergy,[],@mean), accumarray(idx,av_vEnergy,[],@std)./sqrt(N)];
figure(11); errorbar(UWidth_kE(:,1),UWidth_kE(:,2),UWidth_kE(:,3));
figure(22);errorbar(UWidth_vE(:,1),UWidth_vE(:,2),UWidth_vE(:,3));
figure(33);%plot(N, );

