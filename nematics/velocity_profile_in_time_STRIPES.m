% Analysis of flows in stripes
% for PIV sesssion on stripe or wire

% ------  CALCULATES: -----------
% 1) profiles of Vx, Vy average in time [x=x_axis,y=u/v_profile, errorbar=u/v_std]
% 2) kymograph of Vx, Vy 
% 3) Enstrophy - Energy
% 4) velocity on boundaries (LEFT/RIGHT)
% 5) Distribution of bounadary velocities

% ------  USAGE: -----------
% Import PIV session from PIVlab
% wire velocity analysis averaged by length and time
% images should be alighned up/down

% ------  PARAMETERS: -----------
% 1) pixel size 
% 2) frame rate
% 3) First frame
% 4) Last frame
%% 
% ---------  START   DIRECT READ FROM PIVlab session  ----------
clear all
close all
%%
[filename, pathname] = uigetfile( ...
    '*.mat', 'Pick a file',...
'D:\CURIE_DATA\HT1080\stripe_PIV');

load([pathname,filename],'resultslist'); %i
A = resultslist;  % Assign it to a new variable with different name.
clear('resultslist'); % If it's really not needed any longer.
x(:,1) = A(1,:)';
y(:,1) = A(2,:)';
u_filtered(:,1) = A(7,:)';
v_filtered(:,1) = A(8,:)';
% ---------- END  ----------------------------------------------
%% CELL 1: parameters
px_size = .74; %40x (0.185); 20x (0.37); 10x (0.74)
dxx = px_size* (x{1,1}(1,2)- x{1,1}(1,1));
frame_per_hr = 4;
px2mic = px_size * frame_per_hr;

%% CELL 2: import and calculate 
[L,W] = size(v_filtered{1,1}); % Width Length
T = size(v_filtered,1);

Fi = 1;
Fend = T-1;
T = Fend-Fi;

% dx = imageWidth_px * px_size/W; % vector field step resolution
x_axis = px_size*x{1,1}(1,:);
time_hr = 0:1/frame_per_hr:T/frame_per_hr;
[XX,TT] = meshgrid(x_axis,time_hr);

%% 
% make u average over y-coordinate*
u_profile = zeros(1,W);
v_profile = zeros(1,W);
u_std = zeros(1,W);
v_std = zeros(1,W);
u_profile_map = zeros(T,W);
v_profile_map = zeros(T,W);

ii = 1; 

for i=Fi:Fend   
% correct shift by reducing average velocity    
%         u_profile = u_profile + mean(u_filtered{ii,1} - mean(u_filtered{ii,1}(:)),1);
        u_profile = u_profile + mean(u_filtered{ii,1},1);
        u_std = u_std + std(u_filtered{ii,1},1);
%         v_profile = v_profile + mean(v_filtered{ii,1} - mean(v_filtered{ii,1}(:)),1);
        v_profile = v_profile + mean(v_filtered{ii,1},1);
        v_std = v_std + std(v_filtered{ii,1},1);
        
        u_profile_map(ii,:) =  mean(u_filtered{ii,1} - mean(u_filtered{ii,1}(:)),1);
        v_profile_map(ii,:) = mean(v_filtered{ii,1} - mean(v_filtered{ii,1}(:)),1);         
%         u_profile_map(ii,:) = mean(u_filtered{ii,1},1);
%         v_profile_map(ii,:) = mean(v_filtered{ii,1},1);
ii = ii + 1;
end
u_profile = u_profile/T * px2mic;
v_profile = v_profile/T * px2mic;

u_profile_map = u_profile_map * px2mic;
v_profile_map = v_profile_map * px2mic;
%--- u_std:std of each PIV vector------ 
    u_std = u_std/T * px2mic / sqrt(T*L);
    v_std = v_std/T * px2mic / sqrt(T*L);
%--- u_std1:std of averaged by y profile in time ------ 
    u_std1 = std(u_profile_map,1);
    v_std1 = std(v_profile_map,1);
%-------------substract the average----------------
%     u_profile = u_profile - mean(u_profile);
%     v_profile = v_profile - mean(v_profile);
%-------------------------------------------------

subplot(2,4,1);
errorbar(x_axis, u_profile, u_std);hold on
errorbar(x_axis, v_profile, v_std);hold off
xlabel('width (\mum)'); ylabel('speed (\mum/hr)'); legend('<u>_{y}','<v>_{y}'); axis tight
hold off

subplot(2,4,2);
y_axis = zeros(size(x_axis));
quiver(x_axis,y_axis,u_profile,v_profile,0)
xlabel('width (\mum)'); ylabel('speed (\mum/hr)'); legend('<v>_{y}');axis tight 
hold off

subplot(2,4,3);
surf(XX,TT,u_profile_map);view(2);shading interp;axis tight; title('<u>_{y}');xlabel('width (\mum)');ylabel('time (hr)');
umax = [max(mean(abs(u_profile_map(:,1))), mean(abs(u_profile_map(:,end))))]; 
caxis([-umax umax]); colorbar;
hold off

subplot(2,4,4);
surf(XX,TT,v_profile_map);view(2);shading interp;axis tight; title('<v>_{y}');xlabel('width (\mum)');ylabel('time (hr)');
vmax = [max(mean(abs(v_profile_map(:,1))), mean(abs(v_profile_map(:,end))))]; 
caxis([-vmax vmax]); colorbar;
hold off

%%
subplot(2,4,5:6);
% plot(TT(:,1),mean(v_profile_map(:,1:2),2),'-');hold on
% plot(TT(:,1),mean(v_profile_map(:,end-1:end),2),'-');
plot(TT(:,1),v_profile_map(:,1),'-');hold on
plot(TT(:,1),v_profile_map(:,end),'-');
ylabel('boundary speed (\mum/hr)');xlabel('time (hr)');
title(['stripe width = ', num2str(x_axis(end),4), ' \mum']);
axis([-2 time_hr(end)+2 -inf inf]); hold off

v1=mean(v_profile_map(:,1));
v1_std=std(v_profile_map(:,1))/sqrt(L);
v2=mean(v_profile_map(:,end));
v2_std=std(v_profile_map(:,end))/sqrt(L);

subplot(2,4,8);
h1 = histogram(v_profile_map(:,end),20,'Normalization','pdf');
hold on
h2 = histogram(v_profile_map(:,1),20,'Normalization','pdf');
% axis([-2*vmax 2*vmax 0 inf]);
ylabel('PDF');
legend(['v = ' num2str(v1)...
    '| std=' num2str(v1_std)],...
    ['v = ' num2str(v2)...
    '| std=' num2str(v2_std)],'Location','southoutside');
hold off

%%  (RUN 2) CHECK VELOCITY FOR PART OF FRAMES 
[tt,~] = ginput(2);

Fi = floor(tt(1)*frame_per_hr);
Fend = floor(tt(2)*frame_per_hr);

if tt(2)>time_hr(end)
    Fend = floor(time_hr(end)*frame_per_hr);
end
if tt(1)<=4
    Fi = 1;
end

T = Fend-Fi;

d_time_hr = 0:1/frame_per_hr:T/frame_per_hr;

[XX,TT] = meshgrid(x_axis,d_time_hr);
 
% make u average over y-coordinate*
u_profile = zeros(1,W);
v_profile = zeros(1,W);
u_std = zeros(1,W);
v_std = zeros(1,W);
u_profile_map = zeros(T,W);
v_profile_map = zeros(T,W);

ii = 1; 

for i=Fi:Fend

        u_profile = u_profile + mean(u_filtered{ii,1} - mean(u_filtered{ii,1}(:)),1);
%         u_profile = u_profile + mean(u_filtered{i,1},1);
        v_profile = v_profile + mean(v_filtered{ii,1} - mean(v_filtered{ii,1}(:)),1);  
%         v_profile = v_profile + mean(v_filtered{i,1},1);         
        u_std = u_std + std(u_filtered{i,1},1);        
        v_std = v_std + std(v_filtered{i,1},1);
        
        u_profile_map(ii,:) =  mean(u_filtered{ii,1} - mean(u_filtered{ii,1}(:)),1);
        v_profile_map(ii,:) = mean(v_filtered{ii,1} - mean(v_filtered{ii,1}(:)),1);       
%         u_profile_map(ii,:) = mean(u_filtered{i,1},1);
%         v_profile_map(ii,:) = mean(v_filtered{i,1},1);
ii = ii + 1;
end
u_profile = u_profile/T * px2mic;
v_profile = v_profile/T * px2mic;

u_profile_map = u_profile_map * px2mic;
v_profile_map = v_profile_map * px2mic;
%--- u_std:std of each PIV vector------ 
    u_std = u_std/T * px2mic / sqrt(T*L);
    v_std = v_std/T * px2mic / sqrt(T*L);
%--- u_std1:std of averaged by y profile in time ------ 
    u_std1 = std(u_profile_map,1);
    v_std1 = std(v_profile_map,1);
%-------------substract the average----------------
%     u_profile = u_profile - mean(u_profile);
%     v_profile = v_profile - mean(v_profile);
%-------------------------------------------------
% %%
figure('Name',['Cropped: F(1)', num2str(Fi),' - F(end)', num2str(Fend)],...
    'NumberTitle','off')

subplot(2,4,1);
errorbar(x_axis, u_profile, u_std);hold on
errorbar(x_axis, v_profile, v_std);hold off
xlabel('width (\mum)'); ylabel('speed (\mum/hr)'); legend('<u>_{y}','<v>_{y}'); axis tight
hold off

subplot(2,4,2);
y_axis = zeros(size(x_axis));
quiver(x_axis,y_axis,u_profile,v_profile,0)
xlabel('width (\mum)'); ylabel('speed (\mum/hr)'); legend('<v>_{y}'); axis tight 
hold off

subplot(2,4,3);
surf(XX,TT,u_profile_map);view(2);shading interp;axis tight; title('<u>_{y}');xlabel('width (\mum)');ylabel('time (hr)');
umax = [max(mean(abs(u_profile_map(:,1))), mean(abs(u_profile_map(:,end))))]; 
caxis([-umax umax]); colorbar;
hold off

subplot(2,4,4);
surf(XX,TT,v_profile_map);view(2);shading interp;axis tight; title('<v>_{y}');xlabel('width (\mum)');ylabel('time (hr)');
vmax = [max(mean(abs(v_profile_map(:,1))), mean(abs(v_profile_map(:,end))))]; 
caxis([-vmax vmax]); colorbar;
hold off


subplot(2,4,5:6);
% plot(TT(:,1),mean(v_profile_map(:,1:2),2),'-');hold on
% plot(TT(:,1),mean(v_profile_map(:,end-1:end),2),'-');
plot(TT(:,1),v_profile_map(:,1),'-');hold on
plot(TT(:,1),v_profile_map(:,end),'-');
ylabel('boundary speed (\mum/hr)');xlabel('time (hr)');
title(['stripe width = ', num2str(x_axis(end),4), ' \mum']);
axis([-2 d_time_hr(end)+2 -inf inf]); hold off

v1=mean(v_profile_map(:,1));
v1_std=std(v_profile_map(:,1))/sqrt(L);
v2=mean(v_profile_map(:,end));
v2_std=std(v_profile_map(:,end))/sqrt(L);

subplot(2,4,8);
h1 = histogram(v_profile_map(:,end),'Normalization','pdf');
hold on
h2 = histogram(v_profile_map(:,1),'Normalization','pdf');
% axis([-2*vmax 2*vmax 0 inf]);
ylabel('PDF');
legend(['v = ' num2str(v1)...
    '| std=' num2str(v1_std)],...
    ['v = ' num2str(v2)...
    '| std=' num2str(v2_std)],'Location','southoutside');
hold off

%% VORTICITY ETC... (from Carles file)
subplot(2,4,7);
% px_size = .74; %20x (0.37); 10x (0.74)
% frame_per_hr = 4;
% px2mic = px_size * frame_per_hr;
[W,L] = size(v_filtered{1,1}); % Width Length
% T = size(v_filtered,1);
% x_axis = px_size*x{1,1}(1,:);
dx = x_axis(2)-x_axis(1);
% time_hr = 0:1/frame_per_hr:(T-1)/frame_per_hr;
% [XX,TT] = meshgrid(x_axis,time_hr);

kEnergy = zeros(W,L,T); % kinetic energy
vEnergy = zeros(W,L,T); % 
k2v_Energy_ratio = zeros(W,L,T); % 
vorticity = zeros(W,L,T); % 
shear = zeros(W,L,T); % 
strain = zeros(W,L,T); % 
mean_kE = zeros(T,1);
mean_vE = zeros(T,1);
std_kE = zeros(T,1);
std_vE = zeros(T,1);

uu = zeros(W,L);
vv = zeros(W,L);
uu1 = zeros(W,L,T);
vv1 = zeros(W,L,T);

dt=1;
ii=1;
for i=Fi:dt:(Fend+1)-dt
uu = 0*uu;
vv = 0*vv;
% ----- averaging of velocity fields (dt=1: cancels averaging) -----
    for t=1:dt
        uu = uu + u_filtered{ii+t-1,1};
        vv = vv + v_filtered{ii+t-1,1};
    end
    uu = px2mic*uu/dt;
    vv = px2mic*vv/dt;
    uu1(:,:,ii) = uu;
    vv1(:,:,ii) = vv;
% ----------------------------------------------------------------------    
    [u_x,u_y] = gradient(uu/dx);% gradient need to be corrected for the dx 
    [v_x,v_y] = gradient(vv/dx);
    vEnergy(:,:,ii) = 0.5*(v_x - u_y).^2; 
    vorticity(:,:,ii) = (v_x - u_y); 
    shear(:,:,ii) = (v_x + u_y).^2; 
    strain(:,:,ii) = (u_x + v_y).^2; 
    kEnergy(:,:,ii) = 0.5*(uu.^2 + vv.^2);
%     k2v_Energy_ratio(:,ii) = reshape(kEnergy(:,:,ii)./vEnergy(:,:,ii), [W*L, 1]); 
    k2v_Energy_ratio(:,:,ii) = kEnergy(:,:,ii)./vEnergy(:,:,ii); 
    mean_vE(ii) = mean(mean(vEnergy(:,:,ii)));
    std_vE(ii) = mean(std(vEnergy(:,:,ii)))/sqrt(W*L);
    mean_kE(ii) = mean(mean(kEnergy(:,:,ii)));
    std_kE(ii) = mean(std(kEnergy(:,:,ii)))/sqrt(W*L);
    plot(mean_vE(ii),mean_kE(ii), ....
    'o', 'MarkerFaceColor', [1-(ii-1)/T, 1-(ii-1)/T, 1-(ii-1)/T]...
        ,'MarkerEdgeColor', [0,0,0]);
%     errorbar(mean_kE(i),mean_vE(i),std_vE(i) ....
%     ,'o', 'MarkerFaceColor', [1-i/T, 1-i/T, 1-i/T]...
%         ,'MarkerEdgeColor', [0,0,0]);
    hold on
    ii = ii + 1;
end

x_max = max(mean_kE(:));
y_max = max(mean_vE(:));
axis([0 inf 0 inf]);
ylabel('\Omega, Enstrophy');xlabel('E_k, Averaged Energy');
% text(x_max/10,y_max*6/8,'C2C12\s11');

%%
% -----  fit Ek/Ev ratio
[f,fq] = fit(mean_vE,mean_kE,'poly1','Lower',[0 0],'Upper',[Inf 0]);
Fy = f.p1.*mean_vE + f.p2;
plot(mean_vE,mean_kE,'o'); hold on
plot(mean_vE,Fy); hold off
% % title(['R^2: ', num2str(fq.rsquare,3)])
title(['Ek/Ev fit: ', num2str(f.p1,5),'    R^2: ', num2str(fq.rsquare,3)]);
xlabel('\Omega, Enstrophy');ylabel('E_k, Kinetic Energy');hold off
fig = gcf;
set(fig,'Position',[100 100 1200 800]);
saveas(gcf, [pathname, filename,'F', num2str(Fi),'-',num2str(Fend),'.png']);
% scrsz = get(groot,'ScreenSize');

%%
% n = 80;
% k2v_Energy_ratio(k2v_Energy_ratio>1000)=-100;
% %%
% histogram(k2v_Energy_ratio(:,:,:),100);mean(k2v_Energy_ratio(:))
% %%
% histogram(uu1,100,'Normalization','pdf');hold on
% histogram(vv1,100,'Normalization','pdf');hold off
% legend(['v_x' num2str(std(uu1(:)))],['v_y' num2str(std(vv1(:)))])

%%  SAVE THE DATA TO EXCEL

EX = zeros(size(v_filtered,1)+1,7);
EX(2:end,1) =  time_hr';  
% EX(2:end,2) =  edgeAll; 
% EX(2:end,3) =  std_edgeAll; 
% EX(2:end,4) =  midAll; 
% EX(2:end,5) =  std_midAll; 
% EX(2:end,6) =  edge_OP;
% EX(2:end,7) =  mid_OP;
EX = num2cell(EX);
EX(1,:) = {'Time','Edge Angle','Edge Angle STD',...
                  'Mid Angle','Mid Angle STD',...
                  'Edge Order Parameter', 'Mid Order Parameter'};
%%
EXcropped = zeros(T+2,7);
EXcropped(2:end,1) =  d_time_hr;  
% EXcropped(2:end,2) =  d_edgeAll; 
% EXcropped(2:end,3) =  d_std_edgeAll; 
% EXcropped(2:end,4) =  d_midAll; 
% EXcropped(2:end,5) =  d_stdmidAll; 
% EXcropped(2:end,6) =  d_edge_OP;
% EXcropped(2:end,7) =  d_mid_OP;
EXcropped = num2cell(EXcropped);
EXcropped(1,:) = EX(1,:);

% xlswrite([pathname, filename,'.xlsx'],EX,1);% xlswrite(filename,DATA,sheet)
% xlswrite([pathname, filename,'.xlsx'],EXcropped,2);