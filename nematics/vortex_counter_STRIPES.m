% DEFECT detection and classification
% % READ PIV
[pfilename, ppathname] = uigetfile( ...
    '*.mat', 'Pick a PIV file',...
    'C:\Users\vici\Desktop\HT1080\stripe_PIV');

load([ppathname,pfilename],'resultslist');
A = resultslist;  % Assign it to a new variable with different name.
clear('resultslist'); % If it's really not needed any longer.
px_size = .74;
X = px_size*A{1,1};
Y = px_size*A{2,1};

%% Check in one frame (k)
dt=1;
ff = 5;
filt = fspecial('gaussian',ff,ff);
k=80;

    % --------------------------PIV import ---------------------------------  
    u_filtered = A(7,k:k+dt)';
    v_filtered = A(8,k:k+dt)';
    
    uu = zeros(size(u_filtered{1,1}));
    vv = uu;
    % ----- averaging of velocity fields (dt=1: cancels averaging) -----
    dt = 1; % velocity avearage
    for t=1:dt
        uu = uu + imfilter(u_filtered{t,1}, filt);
        vv = vv + imfilter(v_filtered{t,1}, filt);
    end    
    uu = uu/dt;
    vv = vv/dt;

% Velocity
figure(23);
vstep = 1;
q0 = quiver(X(1:vstep:end,1:vstep:end),Y(1:vstep:end,1:vstep:end),...
    uu(1:vstep:end,1:vstep:end),vv(1:vstep:end,1:vstep:end),1.5);
q0.LineWidth=1.5;
q0.Color = [1 1 1];hold on

% Vorticity
[u_x,u_y] = gradient(uu);%/dx gradient need to be corrected for the dx
[v_x,v_y] = gradient(vv);%/dx
vorticity = (v_x - u_y);
ff = 7;
filt = fspecial('gaussian',ff,ff);
u1 = imfilter(vorticity, filt);
surf(X(1:vstep:end,1:vstep:end),Y(1:vstep:end,1:vstep:end),...
    u1(1:vstep:end,1:vstep:end)-10);caxis([-max(u1(:))/1.5-10, max(u1(:))/1.5-10]);
view(2);shading interp;colormap jet;axis equal;axis tight;axis off
view([-90 90]);
hold off 

Ok_Wi(:,:)  = (u_x+v_y).^2-4*(u_x.*v_y-v_x.*u_y);
ff1 = 5; 
ffilt = fspecial('gaussian',ff1,ff1);
u1 = vorticity(:,:);%imfilter(vorticity(:,:), filt);
u2 = imfilter(Ok_Wi(:,:), ffilt);  % imfilter(u1(:,:), ffilt);
u2_1 = u2 < min(u2(:))/30;% works good for 10
% u2_1 = bwareaopen(u2_1, 4, 4);
WS = bwlabel(u2_1);
% bw = u2_1;
% D = bwdist(~bw);
% D = -D;
% D(~bw) = Inf;
% WS = watershed(D,4);
% WS(~bw) = 0; 
WSflipped = flipud(WS);
s = regionprops('table', WS, vorticity,'centroid','Area','MeanIntensity');
figure(223);
imagesc(WS); hold on; %imagesc(u2);
scatter(s.Centroid(:,1),s.Centroid(:,2),20,[1 1 1],'filled');
hold off 
view(2);shading interp;colormap jet;axis equal;axis tight;%axis off
view([-90 90])