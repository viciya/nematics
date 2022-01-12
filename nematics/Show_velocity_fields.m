% DEFECT detection and classification
% % READ PIV
[pfilename, ppathname] = uigetfile( ...
    '*.mat', 'Pick a PIV file',...
    'C:\Users\vici\Desktop\HT1080');

load([ppathname,pfilename],'resultslist');
A = resultslist;  % Assign it to a new variable with different name.
clear('resultslist'); % If it's really not needed any longer.
X = A{1,1};
Y = A{2,1};

%% Check in one frame (k)
dt=1;
ff = 5;
filt = fspecial('gaussian',ff,ff);
k=300;

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
figure(22);
vstep = 2;
vstep1 = 1;
q0 = quiver(X(1:vstep:end,1:vstep:end),Y(1:vstep:end,1:vstep:end),...
    uu(1:vstep:end,1:vstep:end),vv(1:vstep:end,1:vstep:end),1.2);
q0.LineWidth=1;
q0.Color = [0 0 0];hold on

% Vorticity
[u_x,u_y] = gradient(uu);%/dx gradient need to be corrected for the dx
[v_x,v_y] = gradient(vv);%/dx
vorticity = (v_x - u_y);
ff = 5;
filt = fspecial('gaussian',ff,ff);
u1 = imfilter(vorticity, filt);
surf(X(1:vstep1:end,1:vstep1:end),Y(1:vstep1:end,1:vstep1:end),...
    u1(1:vstep1:end,1:vstep1:end)-10);
caxis([-.9*max(u1(:))-10, .9*max(u1(:))-10]);
% Divergence
% vstep = 1;%
% divV =  divergence(uu,vv);
% ff = 5;
% filt = fspecial('gaussian',ff,ff);
% u2 = imfilter(divV, filt);
% surf(Xorig(1:vstep:end,1:vstep:end),Yorig(1:vstep:end,1:vstep:end),...
%     u2(1:vstep:end,1:vstep:end)-10);

view(2);shading interp;colormap jet;axis equal;axis tight;axis off
hold off 
% view([-90 90])
%%
figure
colorS = [min(abs(uu(:))) :.5: max(abs(uu(:)))]
% colorS = [-pi:.3:pi];
ncquiverref(X(1:vstep:end,1:vstep:end),Y(1:vstep:end,1:vstep:end),...
    uu(1:vstep:end,1:vstep:end),vv(1:vstep:end,1:vstep:end),...
    'um/hr','mean',1,'col',colorS);
axis equal
%%
[th,z] = cart2pol(uu,vv);
