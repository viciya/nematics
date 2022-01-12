% %% LOAD FILES
% PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
PC_path = 'D:\GD\';                         % RSIP notebook

addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

[dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);
%% SELECT WIDTH AND PARAMETERS
% clearvars -except dirOP  dirPIV  Sorted_Orient_width  indX PC_path pathOP pathPIV
Edge = 70;
WidthRange = [250 300 400 500 600 700 800 1000];
parfor k = 6:8%1:numel(WidthRange) 
    
    Sw = WidthRange(k); % selectd width
    dw = .05*Sw; % define delta
    Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);
    
    LeftXYQ = cell(numel(Range),1);
    RightXYQ = cell(numel(Range),1);

    for i = 1:numel(Range)
        disp(['File ',num2str(i), ' from ',num2str(numel(Range))]);
        filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
        
        [tLeftXYQ, tRightXYQ] = get_Edge_angle_file(filepathOP, Edge);
        LeftXYQ{i} = tLeftXYQ;
        RightXYQ{i} = tRightXYQ;
        
    end
    
    %% UNWRAP CELL
    for i = 1:size(LeftXYQ,1) 
        arrTF = ~cellfun('isempty', LeftXYQ{i});
        tLeftXYQ{i,1} = cell2mat(LeftXYQ{i}(arrTF));
        tRightXYQ{i,1} = cell2mat(RightXYQ{i}(arrTF));
    end
    arrTF = ~cellfun('isempty', tLeftXYQ);
    matLeftXYQ = cell2mat(tLeftXYQ(arrTF));
    matRightXYQ = cell2mat(tRightXYQ(arrTF));
%     figure(111); polarhistogram(matLeftXYQ(:,3)*pi/180,90, 'Normalization','PDF');hold on
%     figure(111); polarhistogram(matRightXYQ(:,3)*pi/180,90, 'Normalization','PDF');hold off
    
    %% Calculate mean
    Left_av_angle(k) = 180/pi*circ_mean(pi/180*matLeftXYQ(:,3));
    Right_av_angle(k) = 180/pi*circ_mean(pi/180*matRightXYQ(:,3));
    Left_std_angle(k) = 180/pi*circ_std(pi/180*matLeftXYQ(:,3));%/sqrt(size(matLeftXYQ(:,3),1))
    Right_std_angle(k) = 180/pi*circ_std(pi/180*matRightXYQ(:,3));%/sqrt(size(matRightXYQ(:,3),1))
    
    strANGLE(k).width = Sw;
    strANGLE(k).deltaWidth = dw;
    strANGLE(k).Langle = matLeftXYQ(:,3);
    strANGLE(k).Rangle = matRightXYQ(:,3);

end
save([PC_path,'Curie\DESKTOP\HT1080\' 'pDdefect_150umEdge_Angle_FLIPPED_LR.mat'],'strANGLE');

%%
PC_path = 'D:\GD\'; 
i= 7;
load([PC_path,'Curie\DESKTOP\HT1080\' 'pDdefect_150umEdge_Angle_FLIPPED_LR.mat'])
% width_ax = [strANGLE.width];
% iLangle = strANGLE(1).Langle;
% Langle = cell2mat(struct2cell(strANGLE.Langle));
% Langle = cell2mat(strANGLE.Langle);
    figure(strANGLE(i).width); 
    polarhistogram(strANGLE(i).Langle*pi/180,90, 'Normalization','PDF');hold on
    polarhistogram(strANGLE(i).Rangle*pi/180,90, 'Normalization','PDF');hold off
    

Ql = 180/pi*circ_mean(pi/180*strANGLE(i).Langle);
Qr = 180/pi*circ_mean(pi/180*strANGLE(i).Rangle);
    title([num2str(strANGLE(i).width) ' \mum , Ql: ' num2str(Ql,2),'  Qr: ', num2str(Qr,3)])
%%    
for i=1:size(strANGLE,2)  
    Left_av_angle(i) = 180/pi*circ_mean(pi/180*strANGLE(i).Langle);
    Left_std_angle(i) = 180/pi*circ_var(pi/180*strANGLE(i).Langle);
    Right_av_angle(i) = 180/pi*circ_mean(pi/180*strANGLE(i).Rangle);
    Right_std_angle(i) = 180/pi*circ_var(pi/180*strANGLE(i).Rangle);
end
    
Right_av_angle(Right_av_angle<0) = Right_av_angle(Right_av_angle<0)+360;
figure(203);
plot([200 1100],[0 0],'r-.');hold on
plot([200 1100],[180 180],'r-.');hold on
errorbar(width_ax, 180+Left_av_angle,Left_std_angle,'o'); hold on
errorbar(width_ax, Right_av_angle,Right_std_angle,'o'); hold off

% [L1, ~] = boundedline(width_ax, Left_av_angle', Left_std_angle', 'alpha');hold on
% L1.LineWidth = 2; L1.Color = [0.1,0.5,0.8];
% [L2, ~] = boundedline(width_ax, Right_av_angle,Right_std_angle, 'alpha');hold off
% L2.LineWidth = 2; L2.Color = [0.8,0.5,0.1];
set(gca,'FontSize',16); axis tight
ylabel('$Angle\ (deg)$','Interpreter','latex','FontSize',28);
xlabel('$ Width\ (\mu m) $','Interpreter','latex','FontSize',28);
% figure(101);
% plot(width_ax, Left_av_angle); hold on
% plot(width_ax, Right_av_angle); hold off

%%
function [LeftXYQ, RightXYQ] = get_Edge_angle_file(filepathOP, Edge)

info = imfinfo(filepathOP);
Nn = numel(info);
LeftXYQ = cell(Nn,1);
RightXYQ = cell(Nn,1);

for k=1:Nn
    
    Ang = imread(filepathOP,k); % k
    Ang = fliplr(Ang);
    Ang = -Ang;
    [Left_ang, Right_ang] = get_Edge_angle(Ang, Edge);
    
    LeftXYQ{k,1} = [Left_ang,Left_ang,Left_ang];
    RightXYQ{k,1} = [Right_ang,Right_ang,Right_ang];
end
end


function [Left_ang, Right_ang] = get_Edge_angle(Ang, Edge)
try
    [ps_xA, ~, plocPsi_vecA, ~, ~, ~] = fun_get_pn_Defects(Ang);
    Ledge_select = ps_xA < Edge;   %1/4*w;
    Redge_select = ps_xA > size(Ang,2)- Edge;     %3/4*w;
    % Lps_x = ps_xA(Ledge_select);
    % Lps_y = ps_yA(Ledge_select);
    Left_ang = plocPsi_vecA(Ledge_select);
    % Rps_x = ps_xA(Redge_select);
    % Rps_y = ps_yA(Redge_select);
    Right_ang = plocPsi_vecA(Redge_select);
catch
end
end



