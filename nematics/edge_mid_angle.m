%%  profile for specific width
% data set from 'correlate_OP_PIV_Defect.m'
% load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_OP_correlation_new3.mat');
% % data set from 'energy_enstrophy.m' (without velocity filtering and net flow removed)
% load("C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\shear_Ek_Ev_unfiltered1.mat");
% 
% V_OP_mAng = [width, Ang_mid,vR-vL];
% [V_OP_mAng, indByWidth] = sortrows(V_OP_mAng,1);
%% HISTOGRAM Edge and Mid

folderOP = 'C:\Users\victo\Google Drive\DATA\HT1080\Orient'; 

px_sizeOP = 3* .74;
frame_per_hr = 4;
px2mic = px_sizeOP * frame_per_hr;

SAVE = false;
Sw_all = [300,400,500,600,700,800,1000,1500]; %/ px_sizeOP; RUN FOR Sw=280,Dw = 0.1  V_OP_mAng(:,3)> 3
stats = zeros(length(Sw_all),3);

for n = 8%:length(Sw_all)
clearvars edge_ang mid_ang file_dirs
    Sw = Sw_all(n);
Dw = 0.05 * Sw;

sRange = indByWidth(V_OP_mAng(:,1)>Sw-Dw & V_OP_mAng(:,1)<Sw+Dw &...
                    V_OP_mAng(:,2)>90 & V_OP_mAng(:,3)> 0);

count = 1;
edge = 10;


for i=1:length(sRange)
    file_dirs{i,1} = [folderOP '\' dirOP(sRange(i)).name];
end

[edge_ang, mid_ang] = get_edge_mid_angles_files(file_dirs, edge);

edge_Ang_mat = edge_ang;
mid_Ang_mat = mid_ang;
edge_mat = edge_Ang_mat(:);
%% PLOT HISTOGRAM AS LINE



% if SAVE
  
%     figure(Sw+3);
    figure(999);
    
    bins = 60;
    [Nm,edges] = histcounts(mid_Ang_mat,bins,'BinLimits',[-pi/2 pi/2], 'Normalization', 'probability');
    theta_mid = mean([edges(1:end-1); edges(2:end)]);
    pp = polarplot(theta_mid, Nm);hold on
    pp.LineWidth=2;pp.Color=[.8,.1,.1];

    figure(998);
    [Nedge,edges] = histcounts(edge_mat,bins,'BinLimits',[-pi pi], 'Normalization', 'probability');
    theta_edge = mean([edges(1:end-1); edges(2:end)]);
    pp = polarplot(theta_edge, Nedge);
    pp.LineWidth=2;pp.Color=[.1,.1,.8];
    
    thetalim([-10 190]);ax = gca; ax.RTickLabel = {''};set(gca,'Fontsize',18);
    set(gcf,'Color',[1 1 1]);
    legend({'midline',['edge', sprintf('\n'),...
        'mean: ', num2str(circ_mean(edge_mat)*180/pi),sprintf('\n'),...
        'std: ', num2str(circ_std(edge_mat)*180/pi)]}...
        ,'Fontsize',13, 'Location', 'southoutside');
    
if SAVE    
    
    save_dir = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\Supplement\edge_ang\';
    saveas(gcf,[save_dir, num2str(Sw),'.svg'])
    dlmwrite([save_dir, num2str(Sw),'egde_mid.txt'],...
        [theta_edge',Nedge', theta_mid', Nm'],'delimiter','\t','precision',3);
  
end

stats(n,:) = [Sw,...
    circ_mean(edge_mat)*180/pi,...
    circ_std(edge_mat)*180/pi];

end

if SAVE
    save_dir = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\Supplement\edge_ang\';
    dlmwrite([save_dir,'edge_stats.txt'],...
        stats,'delimiter','\t','precision',3);
    figure();
    errorbar(stats(1:end-1,1),...
        stats(1:end-1,2),...
        stats(1:end-1,3))
    ylabel('$ Angle\ (deg) $','Interpreter','latex','FontSize',28);
    xlabel('$ Width\ (\mu m) $','Interpreter','latex','FontSize',28);
end


function file_list = get_file_list(folder, names, index_list)
for i=1:length(index_list)
    file_list{i,1} = [folder '\' names(index_list(i))];
end
end

function [edge_ang, mid_ang] = get_edge_mid_angles_files(file_dirs, edge)

edge_ang_cell = cell(length(file_dirs),1);
mid_ang_cell = cell(length(file_dirs),1);

for i=1:length(file_dirs)
    [edge_ang_temp, mid_ang_temp] = get_edge_mid_angles_frames(file_dirs{i}, edge);
    edge_ang_cell{i,1} = edge_ang_temp;
    mid_ang_cell{i,1} = mid_ang_temp;

end
edge_ang = cell2mat(edge_ang_cell);
mid_ang = cell2mat(mid_ang_cell);
end

function [edge_ang, mid_ang] = get_edge_mid_angles_frames(filepathOP, edge)

info = imfinfo(filepathOP);
Nn = numel(info);

edge_ang_cell = cell(Nn,1);
mid_ang_cell = cell(Nn,1);

for k=1:Nn
    Ang = imread(filepathOP,k);
    if any( Ang(:)>4 ) % check if Ang is in RAD
        Ang = Ang * pi/180;
    end
%     Ang(Ang<0) = Ang(Ang<0)+pi;
    
    [edge_ang_temp, mid_ang_temp] = get_edge_mid_angles(Ang, edge);
    
    edge_ang_cell{k,1} = edge_ang_temp;
    mid_ang_cell{k,1} = mid_ang_temp;
    
    
end
edge_ang = cell2mat(edge_ang_cell);
mid_ang = cell2mat(mid_ang_cell);
end

function [edge_ang, mid_ang] = get_edge_mid_angles(Ang, edge)
%     edge is [N,2] matrix [N,left] and [N, right]
%     mid is [N,1] matrix
    edge_ang = [reshape(Ang(:,1:edge),[],1), reshape(Ang(:,end-edge+1:end),[],1)];
    mid_ang = reshape(Ang(:,round(end/2)-round(edge/2):round(end/2)+round(edge/2)),[],1);

end