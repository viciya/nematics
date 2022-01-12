%% make datasets with Orienttation of +1/2 defects
PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
% PC_path = 'D:\GD\';                         % RSIP notebook

addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

[dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);
%%
px_sizeOP = 3* .74;

wset = [300,400,500,600,700,800,1000,1500];
for k=7%:length(wset)
    
    px_sizeOP = 3* .74;
    Sw = wset(k); % selectd width
    dw = .05*Sw; % define delta
    Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);

    ix = []; iy = []; itheta = []; itime = []; idx = [];
    
    for i = 1:min(length(Range),100/(2*k)) % do not run for more than 20 files for each width
        disp([num2str(i),' **** from: ', num2str(min(length(Range),100/(2*k)))])
        disp('----------------------------------------------');
        
        
        filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
        info = imfinfo(filepathOP); % Place path to file inside single quotes
        iwidth = px_sizeOP*info(1).Width;
        
        % (filepathOP,1) - symmetric analysis, includes flipping
        % (filepathOP,0) - regular analysis
        
        tx = []; ty = []; ttheta = []; time = [];
        for t=1:numel(info)
            disp([num2str(i),'(',num2str(t),')']);
            ang = imread(filepathOP,t);
%             [x,y,theta] = get_p_defect_symm(ang);
            [x,y,theta,~,~,~] = fun_get_pn_Defects_newDefectAngle_blockproc(ang);
            tx = [tx;x*Sw/iwidth];
            ty = [ty;y];
            ttheta = [ttheta;theta];
            time = [time;t*ones(size(x))];
        end
        ix = [ix;tx];
        iy = [iy;ty];
        itheta = [itheta;ttheta];
        itime = [itime;time];
        idx = [idx;i*ones(size(tx))];
    end
    
    folder_path = 'Curie\DESKTOP\HT1080\';
    save_file = [PC_path, folder_path, 'symm_pDefect_x_y_angle_frame_idx_2_',num2str(Sw),'.txt'];
    dlmwrite(save_file,[ix,iy,itheta,itime,idx],'delimiter','\t','precision',4);
    
end

%% plot Orientation distribution for single width
px_sizeOP = 3* .74;

PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
folder_path = 'Curie\DESKTOP\HT1080\';
name = 'symm_pDefect_x_y_angle_frame_idx_2_500.txt';
split_name = split(name,[".","_"]);

T = table2array(readtable([PC_path, folder_path, name]));
px = T(:,1);
py = T(:,2);
ptheta = T(:,3);
% 
Sw = str2double(split_name{end-1});
% %% run from here without loading data
% px = ix;
% py = iy;
% ptheta = itheta;
max_xpos = Sw/px_sizeOP;

dw = 70;
Q = [px(:), py(:), ptheta];
figure(2);
cla
nbins = 24;
bins = [-pi,pi];
polarhistogram(Q(Q(:,1)<dw,3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');  hold on
% polarhistogram(Q(Q(:,1)>(max_xpos-dw),3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');  hold on
% polarhistogram(Q(:,3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');  hold off

% figure();histogram(Q(Q(:,1)< dw,3),30);  hold on

%% plot Orientation distribution all defects left/right/all
PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
folder_path = 'Curie\DESKTOP\HT1080\';
name_template = 'symm_pDefect_x_y_angle_frame_idx_2_';
width_set = [300,400,500,600,700,800,1000,1500];
for i = 1:length(width_set)
    set_name{i} = [name_template, num2str(width_set(i)),'.txt'];
end

px_sizeOP = 3* .74;
color = lines(10);
figure('Position', [10 10 700 700])
sfolder = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\defect_orient_2\';
mkdir(sfolder)
for k=1:size(set_name,2)

    set_number  = k;
%     load([PC_path, folder_path, set_name{set_number}, '.mat']);
    T = table2array(readtable([PC_path, folder_path, set_name{set_number}]));
    px = T(:,1);
    py = T(:,2);
    ptheta = T(:,3);
    fname = split(set_name{set_number},[".","_"]);
    Sw = str2double(fname{end-1});

    max_xpos = Sw/px_sizeOP;
    dw = 50;
    nbins = 25;
    bins = [-pi, pi];
    theta = linspace(bins(1),bins(end),nbins);
    Q = [px(:), py(:), ptheta];

    subplot(3,3,k);
    [Nleft,~] = histcounts(Q(Q(:,1)<dw,3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');
    pp = polarplot(theta,Nleft);hold on
    pp.LineWidth=2;pp.Color=[.8,.1,.1];

    [Nright,~] = histcounts(Q(Q(:,1)>(max_xpos-dw),3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');
    pp = polarplot(theta,Nright);
    pp.LineWidth=2;pp.Color=[.1,.1,.8];

    [Nall,~] = histcounts(Q(:,3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');
    pp = polarplot(theta,Nall);hold off
    pp.LineWidth=2;pp.Color=[.1,.1,.1];

    title(['$ width=', num2str(Sw), '\ \mu m$'],'Interpreter','latex')

    % polarhistogram(Q(Q(:,1)<dw,3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');  hold on
    % polarhistogram(Q(Q(:,1)>(max_xpos-dw),3)/180*pi+pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');  hold on
    % polarhistogram(Q(:,3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');  hold off

%     sname = ['def_orient_left_right_',num2str(Sw),'.txt'];
%     dlmwrite([sfolder,sname],[theta;Nleft;Nright;Nall]','delimiter','\t','precision',3)
end
set(gcf,'Color',[1 1 1]);
% saveas(gcf,[[sfolder,'all'],'.png'])

%%
function [x,y,q] = get_p_defect_symm(ang)
x = [];
y = [];
q = [];
[ix, iy, iq, ~, ~, ~] = ...
    fun_get_pn_Defects_newDefectAngle(ang);
x = [x;ix];
y = [y;iy];
q = [q;iq];

[ix, iy, iq,~, ~, ~] = ...
    fun_get_pn_Defects_newDefectAngle(rot90(ang,2));
x = [x;size(ang,2)-ix];
y = [y;size(ang,1)-iy];
q = [q;iq+180];
end