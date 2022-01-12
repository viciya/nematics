% %%  1 - Orient_s286_2.tif
% filepath = 'C:\Users\victo\Google Drive\DATA\HT1080\Orient\Orient_s286_2.tif';
% [px, py, ptheta,~,~,~] ...
%     = fun_get_pn_Defects_multitif(filepath,1);
% %%
% k=35;
% Ang = imread(filepathOP,k);  
% [px, py, ptheta,nx, ny, ntheta] = fun_get_pn_Defects_newDefectAngle(Ang);
% %% make list of Orient files
% PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% % PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
% % PC_path = 'D:\GD\';                         % RSIP notebook
% 
% addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
% pathOP = ([PC_path,'DATA\HT1080\Orient']);
% pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);
% 
% [dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);
% % %%
% px_sizeOP = 3* .74;
% Sw = 400; % selectd width
% dw = .05*Sw; % define delta
% Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);
% 
% file_list = [];
% for i= 1:length(Range)
%     file_list=[file_list;string(dirOP(Range(i)).name)];
% end
% fileID = fopen('C:\Users\victo\Downloads\filelist400.txt','w');
% fprintf(fileID,'%s \n',file_list);
% fclose(fileID);
%% make list Orient-PIV files by width range
px_sizeOP = 3*.74;
Sw=1500;
dw=.05*Sw;
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);
clearvars op_piv_pathlist
for i= 1:length(Range)
    op_piv_pathlist(i,1) = string(dirOP(Range(i)).name);
    op_piv_pathlist(i,2) = string(dirPIV(indX(Range(i),2)).name);

end
fileID = fopen([PC_path,'Curie\DESKTOP\HT1080\orient_piv_filelist_',num2str(Sw),'.txt'],'w');
fprintf(fileID,'%s \t %s \n',op_piv_pathlist');
fclose(fileID);

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

wset = [300,400,500,600,700,800,1000];
for k=1:length(wset)
    
    
px_sizeOP = 3* .74;
Sw = wset(k); % selectd width
dw = .05*Sw; % define delta
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);

px = [];
py = [];
ptheta = [];

for i= 1:min(length(Range),20) % do not rum for more than 20 files for each width
    disp([num2str(i),' **** from: ', num2str(length(Range))])
    disp('----------------------------------------------');

    filepathOP = [dirOP(Range(i)).folder '\' dirOP(Range(i)).name];
    info = imfinfo(filepathOP); % Place path to file inside single quotes
    iwidth = px_sizeOP*info(1).Width;
    
    % (filepathOP,1) - symmetric analysis, includes flipping
    % (filepathOP,0) - regular analysis    
    [ipx, ipy, iptheta,~,~,~] ...
        = fun_get_pn_Defects_multitif(filepathOP,1);
    
    px = [px;ipx*Sw/iwidth];
    py = [py;ipy];
    ptheta = [ptheta;iptheta];
end

folder_path = 'Curie\DESKTOP\HT1080\';
save_file = [PC_path, folder_path, 'symm_pDefect_angle_corr',num2str(Sw),'.mat'];
save(save_file);

end
%% plot Orientation distribution for single width
% % plot
% % info = imfinfo(filepathOP);
% % max_xpos = info.Width;
max_xpos = Sw/px_sizeOP;
dw = 30;
Q = [px(:), py(:), ptheta];
figure(2);
cla
nbins = 24;
bins = [-pi,pi];
polarhistogram(Q(Q(:,1)<dw,3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');  hold on
polarhistogram(Q(Q(:,1)>(max_xpos-dw),3)/180*pi+pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');  hold on
polarhistogram(Q(:,3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');  hold off
%% plot Orientation distribution all defects left/right/all
PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
folder_path = 'Curie\DESKTOP\HT1080\';
set_name = {
    'symm_pDefect_angle_corr300',...
    'symm_pDefect_angle_corr400',...
    'symm_pDefect_angle_corr500',...
    'symm_pDefect_angle_corr600',...
    'symm_pDefect_angle_corr700',...
    'symm_pDefect_angle_corr800',...
    'symm_pDefect_angle_corr1000'};

px_sizeOP = 3* .74;
color = lines(10);
figure(111)
sfolder = 'C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\figs_and_data\defect_orient\';
mkdir(sfolder)
for k=1:size(set_name,2)
   
set_number  = k;
load([PC_path, folder_path, set_name{set_number}, '.mat']);

max_xpos = Sw/px_sizeOP;
dw = 30;
nbins = 31;
bins = [-pi,pi];
theta = linspace(bins(1),bins(end),nbins);
Q = [px(:), py(:), ptheta];

subplot(3,3,k); 
[Nleft,~] = histcounts(Q(Q(:,1)<dw,3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');
pp = polarplot(theta,Nleft);hold on
pp.LineWidth=2;pp.Color=[.8,.1,.1];

[Nright,~] = histcounts(Q(Q(:,1)>(max_xpos-dw),3)/180*pi+pi,'NumBins',nbins,'BinLimits',bins+pi,'Normalization', 'PDF');
pp = polarplot(theta+pi,Nright);
pp.LineWidth=2;pp.Color=[.1,.1,.8];

[Nall,~] = histcounts(Q(:,3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');
pp = polarplot(theta,Nall);hold off
pp.LineWidth=2;pp.Color=[.1,.1,.1];

title(['$ width=', num2str(Sw), '\ \mu m$'],'Interpreter','latex')

% polarhistogram(Q(Q(:,1)<dw,3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');  hold on
% polarhistogram(Q(Q(:,1)>(max_xpos-dw),3)/180*pi+pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');  hold on
% polarhistogram(Q(:,3)/180*pi,'NumBins',nbins,'BinLimits',bins,'Normalization', 'PDF');  hold off

sname = ['def_orient_left_right_all_',num2str(Sw),'.txt'];
dlmwrite([sfolder,sname],[theta;Nleft;Nright;Nall]','delimiter','\t','precision',3)
end
set(gcf,'Color',[1 1 1]);
saveas(gcf,[[sfolder,'all'],'.png'])
