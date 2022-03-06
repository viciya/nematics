% RUN EXAMPLE FILES:
%% Load raw image corresponing to angular map
filepath = 'C:\Users\idan1\OneDrive\Desktop\ניסוי\new data\858 or\Orient_1-00354.tif'
Ang = imread(filepath);
[f,name,ext] = fileparts(filepath);
dir_info  = dir(['C:\Users\idan1\OneDrive\Desktop\ניסוי\new data\858\*',name(8:end),'.tif']);
raw_img_path = [dir_info.folder '\' dir_info.name];
imshow(imread(raw_img_path)); hold on
plot_nematic_field(Ang);
%% Order parameter map from angular map
filepath = '.\example_images\orient\Orient_1_X1.tif'
Ang = imread(filepath);
qq = order_parameter(Ang,10,3);
imshow(qq); hold on

%% Find defects from order angular map (plotted on of order parameter)
filepath = '.\example_images\orient\Orient_1_X1.tif'
Ang = imread(filepath);
qq = order_parameter(Ang,10,3);
imshow(qq); hold on
[xf, yf] = detectDefectsFromAngle(Ang);
scatter(xf, yf, "filled")

%% Classification of +1/2 and -1/2 defects
filepath = '.\example_images\orient\Orient_1_X1.tif'
Ang = imread(filepath);
[f,name,ext] = fileparts(filepath);
dir_info  = dir(['.\example_images\raw\*',name(8:end),'.tif']);
raw_img_path = [dir_info.folder '\' dir_info.name];
imshow(imread(raw_img_path)); hold on
plot_nematic_field(Ang);

% Display defects
[ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec] = ...
    fun_get_pn_Defects_newDefectAngle(Ang);
%         fun_get_pn_Defects_newDefectAngle_blockproc(Ang);

plot_pdefect_ndefect(ps_x, ps_y, plocPsi_vec,...
    ns_x, ns_y, nlocPsi_vec);
%     fun_defectDraw(ps_x, ps_y, plocPsi_vec);

%% 
% Optical Flow load
load('C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\OptFlow\1_X1.mat');
[Xu,Yu] = meshgrid(1:size(Ang,2),1:size(Ang,1));
step = 13;
q7 = quiver( ...
    Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
    u(1:step:end,1:step:end),v(1:step:end,1:step:end), ...
    3 ...
    );
q7.LineWidth=1; q7.Color = [.9 .2 .0]; 
%%
[filepath,name,ext] = fileparts(filepath);
out_path_name = fullfile(filepath+"1",name,".png");
git
saveas(gcf,'C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\Orient1\Orient_1_X1.png');
%%
% nx = cos(Ang);
% ny = -sin(Ang);
%
% LIC = fun_getLIC(nx,ny);
% imageplot(LIC,''); hold on

%%
% Get defects in all folder files save positions and angles in .mat file
MAT_OUTPUT = "C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\pnDefects_circ10_opStep10_Thr023_XX.mat";

SAVE = false; % if true figure will be saved into "Orient1" folder

Ddir = dir('C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\Orient\*.tif');
folder_main = Ddir(1).folder;
filesInFolder = size(Ddir,1);
defNum = cell(size(Ddir,1),6);

for i=1:size(Ddir,1)
    filepath = [Ddir(i).folder '\' Ddir(i).name];
    disp(['file: ' num2str(i) ' from: ' num2str(size(Ddir,1))]);

    Ang = imread(filepath);

    try
        [ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec] = ...
            fun_get_pn_Defects_newDefectAngle(Ang);
        %         fun_get_pn_Defects_newDefectAngle_blockproc(Ang);
        [defNum{i,:}] = deal(ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec);

    catch
        disp('no defects')
    end

    if SAVE
        % its just white background
        % to make image of same size of the raw image
        imshow(Ang~=0); hold on;

        [Xu,Yu] = meshgrid(1:size(Ang,2),1:size(Ang,1));
        step = 12; O_len = .7;
        q6 = quiver( ...
            Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
            cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)), ...
            O_len ...
            );
        q6.LineWidth=1; q6.Color = [.7 .7 .7]; q6.ShowArrowHead='off';
        plot_pdefect_ndefect(ps_x, ps_y, plocPsi_vec,...
            ns_x, ns_y, nlocPsi_vec);
        %     fun_defectDraw(ps_x, ps_y, plocPsi_vec);

        [f,name,ext] = fileparts(filepath);
        out_path_name = fullfile(f+"1", name +".png");
        disp([">", filepath]);
        disp([">", out_path_name]);
        saveas(gcf, out_path_name);
        cla;
    end

end
save(MAT_OUTPUT, 'defNum')
disp("Done")

%%
% Get defects in all folder files save positions and angles in .mat file
MAT_OUTPUT = "C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\pnDefects_circ10_opStep10_Thr023.mat";
load(MAT_OUTPUT);
plength = .08;
nlength = .05;
sz = 6;
line_width = 3;

QUIVER = false;

Ddir = dir('C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\Orient\*.tif');
folder_main = Ddir(1).folder;
filesInFolder = size(Ddir,1);


% sort by file names
[~, reindex] = sort_nat({Ddir.name}); 
Ddir = Ddir(reindex);
defNum = defNum(reindex,:);

for i=1:5%size(Ddir,1)
    disp(Ddir(i).name);
    filepath = [Ddir(i).folder '\' Ddir(i).name];
    disp(['file: ' num2str(i) ' from: ' num2str(size(Ddir,1))]);

    [f,name,ext] = fileparts(filepath);
    dir_info  = dir(['C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\*\',name(8:end),'.tif']);
    raw_img_path = [dir_info.folder '\' dir_info.name];
    image = imread(raw_img_path);
    imshow(image); hold on;
%     imshow(ordermatrixglissant_overlap(Ang, 10, 3)<.3); hold on;
%     im2 = ordermatrixglissant_overlap(Ang, 10, 3);
%     imshow(im2); hold on;
%     s = regionprops('table', im2<.28,'centroid');
%     scatter(s.Centroid(:,1),s.Centroid(:,2), "filled")

    [ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec] = deal(defNum{i,:});

    if QUIVER
        Ang = imread(filepath);
        [Xu,Yu] = meshgrid(1:size(Ang,2),1:size(Ang,1));
        step = 12; O_len = .7;
        q6 = quiver( ...
            Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
            cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)), ...
            O_len ...
            );
        q6.LineWidth=1; q6.Color = [.7 .7 .7]; q6.ShowArrowHead='off';

        half_width = floor(size(Xu,2)/2);
        q6 = quiver( ...
            Xu(1:step:end,1:step:half_width),Yu(1:step:end,1:step:half_width),...
            cos(Ang(1:step:end,1:step:half_width)),-sin(Ang(1:step:end,1:step:half_width)), ...
            O_len ...
            );
        q6.LineWidth=1; q6.Color = [.7 .2 .0]; q6.ShowArrowHead='off';
    end

    plot_pdefect_ndefect(ps_x, ps_y, plocPsi_vec,...
        ns_x, ns_y, nlocPsi_vec,...
        plength, nlength, sz, line_width);
    
    set(gcf, 'Position', [10 10 floor(size(Ang,2)/1.5)+10 floor(size(Ang,1)/1.5)+10])
    [f,name,ext] = fileparts(filepath);
    out_path_name = fullfile(f+"2", name +".png");
    disp([">", filepath]);
    disp([">", out_path_name]);
%     saveas(gcf, out_path_name);
% break;
    cla;
end
% save(MAT_OUTPUT, 'defNum')
disp("Done")
%%
% sort by file names
[~, reindex] = sort_nat({Ddir.name}); 
Ddir = Ddir(reindex);
defNum = defNum(reindex,:);

% Count defect number evolution in time
defn = zeros(size(Ddir,1), 1, 'uint8');
pdef = zeros(size(Ddir,1), 1, 'uint8');
ndef = zeros(size(Ddir,1), 1, 'uint8');
for i=1:size(Ddir,1)
    defn(i) = size(defNum{i,6},1) + size(defNum{i,1},1);
    pdef(i) = size(defNum{i,1},1);
    ndef(i) = size(defNum{i,6},1);
end
plot(defn);hold on
plot(pdef);
plot(ndef);
xlabel('frame');
ylabel('toatal defect number');
