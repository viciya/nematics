%%
pathOP ={'E:\VICI\26072018_HT1080_stripes 200_600\27072018_TIFF',...
    'E:\VICI\01082018_HT1080 CROPPED\HT1080 CROPPED\01082018_TIFF',...
    'E:\VICI\HT1080_CROPPED 23112016'};

for i=1:size(pathOP,2)
    dirOP = dir([pathOP{i}  '\*LR_FLIP*.tif']);
    dirOPcell = struct2cell(dirOP)';
    filePathT = [dirOPcell(:,2), dirOPcell(:,1)];
    if i==1
        filePath = filePathT;
    else
        filePath = [filePath ; filePathT];
    end
end

%%
for i=1:size(filePath,1)
    filePathList{i,1} = [filePath{i,1},'\',filePath{i,2}];
end
%%
px2mic = 3*.74;
for i=1:size(filePathList,1)
    info = imfinfo(filePathList{i});ans
    width(i,1) = px2mic*info(1).Width;
end
%%
WidthRange = [250 300 400 500 600 700 800 1000];
k=4;

Sw = WidthRange(k); % selectd width
dw = .05*Sw; % define delta
Range  = find(width>Sw-dw & width<Sw+dw);
%%
filepathOP = filePathList{Range(1)}

