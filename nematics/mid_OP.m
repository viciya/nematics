%% correspond PIV and Orientation names
% dirOP = dir(['C:\Users\vici\Google Drive\DATA\HT1080\Orient'  '\*.tif']);
% dirPIV = dir(['C:\Users\vici\Google Drive\DATA\HT1080\PIV_DATs'  '\*.mat']);

dirOP = dir(['C:\Users\victo\Google Drive\DATA\HT1080\Orient'  '\*.tif']);
dirPIV = dir(['C:\Users\victo\Google Drive\DATA\HT1080\PIV_DATs'  '\*.mat']);

for i = 1:size(dirOP,1)
    
    nameOP = dirOP(i).name;
    p1=strfind(nameOP,'s');
    p2=strfind(nameOP,'.tif');
    patternOP = nameOP(p1:p2);
    
    %     pA=strfind(nameOP,'Orient');
    %     pB=strfind(nameOP,'oldOrient');
    pC=strfind(nameOP,'old');
    pD=strfind(nameOP,'09_07_2018_Orient');
    pE=strfind(nameOP,'26072018_Orient');
    pF=strfind(nameOP,'01082018_Orient');
    
    if ~isempty(pF)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name,'01082018') &&...
                    contains(dirPIV(j).name, patternOP)
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;
            end
        end
    elseif ~isempty(pE)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name,'26072018') &&...
                    contains(dirPIV(j).name, patternOP)
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;
            end
        end
    elseif ~isempty(pD)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name,'09072018') &&...
                    contains(dirPIV(j).name, patternOP)
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;
            end
        end
    elseif ~isempty(pC)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name,'old') &&...
                    contains(dirPIV(j).name, patternOP)
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;
            end
        end
    elseif isempty(pC)&isempty(pD)&isempty(pE)&isempty(pF)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name, patternOP)&&...
                    ~contains(dirPIV(j).name,'01082018') &&...
                    ~contains(dirPIV(j).name,'26072018') &&...
                    ~contains(dirPIV(j).name,'09072018') &&...
                    ~contains(dirPIV(j).name,'old')
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;
            end
        end
    end
end
%%
px_sizeOP = .74*3;
for i=1:10:size(dirOP,1)
    
    disp(['file: ' num2str(indX(i,1)) ' from: ' num2str(size(dirOP,1))]);
    filepathOP = [dirOP(indX(i,1)).folder '\' dirOP(indX(i,1)).name];
    info = imfinfo(filepathOP); % Place path to file inside single quotes
    Nn = numel(info);
    Orient_area(i,1) = px_sizeOP^2*info(1).Width*info(1).Height;
    Orient_width(i,1) = px_sizeOP*info(1).Width;
    Ang = imread(filepathOP,1); % k
    [l,w] = size(Ang);
    for k=1:Nn          
        Ang = imread(filepathOP,k); 
        if ~any( Ang(:)>2 ) % chek if Ang is in RAD
            Ang=Ang*180/pi;
        end
        mAng = Ang(:, floor(w/2)-5:ceil(w/2)+5);
        mAng(mAng<0) = mAng(mAng<0)+180;
        mAng_frame(k,1) = mean2(mAng);
        mAngVec = reshape(mAng,[size(mAng,1)*size(mAng,2) 1]);
        OP_frame(k,1) = sqrt(sum(sum(cos(2*mAngVec*pi/180)))^2 ...
            +sum(sum(sin(2*mAngVec*pi/180)))^2)/(size(mAng,1)*size(mAng,2));
    end
    OP_mid(i,1) = mean(OP_frame);
    OP_mid_std(i,1) = std(OP_frame)/Nn^.5;%    
end
w_OP_OPstd= [Orient_width, OP_mid,OP_mid_std];
[w_OP_OPstd, indByWidth] = sortrows(w_OP_OPstd,1);

cd=150;% shift jet colors
c1=jet(size(OP_mid,1)+cd);

c=c1(1:end-cd,:);
figure(99);
scatter(w_OP_OPstd(:,1),w_OP_OPstd(:,2),15,c,'filled');set(gca,'Fontsize',18);
xlabel('width'); ylabel('Order Parameter');%axis([0 1 -10 10]);
axis tight;%hold off	

%%
px_sizeOP = .74*3;
i=61;
    filepathOP = [dirOP(indX(i,1)).folder '\' dirOP(indX(i,1)).name];
    info = imfinfo(filepathOP); % Place path to file inside single quotes
    Nn = numel(info);
    Orient_area(i,1) = px_sizeOP^2*info(1).Width*info(1).Height;
    Orient_width(i,1) = px_sizeOP*info(1).Width;
    Ang = imread(filepathOP,1); % k
    [l,w] = size(Ang);
    disp(['file: ' num2str(indX(i,1)) ' from: ' num2str(size(dirOP,1))]);
    for k=1:Nn
        Ang = imread(filepathOP,k); % k
        mAng = Ang(:, floor(w/2)-5:ceil(w/2)+5);
        mAng(mAng<0) = mAng(mAng<0)+pi;
        mAng_frame(k,1) = mean2(mAng);
        mAngVec = reshape(mAng,[size(mAng,1)*size(mAng,2) 1]);
        OP_frame(k,1) = sqrt(sum(sum(cos(2*mAngVec*pi/180)))^2 ...
            +sum(sum(sin(2*mAngVec*pi/180)))^2)/(size(mAng,1)*size(mAng,2));
    end
    mean(OP_frame)
    std(OP_frame)/Nn^.5;%    
	