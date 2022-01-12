%%
Ddir = dir('V:\HT1080_small_stripes_glass_22112017\CROPPED\Orient');
folder_main = Ddir(1).folder;
filesInFolder = size(Ddir,1);
frame_per_hr = 4;
frames = 50;
dt=1;

px_size = .74*3;
qstep = 7;overlap = 1;
% qstep = 20;overlap = 6;
px2mic = px_size * frame_per_hr;
%%
clear var AngHist
mcount = 1;
for stripeL=30:20:200
LL = stripeL/px_size;
d = 10;

n=60;
ii =10;
count = 1;
for i=1:size(Ddir,1)
    if contains(Ddir(i).name, '.tif' )
        %         Ddir(i).name
        %         disp(['file: ' num2str(i) ' from: ' num2str(size(Ddir,1))]);
                filepath = [Ddir(i).folder '\' Ddir(i).name];
        
        info = imfinfo(filepath); % Place path to file inside single quotes
        Nn = numel(info);
        imWidth = info(1).Width;
        
        if imWidth>=LL-d && imWidth<=LL+d
            Ddir(i).name
            Ang = imread(filepath,1); % k
            [l,w] = size(Ang);

            
            kk=1;
            clear var midAng
            %         for k=Nn-frames:Nn
            for k=1:Nn
                % --------------------------ORIENT import ---------------------------------
                Ang = imread(filepath,k); % k
                Ang(Ang<0) = Ang(Ang<0)+180;
                temp = Ang(:,floor(end/2)-2:ceil(end/2)+2);
                midAng{k} = temp(:);
                kk=kk+1;disp([num2str(k) ':' num2str(Nn)])
            end
                    
            count = count+1;
            [AngHist{count,1},~] = histcounts(cell2mat(midAng'),60,'BinLimits',...
                    [0,180],'Normalization', 'probability');
        end
    end
end

%%
HistD = zeros(1,size(AngHist{2},2));
for i=2:size(AngHist,1)
    HistD = HistD + AngHist{i};
end
HistD = HistD/i;
Ang_ax = 1.5:3:180-1.5;
figure(11)
plot(Ang_ax,HistD);hold on
AngDistEXP{mcount,1}(:,1) = Ang_ax;
AngDistEXP{mcount,1}(:,2) = HistD;
%%
figure(10)
polarplot(Ang_ax*pi/180,HistD); hold on
mcount = mcount+1;
end

%% Plot from import MAT file
load('C:\Users\vici\Desktop\HT1080\midAngleHistograms_from30_step20_upto200_plusMinus10.mat');
for i=1:size(AngDistEXP,1)
    figure(11)
    polarplot(AngDistEXP{i,1}(:,1)*pi/180,AngDistEXP{i,1}(:,2)); hold on
    figure(12)
    plot(AngDistEXP{i,1}(:,1),AngDistEXP{i,1}(:,2)); hold on
end
    figure(11);hold off
    figure(12);hold off