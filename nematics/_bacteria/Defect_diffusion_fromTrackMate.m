%% IMPORT FILES
pathname = 'C:\Users\USER\Downloads\B-sub-sur-minus-in-supernatant-40X-100fps\TrackMate\spots_p.csv';
ALL = readtable(pathname);
%%
i=2;
im_size = 400;
track_ID = ALL{i:end,3};
frame = ALL{i:end,9};
x_pos = ALL{i:end,5};
y_pos = ALL{i:end,6};

%% defect motion, diffusion coeffecient (sub/super diffusive)
% ----    show tracks by motion  type
Lc_fit = fittype('c+a*x^b','dependent',{'y'},'independent',{'x'},'coefficients',{'a','b','c'});

pix2mic = 1;%1500/im_size;
frameRate = 1;
MSD = cell(max(track_ID),1); % square displacement
tracks_all = cell(max(track_ID),1); % square displacement
Frame = zeros(max(track_ID),1);

cc = 1;
for i=1:max(track_ID)
    x = x_pos(track_ID==i);
    y = y_pos(track_ID==i);

    if length(x)>10
        MSD{cc}(:,1:2) = [1/frameRate*[1:length(x)-1]',...
                    cumsum((pix2mic*(x(1:end-1)-x(2:end))).^2)...
                    + cumsum((pix2mic*(y(1:end-1)-y(2:end))).^2)]; 

        tracks_all{cc}(:,1:3) = [linspace(1,length(x),length(x))',x, y];
        
%         fy = MSD{cc,2};
       figure(1); plot(x,y,'-','Color',[0 0 1 .2]); hold on
       [f,fq] = fit(MSD{cc}(:,1),MSD{cc}(:,2),Lc_fit,'StartPoint',[1,1,0],'Lower',[0,0,0],'Upper',[inf,inf,0]);
        MSD_exp(cc) = f.b;
        MSD_diff(cc) = f.a;        
%         plot(fx ,f.c + f.a .* fx.^f.b); hold on
     
    cc=cc+1;
    end
end
MSD_exp(isnan(MSD_exp))=[];
MSD_diff(isnan(MSD_diff))=[];

tracks_all = tracks_all(1:cc-1,1);

% figure(1);
% xlabel('time(hr)','FontSize',20);ylabel('MSD (\mum^2)','FontSize',20);
% set(gca, 'XScale', 'log');set(gca, 'YScale', 'log');
% axis tight
%%
% ------  defect exponent distibution
figure(2)
histogram(MSD_diff,40,'Normalization','pdf'); hold on
xlabel('Diff. const. (\mum^2/hr)','FontSize',20); ylabel('PDF','FontSize',20);
axis tight; %hold  off
figure(3)
histogram(MSD_exp,40,'Normalization','pdf'); hold on
xlabel('EXP','FontSize',20); ylabel('PDF','FontSize',20);
axis tight; %hold  off
%% 
% take all MSD tracks and make an average
[max_size, max_index] = max(cellfun('size', MSD, 1));

mMSD = cell2mat(MSD);
[UmMSD1,~,idx]  = unique(mMSD(:,1));
N = histc(mMSD(:,1), UmMSD1); % repetition number
UmMSD = [UmMSD1, accumarray(idx, mMSD(:,2),[],@mean), accumarray(idx,mMSD(:,2),[],@std)./sqrt(N)];
UmMSD = sortrows(UmMSD,1);
figure(4);errorbar(UmMSD(:,1),UmMSD(:,2),UmMSD(:,end)); hold on
xlabel('time(hr)','FontSize',20); ylabel('MSD (\mum^2)','FontSize',20);
set(gca, 'XScale', 'log'); set(gca, 'YScale', 'log');
axis tight
%%


