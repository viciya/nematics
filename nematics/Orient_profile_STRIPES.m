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
n=130;

for i=1%:size(dirOP,1)
        filepath = [dirOP(indX(i,1)).folder '\' dirOP(indX(i,1)).name];
        
        info = imfinfo(filepath); % Place path to file inside single quotes
        Nn = numel(info);
        
        Ang = imread(filepath,1); % k
        [l,w] = size(Ang);
        Ang_profile = zeros(1,size(Ang,2));
        Ang_std = Ang_profile;
        
        %         OP = ordermatrixglissant_overlap(pi/180*Ang,qstep,overlap);
%         OP = ordermatrixglissant(pi/180*Ang,qstep);
%         OP_profile = zeros(1,size(OP,2));
%         OP_std = OP_profile;
        
        kk=1;
        
        %         for k=Nn-frames:Nn
        for k=1:Nn
            %             % --------------------------ORIENT import ---------------------------------
            Ang = imread(filepath,k); % k
            Ang(Ang<0) = Ang(Ang<0)+180;
            Ang_profile = Ang_profile + mean(Ang,1);
            Ang_std = Ang_std + std(Ang,1);
            %             OP = ordermatrixglissant_overlap(pi/180*Ang,qstep,overlap);
%             OP = ordermatrixglissant(pi/180*Ang,qstep);
            %             surf(OP); shading interp;colormap jet;axis equal;axis tight;view(2); colorbar
%             OP_profile = OP_profile + mean(OP,1);
%             OP_std = OP_std + std(OP,1);
            kk=kk+1;disp([num2str(k) ':' num2str(Nn)])
        end
        
        Ang_profile = Ang_profile/kk;
        Ang_std = Ang_profile/kk^.5;
%         OP_profile = OP_profile/kk;
%         OP_std = OP_profile/kk^.5;
        
        %         figure(1)
        %         errorbar(1:length(OP_profile),OP_profile,OP_std);hold on
        %         figure(2)
        %         errorbar(1:length(Ang_profile),Ang_profile,Ang_std);hold on
        
        XX = px_size*[1:length(Ang_profile)];
        EXP{i-2,1} = px_size*length(Ang_profile);
        EXP{i-2,2} = [XX', Ang_profile',Ang_std'];
%         XX_OP = px_size*[1:length(OP_profile)];
%         EXP{i-2,3} = [XX_OP', OP_profile',OP_std'];
        %                 EXP{i-2,4} = [XX'/XX(end), u_profile',u_std'];
        %                 EXP{i-2,5} = [XX'/XX(end), v_profile',v_std'];

end
% hold off
% %%
% figure(1)
% xlabel('Width (\mum)','FontSize',20); ylabel('v_x (\mum/hr)','FontSize',20);axis tight;hold off
% figure(2)
% xlabel('Width (\mum)','FontSize',20); ylabel('v_y (\mum/hr)','FontSize',20);axis tight;hold off
% figure(1)
% xlabel('Width (x/L)','FontSize',20); ylabel('v_x/v_{x,max}','FontSize',20);axis tight;hold off
% figure(2)
% xlabel('Width (x/L)','FontSize',20); ylabel('v_y/v_{y,max}','FontSize',20);axis tight;hold off
%%
plot(Ang_profile)
%%
% SORT BY WIDTH
[~, ind] = sort(cell2mat(EXP(:,1)));
ind = flip(ind);
norm = 453;
mid_width_dAng_OP = zeros(size(EXP,1),3);
edge_width_dAng_OP = mid_width_dAng_OP;
d = 3;
for k=1:size(EXP,1)
    i=ind(k);
    figure(1)
    pl = plot(EXP{i,2}(:,1)-EXP{i,2}(end,1)/2,90-EXP{i,2}(:,2));hold on
    pl.Color = [1-EXP{i,2}(end,1)/norm 0 EXP{i,2}(end,1)/norm .4];
    pl.LineWidth = 2;
    figure(2)
    pl = plot(EXP{i,3}(:,1)-EXP{i,3}(end,1)/2,EXP{i,3}(:,2));hold on
    pl.Color = [1-EXP{i,2}(end,1)/norm 0 EXP{i,2}(end,1)/norm .4];
    pl.LineWidth = 2;
    
    mid_dAng = abs(90-mean2(EXP{i,2}(floor(end/2)-d:ceil(end/2)+d,2)));
    mid_OP = mean2(EXP{i,3}(floor(end/2)-d:ceil(end/2)+d,2));
    mid_width_dAng_OP(k,:) = [EXP{i,1},mid_dAng,mid_OP];
    
    edge_dAng = abs(90-mean2(EXP{i,2}([1:d, end-d:end],2)));
    edge_OP = mean2(EXP{i,3}([1:d, end-d:end],2));
    edge_width_dAng_OP(k,:) = [EXP{i,1},edge_dAng,edge_OP];    
end
figure(1);
xlabel('Width (\mum)','FontSize',20); ylabel('Angle (deg)','FontSize',20); axis tight;hold off
figure(2);
xlabel('Width (\mum)','FontSize',20); ylabel('Order Parameter','FontSize',20); axis tight;hold off
%%
[Uw,~,idx]  = unique(mid_width_dAng_OP(:,1));
N = histc(mid_width_dAng_OP(:,1), Uw); % repetition number
Umid_width_dAng = [Uw, accumarray(idx, mid_width_dAng_OP(:,2),[],@mean), accumarray(idx,mid_width_dAng_OP(:,2),[],@std)./sqrt(N)];
Umid_width_dAng = sortrows(Umid_width_dAng,1);
Umid_width_OP = [Uw, accumarray(idx, mid_width_dAng_OP(:,3),[],@mean), accumarray(idx,mid_width_dAng_OP(:,3),[],@std)./sqrt(N)];
Umid_width_OP = sortrows(Umid_width_OP,1);
Uedge_width_dAng = [Uw, accumarray(idx, edge_width_dAng_OP(:,2),[],@mean), accumarray(idx,edge_width_dAng_OP(:,2),[],@std)./sqrt(N)];
Uedge_width_dAng = sortrows(Uedge_width_dAng,1);
Uedge_width_OP = [Uw, accumarray(idx, edge_width_dAng_OP(:,3),[],@mean), accumarray(idx,edge_width_dAng_OP(:,3),[],@std)./sqrt(N)];
Uedge_width_OP = sortrows(Uedge_width_OP,1);

figure(3)
p1 = errorbar(Umid_width_dAng(:,1),Umid_width_dAng(:,2),Umid_width_dAng(:,3));p1.LineWidth = 3;hold on
p2 = errorbar(Uedge_width_dAng(:,1),Uedge_width_dAng(:,2),Uedge_width_dAng(:,3));p2.LineWidth = 3;
% plot(mid_width_dAng_OP(:,1),mid_width_dAng_OP(:,2));hold on
% plot(edge_width_dAng_OP(:,1),edge_width_dAng_OP(:,2));
xlabel('Width (\mum)','FontSize',20); ylabel('Angle tilt','FontSize',20);%axis([0 300 0 15]);
hold off
figure(4)
p3 = errorbar(Umid_width_OP(:,1),Umid_width_OP(:,2),Umid_width_OP(:,3));p3.LineWidth = 3;hold on
p4 = errorbar(Uedge_width_OP(:,1),Uedge_width_OP(:,2),Uedge_width_OP(:,3));p4.LineWidth = 3;
xlabel('Width (\mum)','FontSize',20); ylabel('Order parameter','FontSize',20);%axis([0 300 .8 1]);
hold off
% %%
% for k=1:size(EXP,1)
%     i=ind(k);
%     pp(k)=EXP{i,2}(end,1);
% end
% %%
% %% figure out the OP value
% M= 180+10*rand(100,100);
% Mr=M*pi/180;
% Q=ordermatrixglissant_overlap(Mr,10,2);
% mean2(Q)
% 
% Q1=sqrt((sum(sum(cos(2*Mr))))^2 ...
%     +(sum(sum(sin(2*Mr))))^2)/(size(Mr,1)*size(Mr,2));
% mean2(Q1)