%%
% Ddir = dir(['V:\HT1080_small_stripes_glass_22112017\CROPPED\PIV_DATs'  '\*.mat']);
Ddir = dir(['G:\DATA\HT1080\PIV_DATs'  '\*.mat']);
% Ddir = dir('V:\HT1080_small_stripes_glass_22112017\CROPPED\CROPPED 23112016\PIV_DATs');
folder_main = Ddir(1).folder;
filesInFolder = size(Ddir,1);
px_size = .74;
frame_per_hr = 4;
px2mic = px_size * frame_per_hr;
frames = 1;
dt = 1;
%%
n = 6;
count = 1;

colors = jet(size(Ddir,1));
for i=1:size(Ddir,1)
%     if contains(Ddir(i).name, '.mat' )
        Ddir(i).name
        disp(['file: ' num2str(i) ' from: ' num2str(size(Ddir,1))]);
        filepath = [Ddir(i).folder '\' Ddir(i).name];
        load(filepath);
        clear('resultslist'); % If it's really not needed any longer.
        X = px_size*x{1,1}; Y = px_size*y{2,1};
        u_profile = zeros(1,size(X,2));
        v_profile = zeros(1,size(X,2));
        u_std = zeros(1,size(X,2));
        v_std = zeros(1,size(X,2));
        
            color = colors(count,:);
%         if X(end)>=200
%             colorShift = 1;
%         else
%             colorShift = X(end)/200;
%         end
%         color = [1-colorShift, 0, colorShift];
        kk=1;
        for k=frames:size(x,1)
            % --------------------------PIV import ---------------------------------
            u_profile = u_profile + mean(u{kk});
            u_std = u_std + std(u{kk});
            v_profile = v_profile + mean(v{kk});
            v_std = u_std + std(v{kk});
            kk=kk+1;
        end
        u_profile = (u_profile-mean(u_profile))/kk;
        % - NORMALISATION
        %         u_profile = 2*u_profile/(max(u_profile) - min(u_profile));
        v_profile = (v_profile-mean(v_profile))/kk;
        
        jj=1;
        if mean(v_profile(1:2))> mean(v_profile(end-1:end))
            v_profile = flip(v_profile);
            jj=jj+1;
        end
        jj_count(count,1) = jj-1;
        
        % - NORMALISATION
        %         v_profile = 2*v_profile/(max(v_profile) - min(v_profile));
        u_std = u_std/kk  / sqrt(kk*size(X,1));
        v_std = v_std/kk  / sqrt(kk*size(X,1));
        
        XX = X(1,:);%-X(1,1);
        XX = (XX - XX(end)/2);%/XX(end); % - NORMALISATION
        
%         figure(1);
%         p1=plot(XX,u_profile,'color', [color, .3]);title('v_x');
%         p1.LineWidth = 2;   hold on
%         figure(2);
%         p2=plot(XX,v_profile, 'color', [color, .3]);title('v_y');
%         p2.LineWidth = 2;   hold on
        
        % % % % % %  insert velocity profiles to EXP   % % % % % %
        EXP{count,1} = 2*XX(end);
%         EXP{count,2} = 1/2*(abs(v_profile(end))+abs(v_profile(end)))/(2*XX(end));% norm by width
%         EXP{count,3} = 1/2*(abs(u_profile(end))+abs(u_profile(end)))/(2*XX(end));% norm by width
        EXP{count,2} = 1/2*(abs(v_profile(end))+abs(v_profile(end)));
        EXP{count,3} = 1/2*(abs(u_profile(end))+abs(u_profile(end)));
        EXP{count,4} = jj-1;
        EXP{count,5} = [XX', u_profile',u_std'];
        EXP{count,6} = [XX', v_profile',v_std'];
        u_profile = 2*u_profile/(max(u_profile) - min(u_profile));
        EXP{count,7} = [XX'/XX(end), u_profile',u_std'];
        v_profile = 2*v_profile/(max(v_profile) - min(v_profile));
        EXP{count,8} = [XX'/XX(end), v_profile',v_std'];
        count=count+1;
%     end
end
figure(1)
xlabel('Width (\mum)','FontSize',20); ylabel('v_x (\mum/hr)','FontSize',20);axis tight;hold off
figure(2)
xlabel('Width (\mum)','FontSize',20); ylabel('v_y (\mum/hr)','FontSize',20);axis tight;hold off

%%
fiG = 13;
WV = cell2mat(EXP(:,1:4));
[UWV1,~,idx]  = unique(WV(:,1));
N = histc(WV(:,1), UWV1); % repetition number
%------shear--------------------
UWV = [UWV1, accumarray(idx, WV(:,2),[],@mean), accumarray(idx,WV(:,2),[],@std)./sqrt(N)];
UWV = sortrows(UWV,1);
figure(fiG);errorbar(UWV(:,1),UWV(:,2),UWV(:,end)); hold on
%-----convergence---------------
UWU = [UWV1, accumarray(idx, WV(:,3),[],@mean), accumarray(idx,WV(:,3),[],@std)./sqrt(N)];
UWU = sortrows(UWU,1);
figure(fiG);errorbar(UWU(:,1),UWU(:,2),UWU(:,end)); hold off
xlabel('Width (x/L)','FontSize',20); ylabel('Shear flow magnitute, \Deltav_y (\mum/hr)','FontSize',20);axis tight;hold off
legend({'v_y', 'v_x'},'FontSize',30)
%----- count fliped cases  ------------
FLIPS = [UWV1,N,accumarray(idx, WV(:,4),[],@sum)];
figure(fiG+100)
plot(FLIPS(:,1),FLIPS(:,2),'-o'); hold on
plot(FLIPS(:,1),FLIPS(:,3),'-o'); hold off
legend({'total', 'flipped'},'FontSize',30)
%----- all shear magnitudes plot  ------------
figure(fiG+200)
% sc1 = scatter(WV(:,1),WV(:,2),50,'filled');
sc1 = plot(WV(:,1),WV(:,2),'o');hold on
sc1.MarkerSize = 10;sc1.MarkerEdgeColor=[1 1 1];sc1.MarkerFaceColor=[0.1,0.3,0.7];
sc2 = plot(WV(:,1),WV(:,3),'o');hold off
sc2.MarkerSize = 10;sc2.MarkerEdgeColor=[1 1 1];sc2.MarkerFaceColor=[0.9,0.4,0.1];
xlabel('Width (x/L)','FontSize',20); ylabel('Shear flow magnitute, \Deltav_y (\mum/hr)','FontSize',20);axis tight
legend({'|v_y|', '|v_x|'},'FontSize',30), axis([0 1000 0 7]);