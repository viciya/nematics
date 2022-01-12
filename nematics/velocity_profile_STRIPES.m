%%
% Ddir = dir('F:\CURIE\guillaume_RPE1\131130_exp2_bandes_analyse\analyse stripes RPE1_exp2\300um\PIV_DATs');
% Ddir = dir(['C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\stripe_PIV_PIVlab', '*.mat']);
Ddir = dir(['C:\Users\vici\Google Drive\DATA\HT1080\PIV_DATs'  '\*.mat']);
folder_main = Ddir(1).folder;
filesInFolder = size(Ddir,1);
px_size = .74;
frame_per_hr = 4;
px2mic = px_size * frame_per_hr;
frames = 50;
dt=1;
%%
n=25;
for i=1:size(Ddir,1)
    if contains(Ddir(i).name, '.mat' )
        Ddir(i).name
        disp(['file: ' num2str(i) ' from: ' num2str(size(Ddir,1))]);
        filepath = [Ddir(i).folder '\' Ddir(i).name];
        load(filepath,'resultslist');
        A = resultslist;  % Assign it to a new variable with different name.
        clear('resultslist'); % If it's really not needed any longer.
        X = A{1,1}; Y = A{2,1};
        u_profile = zeros(1,size(X,2));
        v_profile = zeros(1,size(X,2));
        u_std = zeros(1,size(X,2));
        v_std = zeros(1,size(X,2));
        kk=1;
        for k=frames:size(A,2)
            % --------------------------PIV import ---------------------------------
            u_profile = u_profile + mean(A{7,kk},1);
            u_std = u_std + std(A{7,kk},1);
            v_profile = v_profile + mean(A{8,kk},1);
            v_std = u_std + std(A{8,kk},1);
            kk=kk+1;
        end
        u_profile = (u_profile-mean(u_profile))/kk * px2mic;
        % - NORMALISATION
%         u_profile = 2*u_profile/(max(u_profile) - min(u_profile)); 
        v_profile = (v_profile-mean(v_profile))/kk * px2mic;
        % - NORMALISATION
%         v_profile = 2*v_profile/(max(v_profile) - min(v_profile)); 
        u_std = u_std/kk * px2mic / sqrt(kk*size(X,1));
        v_std = v_std/kk * px2mic / sqrt(kk*size(X,1));
        
        qX = X(1,1):1:X(1,end);
        q_u = pchip(X(1,:),u_profile,qX);
        q_v = pchip(X(1,:),v_profile,qX);
        q_uStd = pchip(X(1,:),u_std,qX);
        q_vStd = pchip(X(1,:),v_std,qX);
        
        XX = X(1,:)-X(1,1);
        XX = (XX - XX(end)/2);%/XX(end); % - NORMALISATION   
        
        figure(1);
        p1=plot(XX,u_profile,'Color',[.8 0 0 .3]);title('v_x');
        p1.LineWidth = 2;   hold on
        figure(2);
        p2=plot(XX,v_profile,'Color',[.8 0 0 .3]);title('v_y');
        p2.LineWidth = 2;   hold on
%       
        qX = qX - qX(1);
        qX = (qX - qX(end)/2)/qX(end);
        figure(3);
        p1=plot(qX,q_u,'Color',[0 0 .8 .3]);title('v_x');
        p1.LineWidth = 2;   hold on
        figure(4);
        p2=plot(qX,q_v,'Color',[0 0 .8 .3]);title('v_y');
        p2.LineWidth = 2;   hold on  

% % % % % %  insert velocity profiles to EXP   % % % % % % 
        EXP{i-2,1} = 2*XX(end);
        EXP{i-2,2} = [XX', u_profile',u_std'];
        EXP{i-2,3} = [XX', v_profile',v_std'];
        u_profile = 2*u_profile/(max(u_profile) - min(u_profile)); 
        EXP{i-2,4} = [XX'/XX(end), u_profile',u_std'];
        v_profile = 2*v_profile/(max(v_profile) - min(v_profile));
        EXP{i-2,5} = [XX'/XX(end), v_profile',v_std'];       
    end
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
% load('C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\velocity_profile_ALL.mat')
% SORT BY WIDTH
[~, ind] = sort(cell2mat(EXP(:,1)));
ind = flip(ind);
for k=1:size(EXP,1)
    i=ind(k);
    color = [1-EXP{i,6}(end,1)/150 0 EXP{i,6}(end,1)/150 .4];
    if EXP{i,6}(end,1)>150
     color = [0 0 1 .4];   
    end
    figure(1)
pl = plot(EXP{i,5}(:,1),EXP{i,5}(:,2));hold on
pl.Color = color; 
pl.LineWidth = 2; 
    figure(2)
pl = plot(EXP{i,6}(:,1),EXP{i,6}(:,2));hold on
pl.Color = color; 
pl.LineWidth = 2;
%     figure(3)
% pl = plot(EXP{i,4}(:,1),EXP{i,4}(:,2));hold on
% pl.Color = [1-EXP{i,2}(end,1)/120 0 EXP{i,2}(end,1)/120 .4]; 
% pl.LineWidth = 4; 
%     figure(4)
% pl = plot(EXP{i,5}(:,1),EXP{i,5}(:,2));hold on
% pl.Color = [1-EXP{i,2}(end,1)/120 0 EXP{i,2}(end,1)/120 .4]; 
% pl.LineWidth = 4;
end
figure(1);
xlabel('Width (\mum)','FontSize',20); ylabel('v_x (\mum/hr)','FontSize',20);axis tight;
hold off
figure(2);
xlabel('Width (\mum)','FontSize',20); ylabel('v_y (\mum/hr)','FontSize',20);axis tight;
hold off
% figure(3)
% xlabel('Width (x/L)','FontSize',20); ylabel('v_x/v_{x,max}','FontSize',20);axis tight;
% hold off
% figure(4)
% xlabel('Width (x/L)','FontSize',20); ylabel('v_y/v_{y,max}','FontSize',20);axis tight;
% hold off


%%
errorbar(X(1,:),u_profile,u_std);hold on
errorbar(X(1,:),v_profile,v_std);hold off
%%
i=54;
len = 50;
for i=3:size(vort_area,2)
    meanVortDensity(i) = mean(vortDensity{i}(end-len:end));
    stdVortDensity(i) = std(vortDensity{i}(end-len:end))/len^.5;
    
    meanVortNum(i) = mean(vortNum{i}(end-len:end));
    stdVortNum(i) = std(vortNum{i}(end-50:end))/len^.5;
    
    meanVortArea(i) = mean(vortArea{i}(end-len:end));
    stdVortArea(i) = std(vortArea{i}(end-len:end))/len^.5;
end

[sortVort_width, ind] = sort(vort_width);
sortVort_area = vort_area(ind);
meanVortDensity = meanVortDensity(ind);
stdVortDensity = stdVortDensity(ind);
meanVortNum = meanVortNum(ind);
stdVortNum = stdVortNum(ind);
figure(1)
errorbar(sortVort_width,meanVortDensity,stdVortDensity);hold on
figure(2)
errorbar(sortVort_width,meanVortNum,stdVortNum);hold on
figure(3)
errorbar(sortVort_width,meanVortArea,stdVortArea);hold on
