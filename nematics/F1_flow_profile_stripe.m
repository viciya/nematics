%% LOAD FILES
PC_path = 'C:\Users\victo\Google Drive\';   % HP notebook
% PC_path = 'C:\Users\vici\Google Drive\';    % Curie PC
% PC_path = 'D:\GD\';                         % RSIP notebook

addpath([PC_path,'Curie\DESKTOP\HT1080\codes']);
pathOP = ([PC_path,'DATA\HT1080\Orient']);
pathPIV = ([PC_path,'DATA\HT1080\PIV_DATs']);

[dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV);
%% SELECT WIDTH AND PARAMETERS
clearvars -except dirOP  dirPIV  Sorted_Orient_width  indX PC_path pathOP pathPIV NET

i = 1;
Sw = 400; % selectd width
dw = .05*Sw; % define delta
pix2mic = .74;
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);

stripe = struct([]);

%% OPTION 1 (One stripe)
% true for shear/ false for confergent flow
shear_flow = true; 
% figure(Sw);
k_v_profile = [];
for i = 1%:numel(Range)
    disp(['File ',num2str(i), ' from ',num2str(numel(Range))]);
    filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];
    
    load(filepathPIV);
    %     NOTE velocity has units of px/hr
    t1=76;t2=t1+25;
    k_v_profile = zeros(size(v,1),size(v{1},2));
    time_segment=t1:min(t1,size(v,1));
%     for k = time_segment
    for k = 1:size(v,1)
        if shear_flow 
            k_v_profile(k,:) = pix2mic*(mean(v{k})- mean2(v{k}));
        else
            k_v_profile(k,:) = pix2mic*(mean(u{k})- mean2(u{k}));
        end
    end
    x_ax = pix2mic*(x{1}(1,:)-(x{1}(1,1)+x{1}(1,end))/2); 
    i_v_profile = mean(k_v_profile,1);
    i_v_std = std(k_v_profile,1)/sqrt(size(k_v_profile,1));
%     stripe(i).profile = i_v_profile;

% ---------------PLOT BLOCK----------------------
    plot(x_ax, mean(k_v_profile(1:40,:),1),'Color',[.8,.8,.8],'LineWidth',2);hold on
    plot(x_ax, mean(k_v_profile(30:70,:),1),'Color',[.8,.8,.8],'LineWidth',2);hold on
    plot(x_ax, mean(k_v_profile(60:size(v,1),:),1),'Color',[.8,.8,.8],'LineWidth',2);hold on
%     errorbar(x_ax,i_v_profile,i_v_std);
    [l,p] = boundedline(x_ax,i_v_profile,i_v_std,'alpha');
%     [l,p] = boundedline(x, y1, e1, '-b*', x, y2, e2, '--ro');
    l.LineWidth=2; l.Color=[.8,.1,.1,.6]; 
    p.FaceAlpha=.1; p.FaceColor=l.Color;
    axis tight square; hold off
    set(gcf,'Color',[1 1 1]);
    set(gca,'FontSize',18);
    if shear_flow 
        ylabel('$ v_{y} \ (\mu m/h)$','Interpreter','latex','FontSize',28);
    else
        ylabel('$ v_{x} \ (\mu m/h)$','Interpreter','latex','FontSize',28);
    end
    xlabel('$ Width (\mu m) $','Interpreter','latex','FontSize',24);
end
%
%% OPTION 2 (All stripes at given range)
% true for shear/ false for confergent flow
shear_flow = true; 
% figure(Sw);
k_v_profile = [];
for i = 1:numel(Range)
    disp(['File ',num2str(i), ' from ',num2str(numel(Range))]);
    filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];
    
    load(filepathPIV);
    %     NOTE velocity has units of px/hr
    t1=76;t2=t1+25;
    k_v_profile = zeros(size(v,1),size(v{1},2));
    time_segment=t1:min(t1,size(v,1));
%     for k = time_segment
    for k = 1:size(v,1)
        if shear_flow 
            k_v_profile(k,:) = pix2mic*(mean(v{k})- mean2(v{k}));
        else
            k_v_profile(k,:) = pix2mic*(mean(u{k})- mean2(u{k}));
        end
    end
    x_ax = pix2mic*(x{1}(1,:)-(x{1}(1,1)+x{1}(1,end))/2); 
    i_v_profile = mean(k_v_profile,1);
    i_v_std = std(k_v_profile,1)/sqrt(size(k_v_profile,1));
    stripe(i).profile = i_v_profile;
end
% now all stripe profiles stored in stripe.profile
% how to make average over different number of points at x
% split profile to left and right edge
% !!! solved by straching of the scale in last cell of !!!
% C:\Users\victo\Google Drive\Curie\DESKTOP\HT1080\codes\profile_Vy_Ang.m
%%
i=1;% comment after first run
p=i+1;
i=p;
profile=stripe(i).profile;
while size(stripe(i+1).profile,2)==size(stripe(i).profile,2)
    profile=profile+stripe(i+1).profile;    
    i=i+1
end
profile=profile/(i-p);
plot(profile);hold on