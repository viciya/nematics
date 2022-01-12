%%
Ddir_Or = dir('C:\Users\vici\Desktop\HT1080\stripe_Orient');
folder_main_Or = Ddir_Or(1).folder;
filesInFolder_Or = size(Ddir_Or,1);

Ddir_PIV = dir('C:\Users\vici\Desktop\HT1080\stripe_PIV');
folder_main_PIV = Ddir_PIV(1).folder;
filesInFolder_PIV = size(Ddir_PIV,1);
t_shift = 0;
%%

ki = 18;
for iii=ki:size(Ddir_Or,1)
% for iii=14:size(Ddir_Or,1)
    %       get ORIENT file name

    if contains(Ddir_Or(iii).name, '.tif' )
        Ddir_Or(iii).name;
        disp(['file: ' num2str(iii) ' from: ' num2str(size(Ddir_Or,1))]);
        filepath_Or = [Ddir_Or(iii).folder '\' Ddir_Or(iii).name];
        k1 = strfind(filepath_Or,'s');
        k2 = strfind(filepath_Or,'.');
        pattern = filepath_Or(k1(end):k2(end)-1);
        %       find corresponong PIV
        for j=1:size(Ddir_PIV,1)
            temp = [Ddir_PIV(j).folder '\' Ddir_PIV(j).name];
            k1 = strfind(temp, pattern);
            if ~isempty(k1)==1
                filepath_PIV = temp;
            end
        end
        %        load ORIENT and PIV
        info = imfinfo(filepath_Or); % Place path to file inside single quotes
        Nn = numel(info)
        load(filepath_PIV,'resultslist');
        A = resultslist;  % Assign it to a new variable with different name.
        clear('resultslist'); % If it's really not needed any longer.
        
        % % % % % % % % % % % % %  MAIN CODE % % % % % % % % % % % % %

        
        ff = 5;
        filt = fspecial('gaussian',ff,ff);
        dt = 1; % velocity avearage        
        pix2mic = .74;
        k_count =1;
        total_count = 0;
        
        Ang = imread(filepath_Or,1); % k
        box = min([.5*size(Ang,2), 80]);
        s_box = floor(sqrt(box^2/2));
            
%         disp(['box width in um: ',num2str(s_box* pix2mic *2,3)]);
        sub_PIV_u = zeros(2*s_box+1);
        sub_PIV_v =sub_PIV_u;
        sub_nx1 = sub_PIV_u;
        sub_ny1 = sub_nx1;
        sub_nx2 = sub_nx1;
        sub_ny2 = sub_nx1;
        
        for k=1:Nn-dt-1
%         start= 80; kStep = 1; last= min(start+40,Nn);
%         for k=start:kStep:last
            k
            clearvars -except EXP pattern iii k kStep k_count info Nn filename pathname...
                filterindex pix2mic A projection t_shift deltaTeta rot_u rot_v...
                box s_box dt start last total_count filt Xu Yu...
                sub_PIV_u sub_PIV_v sub_nx1 sub_ny1 sub_nx2 sub_ny2...
                Ddir_Or Ddir_PIV filepath_Or filepath_PIV
            
            qstep = 20; %mine 10 in pixels
            Ang = imread(filepath_Or,k); % k
            Ang = pi/180*Ang;
            [l,w] = size(Ang);
            

            box = min([.5*w, 80]);
            s_box = floor(sqrt(box^2/2));
            
            q = ordermatrixglissant_overlap(Ang,qstep,6);
            im2 = q < min(q(:))+0.6;%.2; % make binary image to use regionprops
            s = regionprops('table', im2,'centroid');
            % --------------------------------------------------------------------
            % %% !------- Find +/-  defects --------- in ONE FRAME  -------!
            % --------------------------------------------------------------------
            r_Circ = 17;
            ps_x = s.Centroid(:,1);
            ps_y = s.Centroid(:,2);
            s_Circ = zeros(size(Ang));
            TEMP = zeros(2*r_Circ+1,2*r_Circ+1,10);
            Blank = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]);
            
            [XX1,YY1] = meshgrid(-r_Circ:r_Circ,-r_Circ:r_Circ);
            AngTEMP_vec = zeros(10,1);
            
            for j=1:10
                TEMP(:,:,j) = sqrt(XX1.^2+YY1.^2) < r_Circ ...
                    & atan2(YY1, XX1)>= (j-1)*pi/5-pi ...
                    & atan2(YY1, XX1)< j*pi/5-pi;
            end
            
            Ang(Ang <= 0) = pi + Ang(Ang <= 0);
            pcount = 1;
            ncount = 1;
            
            for i=1:length(ps_x)-1
                for j=1:10
                    Blank = zeros(size(Blank));
                    Blank(round(ps_y(i)):round(ps_y(i))+2*r_Circ,...
                        round(ps_x(i)):round(ps_x(i))+2*r_Circ) = TEMP(:,:,j);
                    sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
                    
                    AngTEMP  = Ang.* sBlank;
                    AngTEMP_vec(j,1)  = mean(mean(180/pi*AngTEMP(AngTEMP~=0)));
                end
                
                % +/- 1/2 defect characterization
                pos_neg = (AngTEMP_vec(2:end)- AngTEMP_vec(1:end-1))>0;
                if sum(pos_neg)<= 4
                    pDefect_Centroid(pcount,:)=s.Centroid(i,:);
                    pcount = pcount+1;
                elseif sum(pos_neg)>= 5
                    nDefect_Centroid(ncount,:)=s.Centroid(i,:);
                    ncount = ncount+1;
                end
            end
            if exist('pDefect_Centroid')==0
                disp('No defects!')
                continue
            end
            % +1/2 defect angle------------------------------------------------------
            px = cos(Ang);
            py = -sin(Ang);
            Qxx = (px.*px - 1/2);
            Qxy = (px.*py);
            Qyx = (py.*px);
            Qyy = (py.*py - 1/2);
            
            [dxQxx,~] = gradient(Qxx);
            [dxQxy,dyQxy] = gradient(Qxy);
            [~,dyQyy] = gradient(Qyy);
            pPsi = atan2((dxQxy+dyQyy),(dxQxx+dyQxy));
            
            ps_x = pDefect_Centroid(:,1);
            ps_y = pDefect_Centroid(:,2);
            s_Circ = zeros(size(Ang));
            TEMP = zeros(2*r_Circ+1,2*r_Circ+1);
            Blank = zeros(size(Ang)+[2*r_Circ, 2*r_Circ]);
            
            [XX1,YY1] = meshgrid(-r_Circ:r_Circ,-r_Circ:r_Circ);
            TEMP(:,:) = sqrt(XX1.^2+YY1.^2) < r_Circ;
            
            for i=1:length(ps_x)
                Blank = zeros(size(Blank));
                Blank(round(ps_y(i)):round(ps_y(i))+2*r_Circ,...
                    round(ps_x(i)):round(ps_x(i))+2*r_Circ) = TEMP;
                sBlank = Blank(r_Circ+1:end-r_Circ,r_Circ+1:end-r_Circ);
                plocPsi  = pPsi.* sBlank;
                plocPsi_vec(i,1)  = mean(mean(180/pi*plocPsi(plocPsi~=0)));
            end

            
            % --------------------------PIV import ---------------------------------
            
            u_filtered = A(7,k+t_shift:k+t_shift)';
            v_filtered = A(8,k+t_shift:k+t_shift+dt-1)';
            
            uu = zeros(size(u_filtered{1,1}));
            vv = uu;
            % ----- averaging of velocity fields (dt=1: cancels averaging) -----
            for t=1:dt
                uu = uu + imfilter(u_filtered{t,1}, filt);
                vv = vv + imfilter(v_filtered{t,1}, filt);
            end
            
            uu = uu/dt;
            vv = vv/dt;
            sc = size(Ang,1)/size(uu,1);
            
            [Xorig,Yorig] = meshgrid((1/sc:size(uu,2))*sc,(1/sc:size(uu,1))*sc);
            [Xu,Yu] = meshgrid(1:w,1:l);
            u_interp = interp2(Xorig,Yorig,uu,Xu,Yu,'cubic',0);
            v_interp = interp2(Xorig,Yorig,vv,Xu,Yu,'cubic',0);
            
            % ///////////////   ROTATION //////////////////////////////////////
            rot_u = zeros(2*s_box+1);
            rot_v = rot_u;
            rot_nx1 = rot_u;
            rot_ny1 = rot_u;
            rot_nx2 = rot_u;
            rot_ny2 = rot_u;
            
            i_count = 0;
            ii=13;
            for i=1:length(ps_x)
                % ---  check if box around the defects fits in the PIV field  -----
                if (ps_x(i)-box)>1 && (ps_y(i)-box)>1 && (ps_x(i)+box)<w && (ps_y(i)+box)<l
                    %             take PIV sub-window around the defect
                    su = u_interp(round(ps_y(i))-box:round(ps_y(i))+box,...
                        round(ps_x(i))-box:round(ps_x(i))+box);
                    sv = v_interp(round(ps_y(i))-box:round(ps_y(i))+box,...
                        round(ps_x(i))-box:round(ps_x(i)+box));
                    %             take ORIENT sub-window around the defect
                    sAng = Ang(round(ps_y(i))-box:round(ps_y(i))+box,...
                        round(ps_x(i))-box:round(ps_x(i))+box);
                    snx = cos(sAng);
                    sny = -sin(sAng);
                    
                    %             rotate each PIV vector by angle of the defect
                    suR = cosd(plocPsi_vec(i,1))*su + sind(plocPsi_vec(i,1))*sv;
                    svR = -sind(plocPsi_vec(i,1))*su + cosd(plocPsi_vec(i,1))*sv;
                    %             rotate each ORIENT vector by angle of the defect
                    snxR = cosd(plocPsi_vec(i,1))*snx + sind(plocPsi_vec(i,1))*sny;
                    snyR = -sind(plocPsi_vec(i,1))*snx + cosd(plocPsi_vec(i,1))*sny;
                    
                    %             rotate whole PIV field by angle of the defect
                    suRR = imrotate(suR,plocPsi_vec(i,1),'bilinear','crop');
                    svRR = imrotate(svR,plocPsi_vec(i,1),'bilinear','crop');%,'crop'
                    %             rotate whole ORIENT field by angle of the defect
                    snxRR = imrotate(snxR,plocPsi_vec(i,1),'bilinear','crop');
                    snyRR = imrotate(snyR,plocPsi_vec(i,1),'bilinear','crop');%,'crop'
                    
                    %           Average PIV fields
                    rot_u  = rot_u + suRR(box-s_box:box+s_box,box-s_box:box+s_box);
                    rot_v  = rot_v + svRR(box-s_box:box+s_box,box-s_box:box+s_box);
                    %           Average ORIENT fields (averaging of nematic field involves flipping, see plot of sub_nx/y1 VS sub_nx/y2)
                    nx_temp = snxRR(box-s_box:box+s_box,box-s_box:box+s_box);
                    ny_temp = snyRR(box-s_box:box+s_box,box-s_box:box+s_box);
                    nx_temp1=nx_temp;nx_temp2=nx_temp;
                    ny_temp1=ny_temp;ny_temp2=ny_temp;
                    nx_temp1(nx_temp<0) = -nx_temp(nx_temp<0);
                    ny_temp1(nx_temp<0) = -ny_temp(nx_temp<0);
                    nx_temp2(ny_temp<0) = -nx_temp(ny_temp<0);
                    ny_temp2(ny_temp<0) = -ny_temp(ny_temp<0);
                    
                    rot_nx1  = rot_nx1 + nx_temp1;
                    rot_ny1  = rot_ny1 + ny_temp1;
                    rot_nx2  = rot_nx2 + nx_temp2;
                    rot_ny2  = rot_ny2 + ny_temp2;
                    
                    i_count = i_count+1;
                end
            end
            disp(['analysed defects: ' num2str(i_count) ' from: ' num2str(length(ps_x))]);
            if i_count~=0
                sub_PIV_u = sub_PIV_u + rot_u/mean(sqrt(uu(:).^2+vv(:).^2))/i_count;
                sub_PIV_v = sub_PIV_v + rot_v/mean(sqrt(uu(:).^2+vv(:).^2))/i_count;
                disp(['NaN check: ' num2str(sum(sum(isnan(sub_PIV_u))))]);
            end

            % sub_PIV_u =  rot_u/mean(sqrt(uu(:).^2+vv(:).^2));
            % sub_PIV_v =  rot_v/mean(sqrt(uu(:).^2+vv(:).^2));
            sub_nx1 = sub_nx1+ rot_nx1;
            sub_ny1 = sub_ny1+ rot_ny1;
            sub_nx2 = sub_nx2+ rot_nx2;
            sub_ny2 = sub_ny2+ rot_ny2;
            
            k_count=k_count+1;
            total_count = total_count + i_count;
        end
        
        % % % % % % % % % % % % %  MAIN CODE END % % % % % % % % % % % % %
        
%         save to cell
EXP{iii,1} = pattern;
EXP{iii,2} = [pix2mic*size(Ang,2), s_box* pix2mic *2, total_count];
EXP{iii,3} = sub_PIV_u;
EXP{iii,4} = sub_PIV_v;
EXP{iii,5} = Xu;
EXP{iii,6} = Yu;

    end
end

%%
% select for width
clear var ind
width = 130; delta = 20;
kk=1;
for k=3:size(EXP,1)
    if EXP{k,2}(1)>width-delta & EXP{k,2}(1)<width+delta 
        ind(kk,1)= k;
        kk=kk+1;
    end
end

close all
ind_len = length(ind);
kk=1;
for i=1:ind_len
    k=ind(i);
sub_PIV_u = EXP{k,3};
sub_PIV_v = EXP{k,4};
Xu = EXP{k,5};
Yu = EXP{k,6};
box = min([EXP{k,2}(2), 80]);s_box = floor(sqrt(box^2/2));
figure(121)
subplot(ceil(sqrt(ind_len)),ceil(sqrt(ind_len)),kk)
bin = 2*ceil(5*box/100);
% ff=2*s_box+1;
ff=size(sub_PIV_u,1);
quiver(Xu(1:bin:ff,1:bin:ff),Yu(1:bin:ff,1:bin:ff),...
    sub_PIV_u(1:bin:end,1:bin:end),sub_PIV_v(1:bin:end,1:bin:end),2);
axis equal;axis tight; hold on

p1 = plot(ff/2,ff/2,'o','MarkerFaceColor',[0 .5 .1]);
p1.MarkerSize = 10;
p1.MarkerEdgeColor = 'none';
 title(k); hold off; axis off;
% 
figure(123)
subplot(ceil(sqrt(ind_len)),ceil(sqrt(ind_len)),kk)
% subplot(ceil(((last-start)/kStep).^.5),floor(((last-start)/kStep).^.5),k_count-1)
[u_x,u_y] = gradient(sub_PIV_u);%/dx gradient need to be corrected for the dx
[v_x,v_y] = gradient(sub_PIV_v);%/dx
vorticity = (v_x - u_y);%------------------- OPTION1
divV = (u_x + v_y);%----- OPTION2
filtN = 30;
filt = fspecial('gaussian',filtN,filtN);
u1 = imfilter(vorticity, filt);
surf(Xu(1:1:ff,1:1:ff),Yu(1:1:ff,1:1:ff),u1-10);view(2);shading interp;colormap jet;axis equal;axis tight;hold on
% caxis([-max(u1(:))/2-10, max(u1(:))/2-10]);

q=quiver(Xu(1:bin:ff,1:bin:ff),Yu(1:bin:ff,1:bin:ff),...
    sub_PIV_u(1:bin:end,1:bin:end),sub_PIV_v(1:bin:end,1:bin:end),1);axis equal;axis tight; hold on
q.LineWidth=2;
q.Color = [1 1 1];

p2 = plot3(ff/2,ff/2,40,'o','MarkerFaceColor',[0 0 0]);
p2.MarkerSize = 10;
p2.MarkerEdgeColor= 'none';
% axis off; title([num2str(k),'(',num2str(i_count),')']); 
hold off   

axis off;
title([num2str(k),'(',num2str(EXP{k,2}(3)),') w_{stripe}= ',...
    num2str(EXP{k,2}(1)),'  w_{box}= ',num2str(EXP{k,2}(2))]); hold off
% 
% % uisave({'sub_PIV_u','sub_PIV_v'},['box_size_um_',num2str(s_box*1*2,3)]);
% 
kk=kk+1;
end