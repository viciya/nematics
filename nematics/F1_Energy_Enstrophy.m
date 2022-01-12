%% Check for specific width
px_size = 0.748;
px2mic = px_size*4;
widthChoice = 1514; dw = .05*widthChoice;
jjS = find(Sorted_Orient_width(:,2)<widthChoice+dw &...
    Sorted_Orient_width(:,2)>widthChoice-dw, 20,'first')

for n=1:length(jjS)
jj = jjS(n);

ii = Sorted_Orient_width(jj,1);Sorted_Orient_width(jj,2)
folderPIV = dirPIV(indX(ii,2)).folder;
namePIV = dirPIV(indX(ii,2)).name;
filepathPIV = [folderPIV '\' namePIV];
load(filepathPIV);
dx = px_size*(x{1}(1,2)-x{1}(1,1));

Ek = zeros(size(u,1),1); Ev = Ek;
for i=1:size(u,1)
    uu = px2mic*u{i}; vv = px2mic*v{i};
    Ek(i,1) = 0.5*mean2((uu.^2)+ (vv.^2));
    [u_x,u_y] = gradient(uu,dx);%/dx gradient need to be corrected for the dx
    [v_x,v_y] = gradient(vv,dx);
    Ev(i,1) = mean2(0.5*(v_x - u_y).^2);
end
Ev_all{n,1} = Ev;
Ek_all{n,1} = Ek;

time = (1:size(u))'/4;
figure(22);
yyaxis left; plot(time,Ek);
yyaxis right; plot(time,Ev);
figure(22); hold off

figure(23);
% c1=jet(size(Ek,1));
% scatter(Ev,Ek,5,c1,'filled');hold on
scatter(Ev,Ek,10,'filled');hold on
end
figure(23);ylabel('E_k (\mum/hr)^{2}'); xlabel('\Omega (1/hr)^{2}');set(gca,'Fontsize',18)
%%
Ev_allM = cell2mat(Ev_all);
Ek_allM = cell2mat(Ek_all);
[~,idx] = sort(Ek_allM);
Ev_Ek = [Ev_allM(idx), Ek_allM(idx)];
figure(4);
% plot(Ev_allM, Ek_allM,'o'); hold on
% plot(Ev_Ek(:,1), Ev_Ek(:,2),'.'); 


curvefit = fit(Ev_Ek(:,1), Ev_Ek(:,2),'poly1');
plot(Ev_Ek(:,1), Ev_Ek(:,2),'.');hold on
plot(curvefit,'predobs');
ylabel('E_k (\mum/hr)^{2}'); xlabel('\Omega (1/hr)^{2}');set(gca,'Fontsize',18)
axis([0 inf 0 inf]);
hold off