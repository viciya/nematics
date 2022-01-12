function plot_pdefect_ndefect(ps_x, ps_y, plocPsi_vec,...
    ns_x, ns_y, nlocPsi_vec,...
    plength, nlength,sz,line_width)
if ~exist('plength')
    plength = .11;
    nlength = .07;
    sz = 10;
    line_width = 5;
end


pColor = [.8 .1 0];
nColor = [.1 .1 .8];

if exist('ns_x') == 1
    % ----All defects ----------------------------------------
    % +1/2 defect
    p2 = plot(ps_x,ps_y,'o','MarkerFaceColor',pColor);hold on
    p2.MarkerSize = sz;
    p2.MarkerEdgeColor = pColor;%'white';
    q2 = quiver(ps_x,ps_y,cosd(plocPsi_vec),sind(plocPsi_vec),plength);hold on
    q2.LineWidth = line_width;
    q2.Color = pColor;
    q2.ShowArrowHead = 'off';
    %     % -1/2 defect
    q3 = quiver(ns_x,ns_y,cosd(nlocPsi_vec),sind(nlocPsi_vec),nlength);hold on
    q3.LineWidth=line_width;
    q3.Color = nColor;
    q3.ShowArrowHead = 'off';
    q4 = quiver(ns_x,ns_y,cosd(nlocPsi_vec+120),sind(nlocPsi_vec+120),nlength);hold on
    q4.LineWidth = line_width;
    q4.Color = nColor;
    q4.ShowArrowHead = 'off';
    q5 = quiver(ns_x,ns_y,cosd(nlocPsi_vec+240),sind(nlocPsi_vec+240),nlength);hold on
    q5.LineWidth = line_width;
    q5.Color = nColor;
    q5.ShowArrowHead = 'off';
    p3 = plot(ns_x,ns_y,'o','MarkerFaceColor',nColor);hold on
    p3.MarkerSize = ceil(sz/2);
    p3.MarkerEdgeColor= nColor;%'white';
else
    % separete L R +1/2 defect orientations
    % -------------------------------------------
    rp = plocPsi_vec>-90 & plocPsi_vec<90;
    lp = plocPsi_vec<-90 | plocPsi_vec>90;
    
    p2 = plot(ps_x(rp),ps_y(rp),'o','MarkerFaceColor',pColor);hold on
    p2.MarkerSize = sz;
    p2.MarkerEdgeColor= 'white';
    q2 = quiver(ps_x(rp),ps_y(rp),cosd(plocPsi_vec(rp)),sind(plocPsi_vec(rp)),plength);hold on
    q2.LineWidth = line_width;
    q2.Color = pColor;
    q2.ShowArrowHead = 'off';
    
    p2 = plot(ps_x(lp),ps_y(lp),'o','MarkerFaceColor',nColor);hold on
    p2.MarkerSize = sz;
    p2.MarkerEdgeColor= 'white';
    q2 = quiver(ps_x(lp),ps_y(lp),cosd(plocPsi_vec(lp)),sind(plocPsi_vec(lp)),plength);hold on
    q2.LineWidth = line_width;
    q2.Color = nColor;
    q2.ShowArrowHead = 'off';
    % ------------------------------------------------
end