function fun_defectDraw(ps_x, ps_y, plocPsi_vec, ns_x, ns_y, nlocPsi_vec)
pl_len = .11;
nl_len = .07;
MarkerS = 10;
step = 7;
O_len = 0.5;
pColor = [.8 .1 0];
nColor = [.1 .1 .8];
LineWidth = 5;
if exist('ns_x') == 1
    % ----All defects ----------------------------------------
    % +1/2 defect
    p2 = plot(ps_x,ps_y,'o','MarkerFaceColor',pColor);hold on
    p2.MarkerSize = MarkerS;
    p2.MarkerEdgeColor= 'white';
    q2 = quiver(ps_x,ps_y,cosd(plocPsi_vec),sind(plocPsi_vec),pl_len);hold on
    q2.LineWidth = LineWidth;
    q2.Color = pColor;
    q2.ShowArrowHead = 'off';
    %     % -1/2 defect
    q3 = quiver(ns_x,ns_y,cosd(nlocPsi_vec),sind(nlocPsi_vec),nl_len);hold on
    q3.LineWidth=LineWidth;
    q3.Color = nColor;
    q3.ShowArrowHead = 'off';
    q4 = quiver(ns_x,ns_y,cosd(nlocPsi_vec+120),sind(nlocPsi_vec+120),nl_len);hold on
    q4.LineWidth = LineWidth;
    q4.Color = nColor;
    q4.ShowArrowHead = 'off';
    q5 = quiver(ns_x,ns_y,cosd(nlocPsi_vec+240),sind(nlocPsi_vec+240),nl_len);hold on
    q5.LineWidth = LineWidth;
    q5.Color = nColor;
    q5.ShowArrowHead = 'off';
    p3 = plot(ns_x,ns_y,'^','MarkerFaceColor',nColor);hold on
    p3.MarkerSize = MarkerS+2;
    p3.MarkerEdgeColor= 'white';
else
    % separete L R +1/2 defect orientations
    % -------------------------------------------
    rp = plocPsi_vec>-90 & plocPsi_vec<90;
    lp = plocPsi_vec<-90 | plocPsi_vec>90;
    
    p2 = plot(ps_x(rp),ps_y(rp),'o','MarkerFaceColor',pColor);hold on
    p2.MarkerSize = MarkerS;
    p2.MarkerEdgeColor= 'white';
    q2 = quiver(ps_x(rp),ps_y(rp),cosd(plocPsi_vec(rp)),sind(plocPsi_vec(rp)),pl_len);hold on
    q2.LineWidth = LineWidth;
    q2.Color = pColor;
    q2.ShowArrowHead = 'off';
    
    p2 = plot(ps_x(lp),ps_y(lp),'o','MarkerFaceColor',nColor);hold on
    p2.MarkerSize = MarkerS;
    p2.MarkerEdgeColor= 'white';
    q2 = quiver(ps_x(lp),ps_y(lp),cosd(plocPsi_vec(lp)),sind(plocPsi_vec(lp)),pl_len);hold on
    q2.LineWidth = LineWidth;
    q2.Color = nColor;
    q2.ShowArrowHead = 'off';
    % ------------------------------------------------
end