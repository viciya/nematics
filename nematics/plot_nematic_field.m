function plot_nematic_field(Ang)

if any( Ang(:)>4 ) % check if Ang is in RAD
    Ang = Ang * pi/180;
end

[Xu,Yu] = meshgrid(1:size(Ang,2),1:size(Ang,1));
step = 13; O_len = .7;
q6 = quiver( ...
    Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
    cos(Ang(1:step:end,1:step:end)),-sin(Ang(1:step:end,1:step:end)), ...
    O_len ...
    );
q6.LineWidth=1; q6.Color = [.7 .2 .0]; q6.ShowArrowHead='off';