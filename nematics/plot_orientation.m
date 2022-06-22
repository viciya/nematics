function plot_orientation(nx,ny)
figure()

[Xu,Yu] = meshgrid(1:size(nx,2),1:size(nx,1));
step = round(size(nx,2)/20); 
O_len = .7;


q = quiver( ...
    Xu(1:step:end,1:step:end),Yu(1:step:end,1:step:end),...
    nx(1:step:end,1:step:end), ny(1:step:end,1:step:end), ...
    O_len ...
    );
q.LineWidth=1; q.Color = [.7 .2 .0];
axis equal;axis tight;hold on
q.LineWidth=.5;
q.Color = [0 0 0];
q.ShowArrowHead='off';
p1 = plot(round(size(nx,2)/2),round(size(nx,1)/2),'o','MarkerFaceColor',[0 .5 .1]);
p1.MarkerSize = 10;
p1.MarkerEdgeColor= 'none';
axis off;