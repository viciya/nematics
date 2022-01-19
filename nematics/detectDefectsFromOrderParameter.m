function [x, y] = detectDefectsFromOrderParameter(op)

op_tresh = .5
f_size = 7
filt = (fspecial('gaussian', f_size, f_size));

p = FastPeakFind(1-op, 800, filt);
x_all = p(1:2:end);
y_all = p(2:2:end);

sz = size(op);
ind = sub2ind([sz], y_all, x_all);

x = x_all(op(ind) < op_tresh);
y = y_all(op(ind) < op_tresh);