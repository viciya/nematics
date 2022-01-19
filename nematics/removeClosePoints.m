function [x, y] = removeClosePoints(X, min_dist)

if (nargin < 2)
    min_dist = 10;
end


D1 = pdist2(X,X);
% define minimal distance 
% and get indecies of elements
[yrem, xrem] = find(triu(D1)<min_dist & triu(D1)~=0);
% remove too close elements
X(xrem,:) = [];

x = X(:,1);
y = X(:,2);