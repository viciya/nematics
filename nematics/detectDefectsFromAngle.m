function [xf, yf] = detectDefectsFromAngle(Ang)

if any( Ang(:)>4 ) % check if Ang is in RAD
    Ang = Ang * pi/180;
end

qq = order_parameter(Ang,10,3);

[x, y] = detectDefectsFromOrderParameter(qq);

X = [x,y];
[xf, yf] = removeClosePoints(X, 15);