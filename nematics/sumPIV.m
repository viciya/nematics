function [uu, vv] = sumPIV(filepathPIV,filt)
load(filepathPIV);

% [L,W] = size(u{1,1});
T = size(u,1);
uu = zeros(size(u{1,1}));
vv = uu;


for k =1:T
    uu = uu + imfilter(u{k}, filt);% - mean2(u{k});
    vv = vv + imfilter(v{k}, filt) - mean2(v{k});
end

uu = uu/T;
vv = vv/T;

end