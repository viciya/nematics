function [shear, path_piv, path_op] = get_highest_piv_shear(dirPIV, dirOP, indX, Sorted_Orient_width, width)
px_sizeOP = 3* .74;
Sw = width; % selectd width
dw = .05*Sw; % define delta
Range = Sorted_Orient_width(Sorted_Orient_width(:,2)>Sw-dw & Sorted_Orient_width(:,2)<Sw+dw,1);

shear = 0;
for i = 1:length(Range)
    filepathOP = [dirOP(indX(Range(i),1)).folder '\' dirOP(indX(Range(i),1)).name];
    filepathPIV = [dirPIV(indX(Range(i),2)).folder '\' dirPIV(indX(Range(i),2)).name];
    piv = load(filepathPIV);
    v = mean(cell2mat(piv.v));
%     plot(v); hold on
    if abs(v(end)-v(1))> shear
        shear = abs(v(end)-v(1));
        path_piv = filepathPIV;
        path_op = filepathOP;
    end
    % shear = mean(cell2mat(piv.v)) - mean(piv.v);
    
end