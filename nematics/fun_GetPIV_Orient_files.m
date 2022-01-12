function [dirOP, dirPIV, Sorted_Orient_width, indX] = fun_GetPIV_Orient_files(pathOP, pathPIV)
dirOP = dir([pathOP  '\*.tif']);
dirPIV = dir([pathPIV  '\*.mat']);

for i = 1:size(dirOP,1)
    
    nameOP = dirOP(i).name;
    p1=strfind(nameOP,'s');
    p2=strfind(nameOP,'.tif');
    patternOP = nameOP(p1:p2);
    
    pC=strfind(nameOP,'old');
    pD=strfind(nameOP,'09_07_2018_Orient');
    pE=strfind(nameOP,'26072018_Orient');
    pF=strfind(nameOP,'01082018_Orient');
    
    if ~isempty(pF)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name,'01082018') &&...
                    contains(dirPIV(j).name, patternOP)
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;
            end
        end
    elseif ~isempty(pE)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name,'26072018') &&...
                    contains(dirPIV(j).name, patternOP)
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;
            end
        end
    elseif ~isempty(pD)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name,'09072018') &&...
                    contains(dirPIV(j).name, patternOP)
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;
            end
        end
    elseif ~isempty(pC)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name,'old') &&...
                    contains(dirPIV(j).name, patternOP)
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;
            end
        end
    elseif isempty(pC)&isempty(pD)&isempty(pE)&isempty(pF)
        for j = 1:size(dirPIV,1)
            if contains(dirPIV(j).name, patternOP)&&...
                    ~contains(dirPIV(j).name,'01082018') &&...
                    ~contains(dirPIV(j).name,'26072018') &&...
                    ~contains(dirPIV(j).name,'09072018') &&...
                    ~contains(dirPIV(j).name,'old')
                namePIV = dirPIV(j).name;
                indX(i,:) = [i,j];% index [OP,PIV]
                names{i,1} = nameOP;
                names{i,2} = namePIV;
            end
        end
    end
end

% %% sort files in directrory by width
px_sizeOP = .74*3;
for i=1:size(dirOP,1)
    % --------------------------ORIENT import ---------------------------------
    filepathOP = [dirOP(indX(i,1)).folder '\' dirOP(indX(i,1)).name];
    info = imfinfo(filepathOP); % Place path to file inside single quotes
    Nn = numel(info);
    Orient_width(i,1) = px_sizeOP*info(1).Width;
end
[~,idx] = sort(Orient_width);
Sorted_Orient_width = [idx,Orient_width(idx)];
end