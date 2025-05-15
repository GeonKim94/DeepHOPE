function filePath = findFirstFileWithPattern(dirPath, pattern)
    filePath = '';
    
    files = dir(dirPath);

    for i = 1:length(files)
        name = files(i).name;
        
        if strcmp(name, '.') || strcmp(name, '..')
            continue;
        end
        
        fullPath = fullfile(dirPath, name);
        
        if files(i).isdir
            filePath = findFirstFileWithPattern(fullPath, pattern);
            if ~isempty(filePath)
                return;
            end
        else
            if contains(name, pattern)
                filePath = fullPath;
                return;
            end
        end
    end
end