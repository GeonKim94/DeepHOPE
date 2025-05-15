function filePaths = findFilesWithPattern(dirPath, pattern)
    filePaths = {};

    files = dir(dirPath);

    for i = 1:length(files)
        name = files(i).name;
        
        if strcmp(name, '.') || strcmp(name, '..')
            continue;
        end
        
        fullPath = fullfile(dirPath, name);
        
        if files(i).isdir
            subDirFiles = findFilesWithPattern(fullPath, pattern);
            filePaths = [filePaths; subDirFiles];
        else
            if contains(name, pattern)
                filePaths{end + 1} = fullPath;
            end
        end
    end
end