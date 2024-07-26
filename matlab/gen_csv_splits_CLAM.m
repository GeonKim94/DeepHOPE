clear
clc
close all

% Define the root directory containing the .h5 files

% Initialize cell arrays to store file names
trainFiles = {};
valFiles = {};
testFiles = {};



fileNames = {};
trainFlags = {};
valFlags = {};
testFlags = {};


% Traverse the directory structure

names_class = {'good_ipsc', 'bad_psc'};
counts_class = repmat(zeros(1,3), [length(names_class),1]);
%%

rootDir = '/data02/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_allh_onRA_pfeats/h5_files';

fileList = dir(fullfile(rootDir, '**', '*.h5'));
for i = 1:numel(fileList)
    % Get the relative path of the file
    relativePath = strrep(fileList(i).folder, rootDir, '');

    % Extract file name without extension
    [~, fileName, ~] = fileparts(fileList(i).name);
    
    % Categorize the file based on relative path
    if contains(relativePath, 'train', 'IgnoreCase', true)
        trainFiles{end+1} = fileName;
    elseif contains(relativePath, 'val', 'IgnoreCase', true)
        valFiles{end+1} = fileName;
    elseif contains(relativePath, 'test', 'IgnoreCase', true)
        testFiles{end+1} = fileName;
    end

    % Check if the file belongs to 'train', 'val', or 'test' category
    trainFlag = contains(relativePath, 'train', 'IgnoreCase', true);
    valFlag = contains(relativePath, 'val', 'IgnoreCase', true);
    testFlag = contains(relativePath, 'test', 'IgnoreCase', true);
    
    % Add file name and category flags to arrays
    fileNames{end+1} = fileName;
    trainFlags{end+1} = trainFlag;
    valFlags{end+1} = valFlag;
    testFlags{end+1} = testFlag;

    idx_class = 2;
    if contains(relativePath, '/05_', 'IgnoreCase', true)
        idx_class = 1;
    end
    idx_set = find([trainFlag valFlag testFlag]); 
    counts_class(idx_class, idx_set) = counts_class(idx_class, idx_set) + 1;

end

%%
rootDir = '/data02/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_testiPSC_pfeats/h5_files';

fileList = dir(fullfile(rootDir, '**', '*.h5'));
%%
for i = 1:numel(fileList)
    % Get the relative path of the file
    relativePath = strrep(fileList(i).folder, rootDir, '');

    % Extract file name without extension
    [~, fileName, ~] = fileparts(fileList(i).name);
    
    % Categorize the file based on relative path
    if contains(relativePath, 'train', 'IgnoreCase', true)
        trainFiles{end+1} = fileName;
    elseif contains(relativePath, 'val', 'IgnoreCase', true)
        valFiles{end+1} = fileName;
    elseif contains(relativePath, 'test', 'IgnoreCase', true)
        testFiles{end+1} = fileName;
    end

    % Check if the file belongs to 'train', 'val', or 'test' category
    trainFlag = contains(relativePath, 'train', 'IgnoreCase', true);
    valFlag = contains(relativePath, 'val', 'IgnoreCase', true);
    testFlag = contains(relativePath, 'test', 'IgnoreCase', true);
    
    % Add file name and category flags to arrays
    fileNames{end+1} = fileName;
    trainFlags{end+1} = trainFlag;
    valFlags{end+1} = valFlag;
    testFlags{end+1} = testFlag;

    idx_class = 2;
    if contains(relativePath, '/05_', 'IgnoreCase', true)
        idx_class = 1;
    end
    idx_set = find([trainFlag valFlag testFlag]); 
    counts_class(idx_class, idx_set) = counts_class(idx_class, idx_set) + 1;

end



%%


trainrat = length(trainFiles)/(length(trainFiles)+length(valFiles)+length(testFiles));

% Determine the maximum length among the arrays
maxLen = max([numel(trainFiles), numel(valFiles), numel(testFiles)]);


% Fill arrays to match maximum length
trainFiles = [trainFiles, repmat({''}, 1, maxLen - numel(trainFiles))];
valFiles = [valFiles, repmat({''}, 1, maxLen - numel(valFiles))];
testFiles = [testFiles, repmat({''}, 1, maxLen - numel(testFiles))];


dir_csv = sprintf('/data02/gkim/stem_cell_jwshin/src/python/CLAM_gkim/dataset_csv/task_3_good_psc_vs_bad_psc_%02d',uint8(floor(trainrat*100)));
mkdir(dir_csv)


% Write data to CSV file
T = table(trainFiles', valFiles', testFiles', 'VariableNames', {'train', 'val', 'test'});
fname_csv = 'splits_0.csv';
writetable(T, [dir_csv '/' fname_csv]);



trainFlags = cellfun(@(x) logical(x), trainFlags);
valFlags = cellfun(@(x) logical(x), valFlags);
testFlags = cellfun(@(x) logical(x), testFlags);
T = table(fileNames', trainFlags', valFlags', testFlags', ...
    'VariableNames', {'filename', 'train', 'val', 'test'});
fname_csv = 'splits_0_bool.csv';
writetable(T, [dir_csv '/' fname_csv]);


T = table(names_class', counts_class(:,1), counts_class(:,2), counts_class(:,3), ...
    'VariableNames', {'filename', 'train', 'val', 'test'});
fname_csv = 'splits_0_descriptor.csv';
writetable(T, [dir_csv '/' fname_csv]);
