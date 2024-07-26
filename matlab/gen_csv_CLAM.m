% Define the directory where your .h5 files are stored
dataDir = '/data02/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_allh_onRA_bags_dset/good_psc_vs_bad_psc_AE_features/h5_files';

% Initialize variables to store data
caseIDs = {};
slideIDs = {};
labels = {};

% List all subdirectories in the main directory
subdirs = dir(dataDir);
subdirs = subdirs([subdirs.isdir]); % Keep only directories
subdirs = subdirs(~ismember({subdirs.name}, {'.', '..'})); % Remove '.' and '..' directories

% Iterate through each subdirectory
for i = 1:numel(subdirs)
    subdirPath = fullfile(dataDir, subdirs(i).name);
    subsubdirs = dir(subdirPath);
    subsubdirs = subsubdirs([subsubdirs.isdir]); % Keep only directories
    subsubdirs = subsubdirs(~ismember({subsubdirs.name}, {'.', '..'})); % Remove '.' and '..' directories
    
    % Iterate through each sub-subdirectory
    for j = 1:numel(subsubdirs)
        subsubdirPath = fullfile(subdirPath, subsubdirs(j).name);
        % List all .h5 files in the current sub-subdirectory
        h5Files = dir(fullfile(subsubdirPath, '*.h5'));
        
        % Iterate through each .h5 file
        for k = 1:numel(h5Files)
            % Extract case_id from the second-level directory name
            caseID = subsubdirs(j).name;
            % Extract slide_id from the file name without extension
            [~, fileName, ~] = fileparts(h5Files(k).name);
            slideID = fileName;
            % Determine label based on case_id
            if strcmp(caseID(1:2), '05')
                label = 'good_psc';
            else
                label = 'bad_psc';
            end
            
            % Append data to lists
            caseIDs{end+1} = caseID;
            slideIDs{end+1} = slideID;
            labels{end+1} = label;
        end
    end
end

% Create table
dataTable = table(caseIDs', slideIDs', labels', 'VariableNames', {'case_id', 'slide_id', 'label'});

% Display table
disp(dataTable);

csvFileName = 'good_psc_vs_bad_psc_dummy_clean.csv';
writetable(dataTable, csvFileName);

disp(['Table saved to: ' csvFileName]);