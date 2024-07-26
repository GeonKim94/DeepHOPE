% Define the folder containing the subfolders with .h5 files
% folderPath = '/data02/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_allh_onRA_bags_dset/good_psc_vs_bad_psc_AE_features/pt_files';
folderPath = '/data02/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_testiPSCs_pfeats_dset/pt_files';

% Get a list of all subdirectories in the specified folder
subDirs = dir(fullfile(folderPath, '**', ''));
subDirs = subDirs([subDirs.isdir]);
%subDirs = unique(subDirs);
subDirs = subDirs(~[strcmp({subDirs.name}, {'.'})]);
subDirs = subDirs(~[strcmp({subDirs.name}, {'..'})]);

% Loop through each subdirectory
for i = 1:numel(subDirs)
    subDirPath = fullfile(subDirs(i).folder, subDirs(i).name);
    
    % Get a list of .h5 files in the current subdirectory
    h5Files = dir(fullfile(subDirPath, '*.pt'));
    
    % Move each .h5 file to the main folder
    for j = 1:numel(h5Files)
        h5FilePath = fullfile(h5Files(j).folder, h5Files(j).name);
        movefile(h5FilePath, folderPath);
    end
    
    % Remove the now empty subdirectory
    rmdir(subDirPath);
end

disp('All .h5 files moved to the main folder.');
