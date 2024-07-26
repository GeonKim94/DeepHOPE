% Define the folder containing the subfolders with .h5 files
% folderPath = '/data02/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_allh_onRA_bags_dset/good_psc_vs_bad_psc_AE_features/pt_files';
sourceDir = '/data02/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_testiPSC_pfeats_dset/h5_files';
fileExtension='.h5';
destDir ='/data02/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_testiPSC_pfeats_dset/h5_files';
    % Ensure the source and destination directories are valid
    if ~isfolder(sourceDir)
        error('Source directory does not exist.');
    end
    if ~isfolder(destDir)
        mkdir(destDir);
    end

    % Append a wildcard to the file extension for filtering
    filePattern = fullfile(sourceDir, '**', ['*', fileExtension]);

    % Get the list of all files with the specified extension
    files = dir(filePattern);

    % Loop through each file and move it to the destination directory
    for i = 1:length(files)
        if ~files(i).isdir
            % Construct the full file path
            sourceFile = fullfile(files(i).folder, files(i).name);
            % Construct the destination file path
            destFile = fullfile(destDir, files(i).name);
            
            % Move the file
            movefile(sourceFile, destFile);
        end
    end

    disp(['All ', fileExtension, ' files have been moved successfully.']);