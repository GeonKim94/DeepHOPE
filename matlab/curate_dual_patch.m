dir_curate = '/data02/gkim/stem_cell_jwshin/data/230306_PNG_jwedit';
dir_curate2 = '/data02/gkim/stem_cell_jwshin/data/230306_PNG';
dir_source = '/data02/gkim/stem_cell_jwshin/data/230306_3D';
dir_save = '/data02/gkim/stem_cell_jwshin/data/230306_3D_curated';


cd(dir_curate);
dir_set = dir('0*');
for idx_set = 1:length(dir_set)
    cd(dir_curate);
    cd(dir_set(idx_set).name)
    dir_col = dir('*A12_');
    for idx_col = 1:length(dir_col)

        cd(dir_curate2);
        cd(dir_set(idx_set).name)
        cd(dir_col(idx_col).name)
        dir2_img = dir('*.png');

        cd(dir_curate);
        cd(dir_set(idx_set).name)
        cd(dir_col(idx_col).name)
        dir_img = dir('*.png');
        
        idxs_del = [];
        for idx_img = 1:length(dir_img)
            if contains(dir_img(idx_img).name, '00_colony')
                idxs_del = [idxs_del; idx_img];
            end
        end
        dir_img(idxs_del) = [];
        
        for idx_img = 1:length(dir_img)

            if ~any(strcmp({dir2_img.name},dir_img(idx_img).name))
                continue
            end
            
            mkdir(dir_save)
            cd(dir_save)
            mkdir(dir_set(idx_set).name)
            cd(dir_set(idx_set).name)
            mkdir(dir_col(idx_col).name)
            
            copyfile([dir_source '/' dir_set(idx_set).name '/' dir_col(idx_col).name '/' ...
                replace(dir_img(idx_img).name, '.png', '.mat')],...
                [dir_save '/' dir_set(idx_set).name '/' dir_col(idx_col).name '/' ...
                replace(dir_img(idx_img).name, '.png', '.mat')]);
        end
    end
end
