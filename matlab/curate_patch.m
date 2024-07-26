dir_curate = '/data02/gkim/stem_cell_jwshin/data/220902_PNG_curated';
dir_source = '/data02/gkim/stem_cell_jwshin/data/220902_3D';
dir_save = '/data02/gkim/stem_cell_jwshin/data/220902_3D_curated';


cd(dir_curate);
dir_set = dir('0*');
for idx_set = 1:length(dir_set)
    cd(dir_curate);
    cd(dir_set(idx_set).name)
    dir_cls = dir('0*');
    for idx_cls = 1:length(dir_cls)
        cd(dir_curate);
        cd(dir_set(idx_set).name)
        cd(dir_cls(idx_cls).name)
        dir_img = dir('*.png');
        
        idxs_del = [];
        for idx_img = 1:length(dir_img)
            if contains(dir_img(idx_img).name, '00_colony')
                idxs_del = [idxs_del; idx_img];
            end
        end
        dir_img(idxs_del) = [];
        
        for idx_img = 1:length(dir_img)
            
            mkdir(dir_save)
            cd(dir_save)
            mkdir(dir_set(idx_set).name)
            cd(dir_set(idx_set).name)
            mkdir(dir_cls(idx_cls).name)
            
            
            cd(dir_curate);
            cd(dir_set(idx_set).name)
            cd(dir_cls(idx_cls).name)
            
            copyfile([dir_source '/' dir_set(idx_set).name '/' dir_cls(idx_cls).name '/' ...
                replace(dir_img(idx_img).name, '.png', '.mat')],...
                [dir_save '/' dir_set(idx_set).name '/' dir_cls(idx_cls).name '/' ...
                replace(dir_img(idx_img).name, '.png', '.mat')]);
        end
    end
end

%%

cd '/data02/gkim/stem_cell_jwshin/data/220902_PNG_curated/02_test/01_middle';
length(dir('*.png'))

cd '/data02/gkim/stem_cell_jwshin/data/220902_PNG/02_test/01_middle';
length(dir('*.png'))