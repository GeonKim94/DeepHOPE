% check size
% dataset division is 8:1:1
% for processing standard colony data (230323 + 230407)
% v2 difference: center-prioritizing crop to be curated (+0306 margin 15->20 fixed)

clear
clc
close all


addpath('/data01/gkim/Matlab_subcodes/gk')
addpath('/data01/gkim/Matlab_subcodes')

dir_save = '/data02/gkim/stem_cell_jwshin/data/230811_MIPH5_wider_v2'; 
dir_cur = '/data02/gkim/stem_cell_jwshin/data/230811_MIPPNGcurate_wider_v2';

cd(dir_save)
dir_set = dir('*00_train');

size_pad = 20; 

hours = 0;%0:1:24;

for hour = hours

for iter_set = length(dir_set):-1:1
    cd(dir_save)
    cd(dir_set(iter_set).name)
    dir_cls = dir('*_*');
    
    if iter_set == 3
        stride = 0.5;%1;
    else
        stride = 0.5;
    end
    
    for iter_cls = 1:length(dir_cls)%1:length(dir_cls)
        
        cd(dir_save)
        cd(dir_set(iter_set).name)
        cd(dir_cls(iter_cls).name)
        dir_h5 = dir('23*.h5');
        for iter_stitch = length(dir_h5):-1:1
            
            iter_x = 0;
            iter_y = 0;

            cd(dir_save)
            cd(dir_set(iter_set).name)
            cd(dir_cls(iter_cls).name)
            
            
            fname = dir_h5(iter_stitch).name;
            idx_dot = strfind(fname, '.');
            fname_save = fname(1:idx_dot(end)-1);
            %% comment this if you're fixing the data for the above condition
%             if exist([dir_img, '/', dir_set(iter_set).name, '/',...
%                 dir_cls(iter_cls).name, sprintf('_hr%02d', hour), '/', '00_colony_' fname_save, '.png'])
%                 'skipping... colony data already exist'
%                 continue
%             end
            %%

            dir_file = [dir_save '/'...
            dir_set(iter_set).name '/'...
            dir_cls(iter_cls).name '/'...
            fname];

            dir_search = [dir_cur '/'...
            dir_set(iter_set).name '/'...
            dir_cls(iter_cls).name '/'];
            [dir_find,pat_find] = find_similfile(dir_file,dir_search);

            if ~isfile(dir_find)
                dir_move = [dir_save '/'...
                    dir_set(iter_set).name '_bad/'...
                    dir_cls(iter_cls).name '/'];
                mkdir(dir_move)
                movefile(dir_file,dir_move)
            end
        end
    end
    
end

end