close all
clear
clc

addpath('/data01/gkim/Matlab_subcodes/gk/')

dir1 = '/data02/gkim/stem_cell_jwshin/data/230811+230502_MIPH5_wider_v3_allh_onRA/train/01_high';
dir2 = '/data02/gkim/stem_cell_jwshin/data/230811+230502_SEC1H5_wider_v3_allh_onRA/train/01_high';

cd(dir1)
list_h5 = dir('*.h5');


for iter_h5 = 1:length(list_h5)
    if ~exist([dir2, '/', list_h5(iter_h5).name])
        pause()
%     [dir_find, found] = search_recursive(dir2, list_h5(iter_h5).name, false);
%     if strcmp(found,'none')
%         break
%     end
    end
end