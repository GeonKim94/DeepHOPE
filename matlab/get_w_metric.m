close all
clear
clc

addpath('/data01/gkim/Matlab_subcodes/gk/')
dir_data = '/data02/gkim/stem_cell_jwshin/data/230811+230502_MIPH5_wider_v3_allh_onRA';

nums_data = [];

cd(dir_data)
list_set = dir('*');
list_set = list_set(3:end);
for iter_set = 1:length(list_set)
    
    cd(dir_data)
    cd(list_set(iter_set).name)
    list_cls = dir('*_*');
    for iter_cls = 1:length(list_cls)
        cd(dir_data)
        cd(list_set(iter_set).name)
        cd(list_cls(iter_cls).name)
        list_h5 = dir('*.h5');

        nums_data(iter_set, iter_cls) = length(list_h5);

    end

end

nums_data