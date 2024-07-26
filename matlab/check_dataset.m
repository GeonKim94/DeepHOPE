close all
clear
clc

addpath('/data01/gkim/Matlab_subcodes/gk/')

dir_data = '/data02/gkim/stem_cell_jwshin/data/23_MIPH5_wider_v3_allh_onRA';

cd(dir_data)

list_set = dir('*');
list_set = list_set(3:end);


nums_grp = [];
fnames = {};

for iter_set = 1:length(list_set)
    
    cd(dir_data)
    cd(list_set(iter_set).name)
    list_cls = dir('0*');

    for iter_cls = 1:length(list_cls)
           
        cd(dir_data)
        cd(list_set(iter_set).name)
        cd(list_cls(iter_cls).name)

        list_h5 = [dir('*.h5'); dir('*/mat')];


        list_h5(contains({list_h5.name},'230713')) = [];
        list_h5(contains({list_h5.name},'230427')) = [];
%         list_h5(contains({list_h5.name},'jax')) = [];
%         list_h5(contains({list_h5.name},'JAX')) = [];
%         list_h5(contains({list_h5.name},'Jax')) = [];

        nums_grp = [nums_grp length(list_h5)];

        for iter_h5 = 1:length(list_h5)
            fnames{end+1} = list_h5(iter_h5).name;
        end
    end

end
nums_grp

length(unique(fnames))

%%
dir_data = '/data02/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_allh_onRA';

cd(dir_data)

list_set = dir('*');
list_set = list_set(3:end);


nums_grp_ = [];
fnames_ = {};

for iter_set = 1:length(list_set)
    
    cd(dir_data)
    cd(list_set(iter_set).name)
    list_cls = dir('0*');

    for iter_cls = 1:length(list_cls)
           
        cd(dir_data)
        cd(list_set(iter_set).name)
        cd(list_cls(iter_cls).name)

        list_h5 = [dir('*.h5'); dir('*/mat')];
% 
        list_h5(contains({list_h5.name},'230713')) = [];
        list_h5(contains({list_h5.name},'230427')) = [];
%         list_h5(contains({list_h5.name},'jax')) = [];
%         list_h5(contains({list_h5.name},'JAX')) = [];
%         list_h5(contains({list_h5.name},'Jax')) = [];

        nums_grp_ = [nums_grp_ length(list_h5)];

        for iter_h5 = 1:length(list_h5)
            fnames_{end+1} = list_h5(iter_h5).name;
        end
    end

end
nums_grp_

length(unique(fnames_))

isequal(fnames,fnames_)