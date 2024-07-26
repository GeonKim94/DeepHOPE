close all
clear
clc

addpath('/data01/gkim/Matlab_subcodes/gk/')

% MIP <=> SEC1
% dir_data = '/data02/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_allh_onRA';
% dir_source= '/data02/gkim/stem_cell_jwshin/data/231118_SEC1H5_wider_v3';
% dir_data = '/data02/gkim/stem_cell_jwshin/data/230811+230502_SEC1H5_wider_v3_allh_onRA';
% dir_source= '/data02/gkim/stem_cell_jwshin/data/230502_SEC1H5_wider_v3';
dir_data = '/data02/gkim/stem_cell_jwshin/data/231222_MIPH5_wider_v3_allh_germ';
dir_source = '/data02/gkim/stem_cell_jwshin/data/231221_MIPH5_wider_v3';


cd(dir_source)
cd('00_train')

list_cls = dir('*_*');

for iter_cls = 1:length(list_cls)
    
    cd(dir_source)
    cd('00_train')
    dir_cls = (list_cls(iter_cls).name);
    cd(dir_cls)
    
    if any(strfind(dir_cls,'Ecto')) || any(strfind(dir_cls,'ecto'))
        str_cls = '02_ecto';
    elseif any(strfind(dir_cls,'Meso')) || any(strfind(dir_cls,'meso'))
        str_cls = '01_meso';
    elseif any(strfind(dir_cls,'Endo')) || any(strfind(dir_cls,'endo'))
        str_cls = '00_endo';
    elseif any(strfind(dir_cls,'untreated')) || any(strfind(dir_cls,'Untreated'))
        str_cls = '03_untreat';
    else
        error('no recognizable string sign for the class')
        % continue
    end

    list_h5 = [dir('*.h5'); dir('*/mat')];

    for iter_h5 = 1:length(list_h5)

        if mod(iter_h5,10) == 9
            str_set = '/test/';
        elseif mod(iter_h5,10) == 8
            str_set = '/val/';
        else
            str_set = '/train/';
        end

        str_set

        mkdir([dir_data str_set str_cls])

        movefile([dir_source '/00_train/' ...
            list_cls(iter_cls).name '/'...
            list_h5(iter_h5).name],...
            [dir_data str_set str_cls])
          
    end

end

%%

dir_data = '/data02/gkim/stem_cell_jwshin/data/231222_MIPH5_wider_v3_allh_germ';
dir_source = '/data02/gkim/stem_cell_jwshin/data/23_MIPH5_wider_v3_allh_onRA';

cd(dir_source)

list_set = [dir('test*');dir('train*');dir('val*')];

for iter_set = 1:length(list_set)
    
    cd(dir_source)
    dir_set = (list_set(iter_set).name);
    cd(dir_set)
   

    dir_cls ='01_high';
    str_cls = '03_untreat';
    
    cd(dir_cls)

    list_h5 = [dir('*.h5'); dir('*.mat')];

        
    mkdir([dir_data '/' dir_set '/' str_cls])

    for iter_h5 = 1:length(list_h5)

        if contains(list_h5(iter_h5).name, {'GM','231015','230921','230426','230719'})
            
        else
            continue
        end


        copyfile(list_h5(iter_h5).name,...
            [dir_data '/' dir_set '/' str_cls])
          
    end

end