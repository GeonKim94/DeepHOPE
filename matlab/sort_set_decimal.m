close all
clear
clc

addpath('/data01/gkim/Matlab_subcodes/gk/')

% MIP <=> SEC1
% dir_data = '/data02/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_allh_onRA';
% dir_source= '/data02/gkim/stem_cell_jwshin/data/231118_SEC1H5_wider_v3';
% dir_data = '/data02/gkim/stem_cell_jwshin/data/230811+230502_SEC1H5_wider_v3_allh_onRA';
% dir_source= '/data02/gkim/stem_cell_jwshin/data/230502_SEC1H5_wider_v3';
dir_data = '/data01/gkim/stem_cell_jwshin/data/231229_SEC1H5_wider_v3';
dir_source = '/data01/gkim/stem_cell_jwshin/data/231229_SEC1H5_wider_v3';


cd(dir_source)
cd('00_train')

list_cls = dir('*_*');

for iter_cls = 1:length(list_cls)
    
    cd(dir_source)
    cd('00_train')
    dir_cls = (list_cls(iter_cls).name);
    cd(dir_cls)
    
    str_cls = dir_cls;
% %     if any(strfind(dir_cls,'24h')) || any(strfind(dir_cls,'24H'))
% %         str_cls = '00_low';
% %     elseif any(strfind(dir_cls,'12h')) || any(strfind(dir_cls,'12H'))
% %         str_cls = '00_low';
% %     elseif any(strfind(dir_cls,'untreated')) || any(strfind(dir_cls,'Untreated'))
% %         str_cls = '01_high';
% %     else
% %         error('no recognizable string sign for the class')
% %         % continue
% %     end

    list_h5 = [dir('*.h5'); dir('*.mat')];

    for iter_h5 = 1:length(list_h5)

        if mod(iter_h5,10) == 9
            str_set = '/test/';
        elseif mod(iter_h5,10) == 8
            str_set = '/val/';
        else
            str_set = '/train/';
        end

        str_set;

        mkdir([dir_data str_set str_cls]);

        movefile([dir_source '/00_train/' ...
            list_cls(iter_cls).name '/'...
            list_h5(iter_h5).name],...
            [dir_data str_set str_cls])
          
    end

end

